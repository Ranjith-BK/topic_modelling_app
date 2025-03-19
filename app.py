import streamlit as st
from streamlit_card import card
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import ast
import numpy as np
# Load environment variables
load_dotenv()

# Get MongoDB connection string from .env file
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["sparzaai"]  # Database name
collection = db["ticket-copy"]  # Collection name
collection_clusters = db["clusters"]
dummy_collection = db["dummy"]

# Set page layout
st.set_page_config(layout="wide")

# Load Data Function
@st.cache_data
def load_support_data():
    try:
        # Fetch only records where aem_alerts and aem are False
        query = {"aem": False}
        data = list(collection.find(query, {"_id": 0}))  # Exclude `_id` field
        
        # Convert to DataFrame if data exists
        if data:
            df = pd.DataFrame(data)
            df.columns = df.columns.str.strip()
            return df
        else:
            st.warning("No records found where 'aem_alerts' and 'aem' are both False.")
            return pd.DataFrame()  # Return empty DataFrame if no data found

    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return pd.DataFrame()

@st.cache_data
def load_support_clusters():
    fields = {
        "cluster_fields.primary_clusters": 1,
        "cluster_fields.primary_cluster_labels": 1,
        "cluster_fields.subclusters": 1,
        "cluster_fields.subcluster_labels": 1,
        "data_type": 1,
        "_id": 0
    }
    data = list(collection_clusters.find({"data_type": "ticket-copy"}, fields))
    
    # Ensure each entry is a dictionary
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return pd.DataFrame(data)
    else:
        st.error("Data format error: Expected list of dictionaries.")
        return pd.DataFrame()  # Return empty DataFrame if structure is incorrect


def process_primary_clusters(data):
    primary_data = []
    for entry in data.to_dict("records"):
        if not isinstance(entry, dict):  # Ensure entry is a dictionary
            continue
        
        cluster_fields = entry.get("cluster_fields", {})
        primary_clusters = cluster_fields.get("primary_clusters", {})
        cluster_labels = cluster_fields.get("primary_cluster_labels", {})

        for cluster_id, keywords in primary_clusters.items():
            label = cluster_labels.get(str(cluster_id), f"Cluster {cluster_id}")
            primary_data.append({
                "primary_cluster": f"primary {cluster_id}",
                "keywords": ", ".join(keywords) if isinstance(keywords, list) else "",
                "label_generated": label,
            })
    return pd.DataFrame(primary_data)

def process_subclusters(data):
    subcluster_data = []

    for entry in data.to_dict("records"):
        if not isinstance(entry, dict):  # Ensure entry is a dictionary
            continue
        
        cluster_fields = entry.get("cluster_fields", {})
        subclusters = cluster_fields.get("subclusters", [])
        subcluster_labels = cluster_fields.get("subcluster_labels", [])

        subcluster_labels_map = {}
        for sub in subcluster_labels:
            if isinstance(sub, dict):
                primary_cluster_id = sub.get("primary_cluster_id", "-1")
                subcluster_labels_map[primary_cluster_id] = sub.get("subcluster_labels", {})

        for sub in subclusters:
            if not isinstance(sub, dict):
                continue  
            primary_cluster_id = sub.get("primary_cluster_id", "-1")
            subcluster_dict = sub.get("subclusters", {})

            if not isinstance(subcluster_dict, dict):
                continue  
            labels_dict = subcluster_labels_map.get(primary_cluster_id, {})

            for subcluster_id, keywords in subcluster_dict.items():
                label = labels_dict.get(subcluster_id, f"Subcluster {subcluster_id}")
                subcluster_data.append({
                    "primary_cluster": f"primary {primary_cluster_id}",
                    "subcluster": subcluster_id,
                    "sub_keywords": ", ".join(keywords) if isinstance(keywords, list) else "",
                    "label_generated": label
                })

    return pd.DataFrame(subcluster_data)

def update_manual_label(cluster_id, new_label):
    dummy_collection.update_one(
        {"primary_cluster": cluster_id},
        {"$set": {"primary_manual_label": new_label}},
        upsert=True
    )


# Function to update page state & rerun
def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()


def home_page():
    st.markdown("<h1 style='text-align: center;'>Topic Modelling Using LLM</h1>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if card("Tickets", "Click to explore Tickets"):
            set_page("tickets_home")  # ✅ Updates session state and reruns

    with col2:
        if card("Emails", "Click to explore Emails"):
            set_page("emails")

    with col3:
        if card("Messages", "Click to explore Messages"):
            set_page("messages")

    with col4:
        if card("Group Messages", "Click to explore Group Messages"):
            set_page("group_messages")
# Tickets Home Page Function
def tickets_home_page():
    st.markdown("<h1 style='text-align: center;'>Welcome to the Ticket Analysis Dashboard</h1>", unsafe_allow_html=True)

    st.markdown("""
    <h2>Overview</h2>
    <ul style="font-size:18px;">
        This system leverages <b>LLM-based keyphrase extraction</b>, <b>HDBSCAN clustering</b>, and <b>manual validation</b> to analyze and categorize IT alert and support tickets efficiently. By combining <b>AI-powered processing</b> with <b>human insights</b>, it enhances issue resolution and automates categorization.
    </ul>
    """, unsafe_allow_html=True)

    # ✅ Key Features Section
    st.markdown("""
    ## Key Features:
    - *AI-Powered Keyphrase Extraction:* Extracts relevant terms from ticket descriptions using LLM models.
    - *Smart Clustering:* Groups related tickets using HDBSCAN to identify recurring themes.
    - *Outlier Handling:* Detects and processes unusual or rare tickets separately.
    - *Automated & Manual Labeling:* Uses AI-generated labels, refined through human verification, for precise categorization.
    """)

    # Ticket Category Cards
    col1, col2 = st.columns(2)
    
    with col1:
        if card("🚨 Alert Tickets", "View and analyze alert tickets."):
            set_page("alert_tickets")

    with col2:
        if card("📂 support Tickets", "View and analyze support tickets."):
            set_page("support_tickets")

    # Display Image
    st.image("Diagram 1.png", caption="System Overview", use_container_width=True)

    # Back Button
    if st.button("⬅️ Back to Home"):
        set_page("home")

def support_tickets_page():
    st.sidebar.title("🔍 support Tickets")
    option = st.sidebar.radio("", ["support Tickets", "LLM Label Generator", "Topic Analysis", "manual analysis"])

    if option == "support Tickets":
        st.title("📂 support Tickets")
        support_ticket_home()
        
    elif option == "LLM Label Generator":
        st.title("🤖 LLM Label Generator")
        st.write("Generate AI-powered labels for ticket categorization.")
        #support_label_generator_page()

    elif option == "Topic Analysis":
        st.title("📄 Topic Analysis")
        support_topic_analysis()

    elif option == "manual analysis":
        manual_analysis()

    st.sidebar.markdown("---")
    if st.sidebar.button("⬅️ Tickets"):
        set_page("tickets_home")

def support_ticket_home():
    st.subheader("Basic Statistics")

    df = load_support_data()
    df_clusters = load_support_clusters()
    
    # Load processed clusters
    primary_clusters_df = process_primary_clusters(df_clusters)
    subclusters_df = process_subclusters(df_clusters)

    # Calculate additional statistics
    total_primary_labels = primary_clusters_df['label_generated'].nunique()
    total_secondary_clusters = subclusters_df['label_generated'].nunique()

    # Convert 'llm_subtopics' strings to proper lists if needed
    df['llm_subtopics'] = df['llm_subtopics'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['llm_subtopics'] = df['llm_subtopics'].apply(lambda x: x if isinstance(x, list) else [])

    # Unique llm_dominant_topic with ticket count
    dominant_topic_counts = df['llm_dominant_topic'].value_counts().reset_index()
    dominant_topic_counts.columns = ['llm_dominant_topic', 'ticket_count']

    # Unique llm_subtopics with ticket count
    subtopic_counts = pd.DataFrame(df['llm_subtopics'].sum(), columns=['llm_subtopics'])
    subtopic_counts = subtopic_counts['llm_subtopics'].value_counts().reset_index()
    subtopic_counts.columns = ['llm_subtopics', 'ticket_count']

    # Define data for display
    data_size = len(df)
    data_with_keyphrases = df["extracted_keywords"].notna().sum()
    data_without_keyphrases = data_size - data_with_keyphrases
    total_keyphrases = df["extracted_keywords"].dropna().apply(lambda x: len(x.split(","))).sum()
    total_filtered_keyphrases = df["filtered_keywords"].dropna().apply(lambda x: len(x) if isinstance(x, list) else len(x.split(","))).sum()
    unique_filtered_keyphrases = len(
        set(
            ",".join(
                [".".join(x) if isinstance(x, list) else str(x) for x in df["filtered_keywords"].dropna() ]
            ).split(",")
        )
    )

    # Convert to DataFrame for display
    data = {
        "Metric": [
            "Data Size",
            "Data with Keyphrases",
            "Data without Keyphrases",
            "Total no of Keyphrases",
            "Keyphrases After Filtering",
            "Unique Keyphrases After Filtering"
        ],
        "General Tickets": [
            data_size,
            data_with_keyphrases,
            data_without_keyphrases,
            total_keyphrases,
            total_filtered_keyphrases,
            unique_filtered_keyphrases
        ]
    }

    df_summary = pd.DataFrame(data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    st.markdown(f"**Total Primary Labels (Including Outlier):** {total_primary_labels}")
    st.markdown(f"**Total Secondary Labels (Including Outlier):** {total_secondary_clusters}")

    if "llm_dominant_topic" in df.columns and not df.empty:
        # Standardizing values to avoid mismatch
        df["llm_dominant_topic"] = df["llm_dominant_topic"].astype(str).str.strip().str.lower()

        # Remove NaN values from the 'llm_dominant_topic' column
        df_cleaned = df.dropna(subset=["llm_dominant_topic"])

        # Get list of labels and their frequencies
        labels = df_cleaned["llm_dominant_topic"].tolist()
        label_freq = Counter(labels)

        # Remove any NaN or empty string from the frequency counter
        label_freq = {k: v for k, v in label_freq.items() if k and k != "nan"}

        # If there are no valid topics, do not generate the word cloud
        if label_freq:
            primary_wordcloud = WordCloud(
                width=600, height=400, background_color='white', colormap='viridis', 
                prefer_horizontal=1.0, relative_scaling=0.5
            ).generate_from_frequencies(label_freq)

            st.subheader("Dominant Topic")
            st.image(primary_wordcloud.to_array(), use_container_width=False, width=600)
        else:
            st.warning("No valid dominant topics available for the word cloud.")
        
        unique_labels = sorted(set(labels))
        selected_label = st.selectbox("Select a Topic Label:", ["Select"] + unique_labels)
        
        if selected_label != "Select":
            # Ensure consistency when fetching the count
            selected_label_cleaned = selected_label.strip().lower()
            count = label_freq.get(selected_label_cleaned, 0)
            st.markdown(f"**{selected_label}: {count} records**")

            filtered_df = df[df["llm_dominant_topic"] == selected_label_cleaned]

            if "llm_subtopics" in filtered_df.columns:
                subtopics_list = filtered_df["llm_subtopics"].dropna().tolist()

                all_subtopics = []
                for sub in subtopics_list:
                    if isinstance(sub, list):
                        all_subtopics.extend(sub)
                    else:
                        all_subtopics.append(sub)

                subtopic_freq = Counter(all_subtopics)
                num_subtopics = len(subtopic_freq)

                if subtopic_freq:
                    st.markdown(f"**Number of Subtopics for '{selected_label}':** {num_subtopics}")

                    subtopic_df = pd.DataFrame(subtopic_freq.items(), columns=["Subtopic", "Number of Records"])
                    subtopic_df = subtopic_df.sort_values(by="Number of Records", ascending=False)

                    st.subheader("Subtopic Distribution")
                    st.dataframe(subtopic_df, use_container_width=True, hide_index=True)

                    subtopic_wordcloud = WordCloud(
                        width=600, height=400, background_color='white', colormap='plasma', 
                        prefer_horizontal=1.0, relative_scaling=0.5
                    ).generate_from_frequencies(subtopic_freq)

                    st.subheader("Subtopics Word Cloud")
                    st.image(subtopic_wordcloud.to_array(), use_container_width=False, width=600)
                else:
                    st.warning(f"No subtopics found for '{selected_label}'.")
            else:
                st.warning("No 'llm_subtopics' column found in the dataset.")

    else:
        st.warning("No valid 'llm_dominant_topic' column found in the dataset.")

def support_topic_analysis():
    """Main function for displaying and filtering support tickets with dynamic topic and subtopic filtering."""
    
    # Load data
    if "df" not in st.session_state:
        st.session_state.df = load_support_data()
    
    df = st.session_state.df

    if df is None or df.empty:
        st.error("Error: The dataset is empty or could not be loaded.")
        return
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Ensure required columns exist
    required_columns = {"ticket_number", "title", "description", "llm_dominant_topic", "llm_subtopics"}
    if not required_columns.issubset(df.columns):
        st.error(f"Error: Required columns {required_columns} not found in the dataset.")
        return
    
    # Extract unique dominant topics and subtopics
    topic_subtopic_map = {}
    subtopic_topic_map = {}
    
    for _, row in df.iterrows():
        dominant_topics = row["llm_dominant_topic"]
        subtopics = row["llm_subtopics"]

        # Convert dominant topics safely
        if isinstance(dominant_topics, list):
            topics_list = dominant_topics
        elif isinstance(dominant_topics, str):
            topics_list = [s.strip() for s in dominant_topics.split(",")] if dominant_topics else []
        else:
            topics_list = []

        # Convert subtopics safely
        if isinstance(subtopics, list):
            subtopics_list = subtopics  # Already a list
        elif isinstance(subtopics, str):
            try:
                subtopics_list = ast.literal_eval(subtopics)  # Convert string list to actual list
                if not isinstance(subtopics_list, list):
                    subtopics_list = [subtopics_list]
            except (SyntaxError, ValueError):
                subtopics_list = [s.strip() for s in subtopics.split(",")] if subtopics else []
        else:
            subtopics_list = []

        # Build mappings
        for topic in topics_list:
            topic_subtopic_map.setdefault(topic, set()).update(subtopics_list)
        
        for subtopic in subtopics_list:
            subtopic_topic_map.setdefault(subtopic, set()).update(topics_list)
    
    # Streamlit UI for topic and subtopic selection
    col1, col2 = st.columns(2)
    
    with col1:  # Dominant Topic on the left
        selected_dominant_topic = st.selectbox(
            "Select a Dominant Topic:",
            ["Select a topic"] + sorted(topic_subtopic_map.keys()),
            key="dominant_topic_selectbox"
        )
    
    # Dynamically update available subtopics based on dominant topic selection
    if selected_dominant_topic != "Select a topic":
        available_subtopics = sorted(topic_subtopic_map.get(selected_dominant_topic, []))
    else:
        available_subtopics = sorted(subtopic_topic_map.keys())
    
    with col2:  # Subtopic on the right
        selected_subtopics = st.multiselect(
            "Select Subtopics:",
            options=["Select a subtopic"] + available_subtopics,
            key="subtopic_multiselect",
            default=[]
        )
    
    # If the user selects a subtopic, filter the dominant topics that are associated with that subtopic
    if selected_subtopics:
        available_dominant_topics = set()
        for subtopic in selected_subtopics:
            available_dominant_topics.update(subtopic_topic_map.get(subtopic, []))
    else:
        available_dominant_topics = set(topic_subtopic_map.keys())
    
    # Now, dynamically update the Dominant Topic dropdown based on selected subtopics (if any)
    if selected_subtopics:
        selected_dominant_topic = st.selectbox(
            "Select a Dominant Topic for the subtopic:",
            ["Select a topic"] + sorted(available_dominant_topics),
            key="dominant_topic_filtered_selectbox",
            index=0 if selected_dominant_topic == "Select a topic" else sorted(available_dominant_topics).index(selected_dominant_topic)
        )

    # Filter logic: Exclude records where dominant topic or subtopics are NaN
    if selected_dominant_topic == "Select a topic" and not selected_subtopics:
        st.info("Please select a dominant topic or subtopic to display records.")
        return
    
    filtered_df = df.copy()

    # Filter out rows where dominant topic or subtopics are NaN
    filtered_df = filtered_df[filtered_df["llm_dominant_topic"].notna() & filtered_df["llm_subtopics"].notna()]
    
    # Apply dominant topic filter
    if selected_dominant_topic != "Select a topic":
        filtered_df = filtered_df[filtered_df["llm_dominant_topic"].astype(str).apply(
            lambda x: selected_dominant_topic.strip() == x.strip()
        )]

    # Apply subtopic filter
    if selected_subtopics and selected_subtopics != ["Select a subtopic"]:
        filtered_df = filtered_df[filtered_df["llm_subtopics"].fillna("").astype(str).apply(
            lambda x: any(sub.strip() in [s.strip() for s in ast.literal_eval(x)] if isinstance(x, str) and x.startswith("[") else x.split(",") for sub in selected_subtopics)
        )]

    st.write(f"**Found {len(filtered_df)} matching records**")
    
    # Pagination setup
    records_per_page = 10
    total_records = len(filtered_df)
    total_pages = (total_records // records_per_page) + (1 if total_records % records_per_page > 0 else 0)
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = 0
    
    current_page = st.session_state.current_page
    start_idx = current_page * records_per_page
    end_idx = start_idx + records_per_page
    page_data = filtered_df.iloc[start_idx:end_idx]
    
    # Display records
    for _, row in page_data.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            col1.markdown(f"**{row['ticket_number']}**")
            col2.markdown(f"**{row['title']}**")
            if col3.button("🔍 View Details", key=f"view_{row['ticket_number']}"):
                st.session_state.clicked_ticket_number = row['ticket_number']
                st.session_state.show_modal = True
       
    # Pagination controls
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if current_page > 0:
                if st.button("⬅️ Previous"):
                    st.session_state.current_page -= 1
                    st.rerun()
        with col2:
            st.markdown(f"<div style='text-align: center; font-weight: bold;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
        with col3:
            if current_page < total_pages - 1:
                if st.button("Next ➡️"):
                    st.session_state.current_page += 1
                    st.rerun()
    
    # Handle modal (view details)
    if st.session_state.get("show_modal", False) and st.session_state.get("clicked_ticket_number") is not None:
        clicked_ticket_number = st.session_state.clicked_ticket_number
        clicked_ticket = df[df['ticket_number'] == clicked_ticket_number]
        
        if not clicked_ticket.empty:
            selected_ticket = clicked_ticket.iloc[0]
            
            with st.expander("📄 Ticket Details", expanded=True):
                st.write(f"### Ticket Number: {selected_ticket['ticket_number']}")
                st.text_area("Ticket Description:", selected_ticket.get("description", "No description available"), height=200)

                # Check for NaN in dominant topic and subtopics
                dominant_topic = selected_ticket.get("llm_dominant_topic", "N/A")
                subtopics = selected_ticket.get("llm_subtopics", "N/A")
                
                st.write(f"**Dominant Topic:** {dominant_topic if pd.notna(dominant_topic) else 'N/A'}")
                
                # Check if subtopics is a list or string
                if isinstance(subtopics, list):
                    st.write(f"**Subtopics:** {', '.join(subtopics) if subtopics else 'N/A'}")
                elif isinstance(subtopics, str):
                    st.write(f"**Subtopics:** {subtopics if subtopics else 'N/A'}")
                else:
                    st.write(f"**Subtopics:** {'N/A' if pd.isna(subtopics) else subtopics}")
                
                if st.button("❌ Close"):
                    st.session_state.show_modal = False
                    st.session_state.clicked_ticket_number = None
                    st.experimental_rerun()

def manual_analysis():
    st.title("Manual Analysis of Clusters")
    data = load_support_clusters()

    if data.empty:
        st.warning("No data available for analysis.")
        return
    
    selected_tab = st.radio("Select Analysis Type:", ["Select Analysis Type", "Primary Clusters", "Secondary Clusters", "Secondary Outliers", "Outlier Phrases"])
    
    # Initialize session state for manual labels if not already present
    if "manual_labels" not in st.session_state:
        st.session_state.manual_labels = {}

    # Process primary clusters and store mapping {primary_cluster_id: primary_label}
    primary_df = process_primary_clusters(data)
    primary_labels_map = dict(zip(primary_df["primary_cluster"], primary_df["label_generated"]))

    # Primary Clusters
    if selected_tab == "Primary Clusters":
        primary_df = primary_df[~primary_df["primary_cluster"].isin(["primary -1", "primary 0"])]  # Exclude primary -1 and primary 0
        
        st.write("### Primary Clusters Table")

        for index, row in primary_df.iterrows():
            col1, col2, col3, col4, col5 = st.columns([2, 3, 4, 3, 2])  # Adjusted column widths
            with col1:
                st.text(f"{row['primary_cluster']}")  # Show the primary cluster number
            with col2:
                st.text(f"{row['label_generated']}")  # Show the primary label
            with col3:
                st.text(f"Keywords: {row['keywords']}")
            
            existing_labels = st.session_state.manual_labels.get(row['primary_cluster'], [])
            
            with col4:
                new_label = st.text_input(f"Enter label for {row['primary_cluster']}", value=", ".join(existing_labels), key=f"label_{index}")
            
            with col5:
                if st.button("Save", key=f"save_{index}") and new_label:
                    new_labels = list(set(existing_labels + [label.strip() for label in new_label.split(',')]))

                    st.session_state.manual_labels[row['primary_cluster']] = new_labels
                    update_manual_label(row['primary_cluster'], new_labels)  
                    st.success(f"Labels for {row['primary_cluster']} updated successfully.")

    # Secondary Clusters
    if selected_tab == "Secondary Clusters":
        subcluster_df = process_subclusters(data)
        
        # Exclude primary -1 and primary 0 from secondary clusters as well
        subcluster_df = subcluster_df[~subcluster_df["primary_cluster"].isin(["primary -1", "primary 0"])]

        st.write("### Secondary Clusters Table")

        for index, row in subcluster_df.iterrows():
            primary_label = primary_labels_map.get(row["primary_cluster"], row["primary_cluster"])  # Get label or fallback to original

            # Adjusted column widths for better alignment
            col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 4, 3, 3, 2])  
            
            with col1:
                st.text(f"{primary_label}")  # Show the mapped primary label
            with col2:
                st.text(f"{row['subcluster']}")
            with col3:
                st.text(f"{row['sub_keywords']}")
            with col4:
                st.text(f"{row['label_generated']}")

            existing_labels = st.session_state.manual_labels.get(row['subcluster'], [])

            with col5:
                new_label = st.text_input(f"Label for {row['subcluster']}", value=", ".join(existing_labels), key=f"label_sec_{index}")

            with col6:
                if st.button("Save", key=f"save_sec_{index}") and new_label:
                    new_labels = list(set(existing_labels + [label.strip() for label in new_label.split(',')]))

                    st.session_state.manual_labels[row['subcluster']] = new_labels
                    update_manual_label(row['subcluster'], new_labels)  

                    st.success(f"Labels for {row['subcluster']} updated successfully.")

    # Secondary Outliers (Primary -1 Subclusters Only)
    if selected_tab == "Secondary Outliers":
        outliers_df = process_subclusters(data)
        
        # Filter only records where primary cluster = "primary -1"
        outliers_df = outliers_df[outliers_df["primary_cluster"] == "primary -1"]

        st.write("### Secondary Outliers Table (Primary -1)")

        for index, row in outliers_df.iterrows():
            col1, col2, col3, col4 = st.columns([2, 4, 4, 2])  # Adjusted column widths
            
            with col1:
                st.text(f"{row['subcluster']}")  # Show subcluster ID
            with col2:
                st.text(f"{row['sub_keywords']}")  # Display keywords
            with col3:
                st.text(f"{row['label_generated']}")  # Show existing labels
            
            existing_labels = st.session_state.manual_labels.get(row['subcluster'], [])

            with col4:
                # Save button and text input placed together in the same column
                new_label = st.text_input(f"Label for {row['subcluster']}", value=", ".join(existing_labels), key=f"label_outlier_{index}")
                if st.button("Save", key=f"save_outlier_{index}") and new_label:
                    new_labels = list(set(existing_labels + [label.strip() for label in new_label.split(',')]))

                    st.session_state.manual_labels[row['subcluster']] = new_labels
                    update_manual_label(row['subcluster'], new_labels)  

                    st.success(f"Labels for {row['subcluster']} updated successfully.")
        
# Secondary Outliers (Primary -1 Subclusters Only)
    if selected_tab == "Secondary Outliers":
        outliers_df = process_subclusters(data)
        
        # Filter only records where primary cluster = "primary -1"
        outliers_df = outliers_df[outliers_df["primary_cluster"] == "primary -1"]

        st.write("### Secondary Outliers Table (Primary -1)")

        for index, row in outliers_df.iterrows():
            col1, col2, col3, col4 = st.columns([2, 4, 4, 2])  # Adjusted column widths
            
            with col1:
                st.text(f"{row['subcluster']}")  # Show subcluster ID
            with col2:
                st.text(f"{row['sub_keywords']}")  # Display keywords
            with col3:
                st.text(f"{row['label_generated']}")  # Show existing labels
            
            existing_labels = st.session_state.manual_labels.get(row['subcluster'], [])

            with col4:
                # Save button and text input placed together in the same column
                new_label = st.text_input(f"Label for {row['subcluster']}", value=", ".join(existing_labels), key=f"label_outlier_{index}")
                if st.button("Save", key=f"save_outlier_{index}") and new_label:
                    new_labels = list(set(existing_labels + [label.strip() for label in new_label.split(',')]))

                    st.session_state.manual_labels[row['subcluster']] = new_labels
                    update_manual_label(row['subcluster'], new_labels)  

                    st.success(f"Labels for {row['subcluster']} updated successfully.")
        
    # Outlier Phrases (primary -1 and subcluster -1) from Secondary Clusters
    if selected_tab == "Outlier Phrases":
        # Assuming you want to process and display the keywords for subcluster 'p-1s-1'
        p1s1_outliers_df = process_subclusters(data)
        
        # Filter for records where subcluster is 'p-1s-1'
        p1s1_outliers_df = p1s1_outliers_df[p1s1_outliers_df["subcluster"] == "p-1s-1"]
    
        st.write("### Outlier Phrases for Subcluster 'p-1s-1'")
    
        for index, row in p1s1_outliers_df.iterrows():
            col1, col2 = st.columns([2, 6])  # Adjusted column widths
            
            with col1:
                st.text(f"Subcluster: {row['subcluster']}")  # Show subcluster ID
                
            with col2:
                # Split the 'sub_keywords' by commas or spaces and display them one by one
                keywords = row['sub_keywords'].split(',')  # Assuming keywords are comma-separated
                for keyword in keywords:
                    st.text(keyword.strip())  # Display each keyword one by one









# Main function for handling session state & routing
def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # Page Routing
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "tickets_home":
        tickets_home_page()
    elif st.session_state.page == "emails":
        st.write("Welcome to the Emails Page!")
    elif st.session_state.page == "messages":
        st.write("Welcome to the Messages Page!")
    elif st.session_state.page == "group_messages":
        st.write("Welcome to the Group Messages Page!")
    elif st.session_state.page == "alert_tickets":
        st.write("Welcome to the Alert Tickets Page!")
    elif st.session_state.page == "support_tickets":
        support_tickets_page()
# Run the app
if __name__ == "__main__":
    main()
