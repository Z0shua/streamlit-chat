import os
import streamlit as st
from openai import AzureOpenAI
from utils import load_env
from tinydb import TinyDB, Query
from datetime import datetime
import uuid
import pandas as pd

# Load environment variables
load_env()

# Initialize Azure OpenAI client
endpoint = os.getenv("AZURE_ENDPOINT")
subscription_key = os.getenv("SUBSCRIPTION_KEY")
model_name = os.getenv("MODEL_NAME")
api_version = os.getenv("API_VERSION")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Initialize TinyDB for logging
log_db_path = "./logs/query_logs.json"
os.makedirs(os.path.dirname(log_db_path), exist_ok=True)
db = TinyDB(log_db_path)

# Streamlit app title
st.set_page_config(page_title="Medical OMOP CDM Assistant", page_icon="🏥", layout="wide")
st.title("🏥 Medical OMOP CDM Assistant")

# OMOP CDM system prompt with detailed medical context
omop_system_prompt = """You are a specialized medical informatics assistant with expertise in the OMOP Common Data Model (CDM).

About OMOP CDM:
- The Observational Medical Outcomes Partnership (OMOP) Common Data Model (CDM) is a standardized data model for observational healthcare data.
- It transforms data from disparate sources into a common format with standardized vocabularies.
- The model includes tables for clinical data (conditions, drugs, procedures, measurements, etc.), as well as standardized vocabulary mappings.
- OMOP CDM is maintained by the Observational Health Data Sciences and Informatics (OHDSI) community.

Provide accurate, factual information about:
1. OMOP CDM structure and tables (Person, Condition_Occurrence, Drug_Exposure, etc.)
2. Standardized vocabularies (SNOMED, RxNorm, LOINC, etc.)
3. ETL processes for data conversion
4. Best practices for OMOP implementation
5. OHDSI tools like ATLAS, ACHILLES, and WebAPI
6. Medical terminology and healthcare data concepts

Always provide precise, evidence-based answers with technical accuracy. Clarify ambiguities in questions before answering.
"""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": omop_system_prompt}]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Function to process the user input
def process_message(user_input):
    if not user_input.strip():
        return

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display a loading message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
    
    # Get response from the Azure OpenAI client
    # Using low temperature for factual medical information
    response = client.chat.completions.create(
        messages=st.session_state.messages,
        max_tokens=4096,
        temperature=0.2,  # Low temperature for consistent, factual responses
        top_p=0.95,
        model=model_name,
    )

    # Add assistant response to chat history
    assistant_message = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

    # Log the query and response
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": st.session_state.session_id,
        "user_query": user_input,
        "assistant_response": assistant_message,
    }
    db.insert(log_entry)

# Display chat history using chat message containers
st.markdown("### Medical OMOP CDM Conversation")
for message in st.session_state.messages[1:]:  # Skip the system message
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input with chat interface
with st.form(key="message_form", clear_on_submit=True):
    user_input = st.text_input("Ask about OMOP CDM or medical informatics:", key="user_message")
    submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input:
        process_message(user_input)
        st.rerun()  # This will only execute after form submission

# Sidebar for logs with improved display
with st.sidebar:
    st.title("Conversation Logs")
    
    # Add information about the model settings
    st.markdown("### Model Settings")
    st.info("✓ Medical domain-specific prompt\n✓ Low temperature (0.2) for factual responses\n✓ Optimized for OMOP CDM accuracy")
    
    if st.button("View Recent Logs", key="view_logs"):
        logs = db.all()
        if logs:
            # Create a more readable dataframe
            df = pd.DataFrame(logs)
            # Format timestamp and truncate long text fields
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df['user_query'] = df['user_query'].str.slice(0, 50) + '...'
            df['assistant_response'] = df['assistant_response'].str.slice(0, 50) + '...'
            st.dataframe(df)
        else:
            st.info("No logs found.")
    
    # Add a clear conversation option
    if st.button("Clear Conversation"):
        st.session_state.messages = [{"role": "system", "content": omop_system_prompt}]
        st.rerun()