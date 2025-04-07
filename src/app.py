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
st.set_page_config(page_title="Chatbot with Logging", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot with Logging")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
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
    response = client.chat.completions.create(
        messages=st.session_state.messages,
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
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
st.markdown("### Conversation")
for message in st.session_state.messages[1:]:  # Skip the system message
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input with form to prevent auto-resubmission
with st.form(key="message_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:", key="user_message")
    submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input:
        process_message(user_input)
        st.rerun()  # This will only execute after form submission

# Sidebar for logs with improved display
with st.sidebar:
    st.title("Conversation Logs")
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
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        st.rerun()