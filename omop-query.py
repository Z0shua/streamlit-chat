import os
import streamlit as st
from openai import AzureOpenAI
from utils import load_env
from tinydb import TinyDB, Query
from datetime import datetime
import uuid
import pandas as pd
import duckdb
from sqlalchemy import create_engine, inspect
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.llms.azure_openai import AzureOpenAI as LlamaIndexAzureOpenAI

# Load environment variables
load_env()

# Initialize Azure OpenAI client
endpoint = os.getenv("AZURE_ENDPOINT")
subscription_key = os.getenv("SUBSCRIPTION_KEY")
model_name = os.getenv("MODEL_NAME")
api_version = os.getenv("API_VERSION")

# DuckDB path
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "./data/synthea-remove-suffix.duckdb")

# Initialize TinyDB for logging
log_db_path = "./logs/query_logs.json"
os.makedirs(os.path.dirname(log_db_path), exist_ok=True)
db = TinyDB(log_db_path)

# OMOP CDM tables with descriptions for better SQL generation
OMOP_TABLES_INFO = {
    'person': '''Demographics of patients including gender, birth date, race and ethnicity.
               Primary key: person_id. Links to: observation_period, visit_occurrence, condition_occurrence''',
    'observation_period': '''Time periods during which a person is observed in the database.
                          Foreign key: person_id references person''',
    'visit_occurrence': '''Records of encounters with healthcare providers or facilities.
                        Foreign key: person_id references person''',
    'condition_occurrence': '''Records of diagnoses or conditions.
                            Foreign keys: person_id references person, visit_occurrence_id references visit_occurrence''',
    'drug_exposure': '''Records of drugs prescribed, administered or dispensed.
                     Foreign keys: person_id references person, visit_occurrence_id references visit_occurrence''',
    'procedure_occurrence': '''Records of procedures or interventions performed.
                            Foreign keys: person_id references person, visit_occurrence_id references visit_occurrence''',
    'measurement': '''Records of clinical or laboratory measurements.
                    Foreign keys: person_id references person, visit_occurrence_id references visit_occurrence''',
    'observation': '''Clinical facts about a patient.
                    Foreign keys: person_id references person, visit_occurrence_id references visit_occurrence''',
    'concept': '''Standardized clinical terminology dictionary.
                Primary key: concept_id. Referenced by all clinical events for standard concepts''',
    'vocabulary': '''Reference table of all vocabularies used.
                   Primary key: vocabulary_id. Referenced by concept table'''
}

# Streamlit app title
st.set_page_config(page_title="Medical OMOP SQL Assistant", page_icon="🏥", layout="wide")
st.title("🏥 Medical OMOP SQL Assistant")

# OMOP CDM system prompt with SQL expertise
omop_system_prompt = """You are a specialized medical informatics assistant with expertise in the OMOP Common Data Model (CDM).

About OMOP CDM:
- The Observational Medical Outcomes Partnership (OMOP) Common Data Model (CDM) is a standardized data model for observational healthcare data.
- It transforms data from disparate sources into a common format with standardized vocabularies.
- The model includes tables for clinical data (conditions, drugs, procedures, measurements, etc.), as well as standardized vocabulary mappings.
- OMOP CDM is maintained by the Observational Health Data Sciences and Informatics (OHDSI) community.

Key OMOP Query Patterns:
1. Patient Demographics:
   - Always use person table for basic demographics
   - Join with observation_period for valid time periods

2. Clinical Events:
   - Join condition_occurrence/drug_exposure/procedure_occurrence with visit_occurrence when needed
   - Use concept_id mappings from the concept table for standardized terms

3. Temporal Analysis:
   - Use start_date and end_date fields for temporal relationships
   - Consider observation_period for patient coverage

4. Vocabulary Mappings:
   - Join with concept table for human-readable terms
   - Use concept_relationship for related concepts

5. Best Practices:
   - Use appropriate table joins based on person_id and visit_occurrence_id
   - Include proper WHERE clauses for valid records
   - Optimize joins and WHERE clause ordering

When given a natural language query about medical data, provide:
1. An SQL query specifically for OMOP CDM that answers the question
2. A clear explanation of the results in medical context
3. Any relevant medical insights from the data
"""

class OMOPSQLAssistant:
    def __init__(self):
        """Initialize the OMOP SQL Assistant."""
        self.openai_client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        self.db_connection = self._connect_to_duckdb()
        self.llama_llm = self._setup_llama_llm()
        self.query_engine = self._setup_query_engine()
        self.query_cache = {}  # Simple cache for query results

    def _connect_to_duckdb(self):
        """Establish connection to DuckDB."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)
            
            # Create SQLAlchemy engine for DuckDB
            engine = create_engine(f"duckdb:///{DUCKDB_PATH}")
            
            # Test the connection
            with engine.connect() as conn:
                available_tables = inspect(engine).get_table_names()
                print(f"Connected to DuckDB. Available tables: {available_tables}")
            
            return engine
        except Exception as e:
            print(f"Error connecting to DuckDB: {e}")
            # Create an in-memory DuckDB as fallback
            return create_engine("duckdb:///:memory:")

    def _setup_llama_llm(self):
        """Initialize the Azure OpenAI LLM for LlamaIndex."""
        return LlamaIndexAzureOpenAI(
            model=model_name,
            api_key=subscription_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            temperature=0.1  # Low temperature for precise SQL generation
        )

    def _setup_query_engine(self):
        """Set up the SQL query engine for OMOP CDM."""
        try:
            # Get list of available tables using SQLAlchemy
            inspector = inspect(self.db_connection)
            available_tables = inspector.get_table_names()
            
            # Find intersection of OMOP tables and available tables
            table_names = [table for table in OMOP_TABLES_INFO.keys() if table in available_tables]
            
            if not table_names:
                # If no OMOP tables found, use all available tables
                table_names = available_tables
                
            print(f"Found {len(table_names)} tables: {table_names}")
            
            # Initialize SQLDatabase with OMOP table names
            sql_database = SQLDatabase(
                engine=self.db_connection,
                include_tables=table_names
            )
            
            # Create query engine with OMOP expertise
            query_engine = NLSQLTableQueryEngine(
                sql_database=sql_database,
                llm=self.llama_llm,
                table_info=OMOP_TABLES_INFO
            )
            
            return query_engine
        except Exception as e:
            print(f"Error setting up query engine: {e}")
            return None

    def process_query(self, user_query):
        """Process a natural language query and return SQL and results."""
        if user_query in self.query_cache:
            return self.query_cache[user_query]
        
        # Generate SQL query using LlamaIndex
        sql_query = self.query_engine.generate_sql(user_query)
        
        # Execute SQL query
        with self.db_connection.connect() as conn:
            result = conn.execute(sql_query).fetchall()
        
        # Cache the result
        self.query_cache[user_query] = (sql_query, result)
        
        return sql_query, result

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