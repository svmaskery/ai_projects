import os
from dotenv import load_dotenv
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

from src.helper import process_file


load_dotenv()

# Page configuration
st.set_page_config(page_title="Chat with Docs", page_icon="📄", layout="wide")
st.title("📄 Chat with PDF/TXT Documents")

# Model selection and file upload
with st.sidebar:
    st.header("1. Configuration")
    api_key = os.getenv("OPENROUTER_API_KEY")
    selected_model = st.selectbox(
        "Select a Model",
        options=[
            "mistralai/mistral-small-3.2-24b-instruct:free",
            "google/gemma-3n-e2b-it:free"
        ]
    )
    st.header("2. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])
    process_button = st.button("Process Document")


# Initialize session state variables
if "chain" not in st.session_state:
    st.session_state.chain = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# Call the process_file function when the button is clicked
if process_button:
    if process_file(uploaded_file, api_key, selected_model):
        st.session_state.messages.append(AIMessage(content=f"Document '{uploaded_file.name}' processed successfully. I am ready to answer your questions!"))
        st.success(f"Processed file: {uploaded_file.name}")
    else:
        st.session_state.messages.append(AIMessage(content="Failed to process the document. Please check the file and API key."))

# Display existing messages from chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Handle new user input
user_input = st.chat_input("Ask something...")
if user_input:
    if not st.session_state.chain:
        st.error("Please upload and process a document first.")
    else:
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.chat_message("user").write(user_input)
        
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.chain({"question": user_input})
                answer = result["answer"]
                st.session_state.messages.append(AIMessage(content=answer))
                st.chat_message("assistant").write(answer)
            except Exception as e:
                st.error(f"An error occurred while getting the response. Please try again. Error: {e}")

