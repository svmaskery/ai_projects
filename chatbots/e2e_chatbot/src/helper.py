import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import tempfile
from langchain.prompts import ChatPromptTemplate

def process_file(uploaded_file, api_key_input, selected_model_name):
    """Load, split, embed and store the uploaded file."""
    if not api_key_input:
        st.error("Please enter your API key.")
        return False
    if not uploaded_file:
        st.error("Please upload a file.")
        return False

    st.session_state.api_key = api_key_input

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Choose loader
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    docs = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Define a system prompt to set model guardrails and behavior
    system_prompt = (
        "You are a helpful and accurate assistant. "
        "Your task is to answer user questions based solely on the information provided in the given document. "
        "Do not use any external knowledge. "
        "If the answer cannot be found in the document, politely state that you do not have enough information to answer the question."
    )
    QA_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}\nChat History: {chat_history}\nContext: {context}")
        ]
    )
    # Build LLM and ConversationalRetrievalChain
    try:
        llm = ChatOpenAI(
            model=selected_model_name,
            openai_api_key=st.session_state.api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=4096,
        )
        st.session_state.llm = llm
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            verbose=True
        )
    except Exception as e:
        st.error(f"Failed to initialize the LLM or chain. Check your API key and model selection. Error: {e}")
        os.unlink(tmp_path)
        return False
    
    os.unlink(tmp_path)
    return True