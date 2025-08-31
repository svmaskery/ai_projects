import os
import json
import logging
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pdfplumber
from .prompts import HEADER_EXTRACTION_PROMPT
from .utils import get_response_from_llm

logging.basicConfig(level=logging.INFO)

def get_resume_headers_dynamically(resume_text: str) -> List[str]:
    """Uses an LLM to dynamically identify and return a list of resume headers."""
    try:
        logging.info("Attempting to dynamically extract resume headers.")
        llm = get_response_from_llm()
        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template=HEADER_EXTRACTION_PROMPT,
            input_variables=["resume_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | llm | parser

        # Invoke the LLM with the raw resume text to get the headers
        raw_headers = chain.invoke({"resume_text": resume_text})
        logging.info(f"Dynamically extracted headers: {raw_headers}")
        return raw_headers
    except Exception as e:
        logging.error(f"Error during dynamic header extraction: {e}")
        return []

def load_and_chunk_document_with_metadata(file_object) -> Tuple[List[Document], List[str]]:
    """Loads a PDF and splits it into semantically meaningful chunks with metadata."""
    try:
        # Step 1: Use pdfplumber to extract text from a wide variety of PDFs
        text_content = ""
        with pdfplumber.open(file_object) as pdf:
            # print("pdf: ", pdf)
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
        
        logging.info("PDF loaded successfully.")

        # Step 1: Dynamically get the headers from the text
        header_names = get_resume_headers_dynamically(text_content)
        
        # Format the headers for the MarkdownHeaderTextSplitter
        headers_to_split_on = [("#", header) for header in header_names]
        
        # Step 2: Split by major resume headers to get semantic sections
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        sections = markdown_splitter.split_text(text_content)
        logging.info(f"Document split into {len(sections)} semantic sections.")

        # Step 3: Chunk each section if it's too large
        final_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        
        for section in sections:
            section_chunks = text_splitter.split_documents([section])
            for chunk in section_chunks:
                if 'header' in chunk.metadata:
                    chunk.metadata['section'] = chunk.metadata['header']
                else:
                    chunk.metadata['section'] = 'Unknown'
            final_chunks.extend(section_chunks)

        logging.info(f"Final document split into {len(final_chunks)} chunks with metadata.")
        return final_chunks, header_names
    except Exception as e:
        logging.error(f"Error loading and chunking document: {e}")
        return [], []

def create_retrievers(chunks: List[Document]):
    """Creates both BM25 and FAISS retrievers from document chunks with metadata."""
    if not chunks:
        return None, None
        
    try:
        bm25_retriever = BM25Retriever.from_documents(chunks)
        logging.info("BM25 retriever created successfully.")

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_vector_store = FAISS.from_documents(chunks, embedding_model)
        faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={"k": 5})
        logging.info("FAISS retriever created successfully.")
        
        return bm25_retriever, faiss_retriever
    except Exception as e:
        logging.error(f"Error creating retrievers: {e}")
        return None, None

def retrieve_context_hybrid(bm25_retriever, faiss_retriever, query: str, header_names: List[str]) -> str:
    """
    Performs a hybrid search with dynamic metadata filtering based on found headers.
    """
    if not bm25_retriever or not faiss_retriever:
        logging.error("One or both retrievers are invalid.")
        return None

    try:
        # A dynamic keyword-to-metadata mapping for filtering
        faiss_filter = None
        for header in header_names:
            if header.lower() in query.lower():
                faiss_filter = {"section": header}
                break
        
        # Get documents from both retrievers
        bm25_docs = bm25_retriever.invoke(query)
        
        # Apply the metadata filter to the FAISS search if a filter was found
        if faiss_filter:
            faiss_retriever.search_kwargs['filter'] = faiss_filter
            logging.info(f"Applying dynamic metadata filter to FAISS: {faiss_filter}")
        else:
            logging.info("No specific metadata filter applied.")
        
        faiss_docs = faiss_retriever.invoke(query)
        
        # Simple fusion: combine and de-duplicate the documents
        all_docs = bm25_docs + faiss_docs
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        
        context = "\n\n".join([doc.page_content for doc in unique_docs])
        logging.info(f"Retrieved {len(unique_docs)} unique chunks via hybrid search.")
        return context
    except Exception as e:
        logging.error(f"Error performing hybrid retrieval: {e}")
        return None