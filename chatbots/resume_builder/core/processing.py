import re
import logging
from typing import List, Tuple
from io import BytesIO
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError, BaseModel
from pypdf import PdfReader

from core.prompts import HEADER_EXTRACTION_PROMPT, QA_PROMPT, FEEDBACK_GENERATION_PROMPT, FULL_RESUME_REWRITE_PROMPT
from core.model import FeedbackModel
from core.utils import get_response_from_llm

# Initialize loggers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize LLM and embedding models
llm = get_response_from_llm()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _extract_text_from_pdf(file_obj: BytesIO) -> Tuple[str, List[str]]:
    """Extracts text and a list of pages from a PDF file."""
    try:
        reader = PdfReader(file_obj)
        full_text = ""
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            full_text += text
            pages.append(text)
        return full_text, pages
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return "", []


def _create_and_run_chain(prompt: str, model_output_schema: BaseModel, input_variables: dict) -> Tuple[BaseModel | None, str | None]:
    """Creates, runs, and validates an LLM chain with Pydantic output parsing."""
    try:
        prompt_template = PromptTemplate(
            template=prompt,
            input_variables=list(input_variables.keys()),
            partial_variables={
                "format_instructions": model_output_schema.schema_json(indent=2)
            }
        )
        chain = prompt_template | llm.with_structured_output(model_output_schema)
        output = chain.invoke(input_variables)
        return output, None
    except ValidationError as e:
        logging.error(f"Validation Error: {e}")
        return None, f"Validation Error: The LLM output did not match the required schema. {e}"
    except Exception as e:
        logging.error(f"LLM Chain Execution Error: {e}")
        return None, f"An error occurred during LLM processing: {e}"


def _get_headers_from_llm(resume_text: str) -> List[str]:
    """Uses an LLM to identify headers in the resume text."""
    try:
        chain = PromptTemplate.from_template(HEADER_EXTRACTION_PROMPT) | llm
        response = chain.invoke({"resume_text": resume_text})
        # The LLM output should be a JSON array string. Attempt to parse it.
        headers = eval(response.content) if isinstance(response.content, str) else []
        return headers
    except Exception as e:
        logging.error(f"Error extracting headers with LLM: {e}")
        # Return a list of default common headers as a fallback
        return ["Summary", "Experience", "Skills", "Education", "Projects"]


def load_and_chunk_document_with_metadata(file_obj: BytesIO) -> Tuple[List[Document], List[str], str]:
    """
    Loads text from a PDF, splits it into chunks based on headers, and returns the chunks with metadata.
    """
    logging.info("Starting document loading and chunking with metadata.")
    full_text, pages = _extract_text_from_pdf(file_obj)
    if not full_text:
        return [], [], ""

    headers = _get_headers_from_llm(full_text)
    if not headers:
        logging.warning("No headers found by LLM. Using a simple splitter.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.create_documents([full_text])
        return chunks, [], full_text

    # Split document based on identified headers
    split_pattern = '|'.join(re.escape(header) for header in headers)
    sections = re.split(f"({split_pattern})", full_text, flags=re.IGNORECASE)
    
    docs = []
    current_section = None
    current_content = ""
    
    for part in sections:
        part = part.strip()
        if not part:
            continue
            
        if any(part.lower() == h.lower() for h in headers):
            if current_section:
                docs.append(Document(page_content=current_content.strip(), metadata={"header": current_section}))
            current_section = part
            current_content = ""
        else:
            current_content += " " + part
    
    if current_section:
        docs.append(Document(page_content=current_content.strip(), metadata={"header": current_section}))

    logging.info(f"Chunking complete. Created {len(docs)} chunks.")
    return docs, headers, full_text


def create_retrievers(chunks: List[Document]) -> Tuple[BM25Retriever, FAISS]:
    """Creates and returns BM25 and FAISS retrievers from document chunks."""
    if not chunks:
        logging.error("Cannot create retrievers from empty chunks list.")
        return None, None
    try:
        logging.info("Creating BM25 and FAISS retrievers.")
        bm25_retriever = BM25Retriever.from_documents(chunks)
        faiss_retriever = FAISS.from_documents(chunks, embedding_model).as_retriever()
        return bm25_retriever, faiss_retriever
    except Exception as e:
        logging.error(f"Error creating retrievers: {e}")
        return None, None


def retrieve_context_hybrid(bm25_retriever, faiss_retriever, query: str, headers: List[str]) -> str:
    """Retrieves relevant context using a hybrid (keyword + semantic) approach."""
    logging.info(f"Retrieving context for query: '{query}'")
    
    retrievers = [bm25_retriever, faiss_retriever]
    ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=[0.5, 0.5])
    
    docs = ensemble_retriever.get_relevant_documents(query)
    
    # Filter documents based on relevancy headers
    # TODO: Improve relevant headers extraction.
    relevant_headers = ["Experience", "Skills", "Summary", "Projects"]
    if headers:
        relevant_headers = [h for h in headers if h in relevant_headers]
    
    filtered_docs = [doc for doc in docs if doc.metadata.get("header") in relevant_headers]

    unique_docs = []
    seen_content = set()
    for doc in filtered_docs:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)
    
    context = "\n\n---\n\n".join([doc.page_content for doc in unique_docs])
    logging.info(f"Retrieved context length: {len(context)} characters.")
    return context


def parse_pdf_with_rag(chunks: List[Document], headers: List[str], model_output_schema: BaseModel, prompt: str) -> Tuple[BaseModel | None, str | None]:
    """Parses a PDF using a RAG pipeline and returns a structured output."""
    logging.info("Starting PDF parsing with RAG.")
    try:
        # Create retrievers
        bm25_retriever, faiss_retriever = create_retrievers(chunks)
        if not bm25_retriever or not faiss_retriever:
            return None, "Failed to create retrievers for PDF parsing."
        
        # Use a generic query to retrieve a broad context
        query = "Extract key information from the resume."
        context = retrieve_context_hybrid(bm25_retriever, faiss_retriever, query, headers)
        
        if not context:
            return None, "Failed to retrieve context for PDF parsing."

        # Run the LLM chain with the retrieved context
        input_variables = {"text": context}
        parsed_output, error = _create_and_run_chain(prompt, model_output_schema, input_variables)
        
        if error:
            return None, error
        
        logging.info("PDF parsing with RAG completed successfully.")
        return parsed_output, None
    except Exception as e:
        logging.error(f"Error in parse_pdf_with_rag: {e}")
        return None, f"Error during PDF parsing: {e}"


def parse_text_with_llm(text: str, model_output_schema: BaseModel, prompt: str) -> Tuple[BaseModel | None, str | None]:
    """Parses plain text using an LLM and returns a structured output."""
    logging.info("Starting text parsing with LLM.")
    input_variables = {"text": text}
    parsed_output, error = _create_and_run_chain(prompt, model_output_schema, input_variables)
    
    if error:
        return None, error
    
    logging.info("Text parsing with LLM completed successfully.")
    return parsed_output, None


def generate_feedback(resume_data: dict, jd_text: str) -> Tuple[FeedbackModel | None, str | None]:
    """Generates comprehensive feedback by comparing resume data and a job description."""
    logging.info("Starting feedback generation.")
    try:
        input_variables = {
            "resume_data": str(resume_data),
            "jd_text": jd_text
        }
        feedback, error = _create_and_run_chain(
            FEEDBACK_GENERATION_PROMPT,
            FeedbackModel,
            input_variables
        )
        if error:
            return None, error
        logging.info("Feedback generation completed successfully.")
        return feedback, None
    except Exception as e:
        logging.error(f"Error in generate_feedback: {e}")
        return None, f"Error generating feedback: {e}"


def answer_query_with_rag(query: str, chunks: List[Document], headers: List[str], jd_text: str) -> Tuple[str | None, str | None]:
    """Answers a user's query based on the provided document chunks."""
    try:
        logging.info(f"Answering query: '{query}'")
        
        bm25_retriever, faiss_retriever = create_retrievers(chunks)
        if not bm25_retriever or not faiss_retriever:
            return None, "I am unable to answer that question at this time. Failed to create retrievers."

        retrieved_resume_context = retrieve_context_hybrid(bm25_retriever, faiss_retriever, query, headers)
        
        if not retrieved_resume_context:
            return None, "I cannot find any relevant information in the document."

        prompt = PromptTemplate(
            template=QA_PROMPT,
            input_variables=["resume_context", "jd_context", "question"]
        )
        chain = prompt | llm
        
        response = chain.invoke({
            "resume_context": retrieved_resume_context, 
            "jd_context": jd_text, 
            "question": query
        })
        
        logging.info("Query answered successfully.")
        return response, None
    except Exception as e:
        logging.error(f"Error answering query with RAG: {e}")
        return None, f"I encountered an error trying to answer that question: {e}"


def generate_full_resume_rewrite(resume_data: dict, jd_text: str, raw_resume_text: str) -> Tuple[str | None, str | None]:
    """Generates a complete, tailored resume rewrite based on a job description."""
    # TODO: Need a better failsafe method when context is too big
    logging.info("Starting full resume rewrite generation.")
    try:
        prompt = PromptTemplate(
            template=FULL_RESUME_REWRITE_PROMPT,
            input_variables=["resume_data", "jd_text", "raw_resume_text"]
        )
        chain = prompt | llm
        
        llm_input = {
            "resume_data": str(resume_data),
            "jd_text": jd_text,
            "raw_resume_text": raw_resume_text
        }
        
        rewritten_resume = chain.invoke(llm_input)
        
        logging.info("Full resume rewrite generated successfully.")
        return rewritten_resume, None
    except Exception as e:
        logging.error(f"Error generating full resume rewrite: {e}")
        return None, f"Error generating full resume rewrite: {e}"
