import os
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
# from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from typing import Type
from .model import *
from .prompts import *
from .rag_pipeline import *
from .utils import get_response_from_llm, generate_error_response

# Configure logging to provide detailed debug information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_text_with_llm(text: str, model: Type, prompt_template: str) -> Type:
    """Parses plain text directly with the LLM. No RAG pipeline needed."""
    try:
        logging.info("Starting direct text parsing for Job Description.")
        llm = get_response_from_llm()
        parser = PydanticOutputParser(pydantic_object=model)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm | parser
        
        parsed_output = chain.invoke({"text": text})
        logging.info("Job Description parsed successfully.")
        return parsed_output, None
    except Exception as e:
        logging.error(f"Error parsing text with LLM: {e}")
        return None, generate_error_response('given job decription')

def parse_document_with_rag(chunks: List[Document], headers: List[str], model: Type, prompt_template: str):
    """
    Parses a document using a RAG pipeline and sends relevant chunks to the LLM.
    
    This function now takes a file path, not the raw text.
    """
    try:
        logging.info("Starting RAG-based parsing.")
        
        if not chunks:
            logging.error("Failed to load and chunk document.")
            return None, generate_error_response('document')
        
        bm25_retriever, faiss_retriever = create_retrievers(chunks)
        if not bm25_retriever or not faiss_retriever:
            logging.error("Failed to create retrievers.")
            return None, generate_error_response('document')

        if "ResumeModel" in str(model):
            query = "Extract personal information, work history, education, skills and other relevant information from the resume."
        elif "JobDescriptionModel" in str(model):
            query = "Extract job title, company name, required responsibilities, and skills."
        else:
            query = "Extract all relevant information."

        retrieved_context = retrieve_context_hybrid(bm25_retriever, faiss_retriever, query, headers)

        if not retrieved_context:
            logging.error("Failed to retrieve relevant context.")
            return None, generate_error_response('document')

        llm = get_response_from_llm()
        parser = PydanticOutputParser(pydantic_object=model)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm | parser
        
        logging.info("Sending retrieved context to the LLM for parsing.")
        parsed_output = chain.invoke({"text": retrieved_context})
        logging.info("Document parsed successfully.")
        return parsed_output, None
    except Exception as e:
        logging.error(f"Error parsing document with LLM: {e}")
        return None, generate_error_response('document')

def generate_feedback(resume_data: dict, jd_text: str) -> FeedbackModel:
    """Generates personalized resume feedback based on parsed resume and job description."""
    llm = get_response_from_llm()
    parser = PydanticOutputParser(pydantic_object=FeedbackModel)
    
    prompt = PromptTemplate(
        template=FEEDBACK_GENERATION_PROMPT,
        input_variables=["resume_data", "jd_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    
    llm_input = {
        "resume_data": str(resume_data),
        "jd_text": jd_text
    }
    
    try:
        logging.info("Attempting to generate feedback.")
        feedback = chain.invoke(llm_input)
        logging.info("Feedback generated successfully.")
        return feedback, None
    except Exception as e:
        logging.error(f"Error generating feedback: {e}")
        return None, generate_error_response('feedback')

def answer_query_with_rag(query: str, chunks: List[str], headers: List[str], jd_text: str) -> str:
    """Answers a user's query based on the provided document chunks."""
    try:
        logging.info(f"Answering query: '{query}'")

        llm = get_response_from_llm()
        bm25_retriever, faiss_retriever = create_retrievers(chunks)
        if not bm25_retriever or not faiss_retriever:
            logging.error("Failed to create retrievers for query answering.")
            return None, generate_error_response('query')

        retrieved_resume_context = retrieve_context_hybrid(bm25_retriever, faiss_retriever, query, headers)
        
        if not retrieved_resume_context:
            return None, generate_error_response('query')

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
        return None, generate_error_response('query')
    
def generate_full_resume_rewrite(resume_data: dict, jd_text: str, raw_resume_text: str):
    """Generates a complete, tailored resume rewrite based on a job description."""
    try:
        logging.info("Starting full resume rewrite generation.")
        llm = get_response_from_llm()
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
        return None, generate_error_response('new resume')