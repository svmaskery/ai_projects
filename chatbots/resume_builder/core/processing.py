import os
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from typing import Type
from .model import *
from .prompts import *

# Load the API key from environment variables
current_dir = Path(__file__).resolve().parent
dotenv_path = current_dir.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
api_key = os.getenv("OPENROUTER_API_KEY")

# Configure logging to provide detailed debug information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Choose your LLM model (OpenRouter gives you many choices)
# For structured output, a powerful model is best (e.g., 'mistralai/mixtral-8x7b-instruct', 'google/gemini-pro')
llm = ChatOpenAI(
    model="mistralai/mistral-small-3.2-24b-instruct:free",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)


def load_pdf_content(uploaded_file):
    """
    Loads text content from an uploaded PDF file.
    Args:
        uploaded_file: The file object from Streamlit's file_uploader.
    Returns:
        The extracted text content as a single string.
    """
    if uploaded_file is None:
        logging.warning("No file uploaded.")
        return None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        logging.info(f"Loading PDF from temporary path: {tmp_file_path}")
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        content = " ".join([doc.page_content for doc in documents])
        logging.info("PDF content loaded successfully.")
        return content
    except Exception as e:
        logging.error(f"Failed to read PDF file: {e}", exc_info=True)
        return None
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def parse_document(text: str, model: Type, prompt_template: str):
    """
    A generic function to parse text into a Pydantic model using an LLM.
    """
    parser = PydanticOutputParser(pydantic_object=model)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['text'],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    # print("Prompt: ", prompt)
    chain = prompt | llm | parser

    try:
        logging.info(f"Attempting to parse document for model: {model.__name__}")
        llm_input = {"text": text}
        # print("--- Input to LLM for Parsing ---")
        # print(prompt.format(**llm_input, format_instructions=parser.get_format_instructions()))
        # print("-------------------------------")
        parsed_output = chain.invoke(llm_input)
        logging.info("Document parsed successfully.")
        return parsed_output
    except OutputParserException as e:
        logging.error(f"Failed to parse LLM output for {model.__name__}: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during parsing: {e}", exc_info=True)
        return None

def generate_feedback(resume_data: dict, jd_text: str):
    """
    Generates the structured feedback using the LLM and parsed data.
    """
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
        # print("--- Input to LLM for Feedback Generation ---")
        # print(prompt.format(**llm_input, format_instructions=parser.get_format_instructions()))
        # print("-----------------------------------------")
        feedback = chain.invoke(llm_input)
        logging.info("Feedback generated successfully.")
        return feedback
    except OutputParserException as e:
        logging.error(f"Failed to parse feedback from LLM: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during feedback generation: {e}", exc_info=True)
        return None