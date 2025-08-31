import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

current_dir = Path(__file__).resolve().parent
dotenv_path = current_dir.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
api_key = os.getenv("OPENROUTER_API_KEY")

def get_response_from_llm():
    """Function to load default OpenAI based LLM model"""
    llm_response = ChatOpenAI(model="mistralai/mistral-small-3.2-24b-instruct:free",
                              openai_api_key=api_key,
                              base_url="https://openrouter.ai/api/v1",
                              temperature=0.7,
                              )
    return llm_response
    
def generate_error_response(pipeline: str="user request"):
    """Generate simple error response for UI"""
    return f"Failed to process {pipeline}. Please check."