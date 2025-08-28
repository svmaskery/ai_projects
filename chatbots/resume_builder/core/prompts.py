# In a real app, these would be fine-tuned and more robust.
# Use this as a starting point.

# Prompt to parse the resume text into a structured Pydantic model
RESUME_PARSER_PROMPT = """
You are an expert resume parsing bot. Your task is to extract all relevant information from the provided resume text and format it into a JSON object that matches the following Pydantic schema:

{format_instructions}

Resume Text:
{text}

JSON Output:
"""

# Prompt to parse the job description text into a structured Pydantic model
JD_PARSER_PROMPT = """
You are an expert job description parsing bot. Your task is to extract all key details from the provided job description text and format it into a JSON object that matches the following Pydantic schema:

{format_instructions}

Job Description Text:
{text}

JSON Output:
"""

# Prompt to generate feedback and content based on both resume and JD
FEEDBACK_GENERATION_PROMPT = """
You are an AI-powered career coach and resume expert. Your goal is to provide a user with actionable feedback and content to improve their resume for a specific job.

Here is the user's resume data in JSON format:
{resume_data}

Here is the job description text:
{jd_text}

Based on this information, generate a JSON response that strictly adheres to the following Pydantic schema:
{format_instructions}

JSON Output:
"""