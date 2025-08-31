"""
All the prompts being used by the pipeline. Each prompt has its own well defined purpose.
"""

# Prompt to parse the resume text into a structured Pydantic model
RESUME_PARSER_PROMPT = """
You are an expert technical recruiter and resume analyst. Your task is to accurately extract key information from a provided resume text and format it into a structured JSON object that matches the Pydantic instruction given below.
JSON Schema Instructions:
{format_instructions}

Strictly adhere to the following rules:
1.Only extract information that is explicitly present in the resume. DO NOT make up or infer any data.
2.Follow the JSON schema instructions provided exactly.

Resume Text:
{text}

JSON Output:
"""

# Prompt for the LLM to dynamically identify resume headers
HEADER_EXTRACTION_PROMPT = """
You are an expert at identifying resume sections. Given the resume text below, identify and list all major section headers that could be used to split the document.

A header is a capitalized, bold, or underlined phrase that introduces a new section.
Examples of common headers are: "Summary", "Experience", "Skills", "Education", "Projects", "Certifications".
List ONLY the header names themselves, without any other text.
Return the headers as a JSON array of strings.

Resume Text:
{resume_text}

JSON Array of Headers:
"""

# Prompt to parse the job description text into a structured Pydantic model
JD_PARSER_PROMPT = """
You are an expert job description parsing bot. Your task is to accurately extract all key details from the provided job description text and format it into a JSON object that matches the following Pydantic schema:
{format_instructions}

Strictly adhere to the following rules to extract key information from the job description:
1.Only extract information that is explicitly present in the job description. DO NOT make up or infer any data.
2.Follow the JSON schema instructions provided exactly.

Job Description Text:
{text}

JSON Output:
"""

# Prompt to generate feedback and content based on both resume and job description
FEEDBACK_GENERATION_PROMPT = """
You are a professional and encouraging career coach. Your task is to provide highly specific and actionable feedback to a candidate based on the provided resume and job description data.

Follow these steps precisely:
1.Analyze and Score: Compare the resume data with the job description to find keywords, skills, and experiences that align.
2.Generate a Tailored Professional Summary: Create a professional summary that is perfectly tailored to the job description and highlights the candidate's most relevant qualifications.
3.Identify Missing Skills: List any critical keywords or skills from the job description that are NOT present in the resume. Be a keyword-matching expert.
4.Rewrite Bullet Points: Identify every bullet points, if any, from the resume that could be rewritten to better align with the job description. For each rewrite, provide the original, the rewritten version, and a clear, concise reason for the change. 
5.Final Points: The rewritten version should use stronger action verbs and quantify achievements where possible, based on the job description context.

Here is the user resume data:
{resume_data}

Here is the job description:
{jd_text}

Based on this information, generate a JSON response that strictly adheres to the following Pydantic schema:
{format_instructions}

JSON Output:
"""

# Prompt to generate for the user to ask query about the resume and/or job description provided
QA_PROMPT = """
You are an expert resume assistant. Your task is to answer a user's question using ONLY the provided resume and job description contexts. You must compare information from both sources where necessary.

Follow these rules:
1. If the answer to the user's question requires information not present in the provided contexts, you MUST respond with, "I cannot answer that question based on the provided information."
2. Answer concisely and professionally.
3. Do not add any information, opinions, or details not found in the context.

Resume Context:
{resume_context}

Job Description Context:
{jd_context}

Question: {question}

Answer:
"""

FULL_RESUME_REWRITE_PROMPT = """
You are a professional resume writer and career coach. Your task is to rewrite an entire resume to be perfectly tailored for a specific job description.

Follow these strict instructions:
1.  Rewrite the professional summary to be a compelling, 3-4 sentence statement that highlights the candidate's most relevant qualifications for the job description.
2.  For each experience entry, rewrite the bullet points to use strong action verbs and incorporate keywords from the job description. Quantify achievements where possible.
3.  Ensure the "Skills" section is updated to prominently feature the keywords and technologies mentioned in the job description.
4.  Maintain the original resume's professional and educational history, dates, and structure. Do not change the years of experience, job titles, or companies.
5.  The final output must be a clean, well-formatted plain text document that is easy to copy and paste. Use clear headings and bullet points. Do not include any extra commentary or conversational text.

Resume Data:
{resume_data}

Job Description:
{jd_text}

Full Original Resume Text:
{raw_resume_text}

Rewritten Resume:
"""