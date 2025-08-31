import streamlit as st
import os
import io
from dotenv import load_dotenv
from core.model import ResumeModel, JobDescriptionModel
from core.processing import generate_feedback, parse_document_with_rag, parse_text_with_llm, answer_query_with_rag
from core.prompts import RESUME_PARSER_PROMPT, JD_PARSER_PROMPT
from core.rag_pipeline import load_and_chunk_document_with_metadata


st.set_page_config(page_title="AI-Powered Resume Analyzer", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "parsed_resume_chunks" not in st.session_state:
    st.session_state.parsed_resume_chunks = None
if "headers" not in st.session_state:
    st.session_state.headers = []
if "error_message" not in st.session_state:
    st.session_state.error_message = ""

st.title("👨‍💼 AI-Powered Resume Analyzer & Coach")
st.markdown("Upload your resume and a job description to get personalized, actionable feedback and content to improve your resume.")

with st.form("input_form"):
    st.subheader("Your Inputs")
    uploaded_resume = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])
    jd_text = st.text_area("Paste the Job Description Text", height=300)
    submitted = st.form_submit_button("Analyze Resume", type="primary")

if st.session_state.error_message:
    st.error(st.session_state.error_message)
    st.stop()

if submitted:
    st.session_state.messages = []
    st.session_state.feedback = None
    st.session_state.parsed_resume_chunks = None
    st.session_state.headers = []
    st.session_state.jd_text = jd_text

    if not uploaded_resume or not jd_text:
        st.session_state.messages.append({"role": "assistant", "content": "Please upload a resume and paste a job description to analyze."})
    else:
        resume_file_obj = io.BytesIO(uploaded_resume.getbuffer())
        with st.spinner("Analyzing your resume..."):
            
            # Chunk the resume and store the chunks and headers in session state
            st.session_state.messages.append({"role": "assistant", "content": "Parsing resume contents..."})
            chunks, headers = load_and_chunk_document_with_metadata(resume_file_obj)
            if not chunks:
                st.session_state.error_message = "Failed to load your resume. Please check the logs."
                st.stop()
            st.session_state.parsed_resume_chunks = chunks
            st.session_state.headers = headers
            
            # Parse resume data from chunks
            resume_data = parse_document_with_rag(chunks, headers, ResumeModel, RESUME_PARSER_PROMPT)
            if not resume_data:
                st.session_state.error_message = "Failed to parse your resume. Please check the logs."
                st.stop()
            
            # Parse Job Description Data
            st.session_state.messages.append({"role": "assistant", "content": "Parsing job description..."})
            jd_data = parse_text_with_llm(jd_text, JobDescriptionModel, JD_PARSER_PROMPT)
            if not jd_data:
                st.session_state.error_message = "Failed to parse the job description. Please check the logs."
                st.stop()
            
            # Generate Feedback
            st.session_state.messages.append({"role": "assistant", "content": "Generating personalized feedback..."})
            feedback = generate_feedback(resume_data.dict(), jd_text)
            if not feedback:
                st.session_state.error_message = "Failed to generate feedback. Please check the logs."
                st.stop()
            
            st.session_state.messages.append({"role": "assistant", "content": "Analysis complete! Here is your personalized feedback."})
            st.session_state.feedback = feedback


# Display existing messages and feedback
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.feedback:
    st.subheader("Analysis & Suggestions")
    feedback = st.session_state.feedback

    st.markdown(f"**Overall Match Score:** `{feedback.overall_score:.2f}%`")
    st.markdown("---")

    st.subheader("Tailored Professional Summary")
    st.info(feedback.tailored_summary)
    st.markdown("---")

    st.subheader("Suggestions for Improvement")
    if feedback.missing_skills:
        st.warning(f"**Missing Keywords/Skills:**")
        st.markdown(", ".join(f"`{s}`" for s in feedback.missing_skills))
        st.markdown("---")

    if feedback.suggested_rewrites:
        st.markdown("**:bulb: Rewritten Bullet Points:**")
        for rewrite in feedback.suggested_rewrites:
            st.text_area("Original:", value=rewrite.original, height=50)
            st.text_area("Rewritten:", value=rewrite.rewritten, height=50)
            st.caption(f"Reasoning: {rewrite.reasoning}")
            st.markdown("---")

    st.markdown("---")
    st.subheader("Interactive Resume Assistant")
    st.markdown("Ask me questions about the resume content.")

    if prompt := st.chat_input("Ask a question about the resume..."):
        # Display user message in chat message container
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = answer_query_with_rag(prompt, st.session_state.parsed_resume_chunks, 
                                                 st.session_state.headers, st.session_state.jd_text)
                st.markdown(response.content)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
