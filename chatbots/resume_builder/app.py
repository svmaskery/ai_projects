import streamlit as st
import io
import logging

from core.processing import (
    load_and_chunk_document_with_metadata,
    parse_pdf_with_rag,
    parse_text_with_llm,
    generate_feedback,
    answer_query_with_rag,
    generate_full_resume_rewrite,
)

from core.prompts import (
    RESUME_PARSER_PROMPT,
    JD_PARSER_PROMPT,
)

from core.model import ResumeModel, JobDescriptionModel

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="AI-Powered Resume Builder & Coach",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("👨‍💼 AI-Powered Resume Builder & Coach")

# --- Session State Management ---
if "parsed_resume_chunks" not in st.session_state:
    st.session_state.parsed_resume_chunks = None
if "headers" not in st.session_state:
    st.session_state.headers = None
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None
if "jd_data" not in st.session_state:
    st.session_state.jd_data = None
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "rewritten_resume" not in st.session_state:
    st.session_state.rewritten_resume = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "error_message" not in st.session_state:
    st.session_state.error_message = ""
if "raw_resume_text" not in st.session_state:
    st.session_state.raw_resume_text = ""

# Display any stored error message and stop the app
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    st.stop()

# --- Reset Function ---
def reset_app():
    st.session_state.parsed_resume_chunks = None
    st.session_state.headers = None
    st.session_state.jd_text = ""
    st.session_state.resume_data = None
    st.session_state.jd_data = None
    st.session_state.feedback = None
    st.session_state.rewritten_resume = None
    st.session_state.messages = []
    st.session_state.error_message = ""
    st.session_state.raw_resume_text = ""

# --- Main Application Form ---
with st.form("input_form"):
    st.subheader("Your Inputs")
    uploaded_resume = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])
    jd_text = st.text_area("Paste the Job Description Text", height=300)
    
    workflow_choice = st.radio(
        "Choose your workflow:",
        ('Analyze and Get Feedback', 'Generate Full Resume Rewrite')
    )
    submitted = st.form_submit_button("Start Process", type="primary")

if submitted:
    if not uploaded_resume or not jd_text:
        st.session_state.error_message = "❌ Please upload a resume and paste a job description to analyze."
        st.stop()
    else:
        st.session_state.messages = []
        st.session_state.feedback = None
        st.session_state.rewritten_resume = None

        # In-memory processing of PDF
        resume_file_obj = io.BytesIO(uploaded_resume.getvalue())
        
        # The main if-else to switch between workflows
        if workflow_choice == 'Analyze and Get Feedback':
            
            with st.status("Starting analysis...", expanded=True) as status:
                
                # Step 1: Parse and Chunk the Resume
                status.write("🔍 Parsing resume content with RAG...")
                chunks, headers, raw_text = load_and_chunk_document_with_metadata(resume_file_obj)
                if not chunks:
                    st.session_state.error_message = "❌ Failed to parse your resume."
                    status.update(label="Analysis Failed!", state="error", expanded=False)
                    st.stop()
                st.session_state.parsed_resume_chunks = chunks
                st.session_state.headers = headers
                st.session_state.raw_resume_text = raw_text

                # Step 2: Extract structured data from Resume
                status.write("🧠 Extracting structured data from the resume...")
                resume_data, error = parse_pdf_with_rag(chunks, headers, ResumeModel, RESUME_PARSER_PROMPT)
                if error:
                    st.session_state.error_message = error
                    status.update(label="Analysis Failed!", state="error", expanded=False)
                    st.stop()
                st.session_state.resume_data = resume_data

                # Step 3: Extract structured data from JD
                status.write("📜 Extracting job description requirements...")
                jd_data, error = parse_text_with_llm(jd_text, JobDescriptionModel, JD_PARSER_PROMPT)
                if error:
                    st.session_state.error_message = error
                    status.update(label="Analysis Failed!", state="error", expanded=False)
                    st.stop()
                st.session_state.jd_data = jd_data
                st.session_state.jd_text = jd_text

                # Step 4: Generate Feedback
                status.write("📝 Generating tailored feedback and suggestions...")
                feedback, error = generate_feedback(resume_data.dict(), jd_text)
                if error:
                    st.session_state.error_message = error
                    status.update(label="Analysis Failed!", state="error", expanded=False)
                    st.stop()
                st.session_state.feedback = feedback

                status.update(label="✅ Analysis complete!", state="complete", expanded=False)

        else: # Generate Full Resume Rewrite workflow
            
            with st.status("Starting resume rewrite...", expanded=True) as status:
            
                # Step 1: Get data for rewrite
                status.write("🔍 Analyzing resume for rewrite...")
                chunks, headers, raw_text = load_and_chunk_document_with_metadata(resume_file_obj)
                if not chunks:
                    st.session_state.error_message = "❌ Failed to parse your resume."
                    status.update(label="Rewrite Failed!", state="error", expanded=False)
                    st.stop()
                st.session_state.raw_resume_text = raw_text

                resume_data, error = parse_pdf_with_rag(chunks, headers, ResumeModel, RESUME_PARSER_PROMPT)
                if error:
                    st.session_state.error_message = error
                    status.update(label="Rewrite Failed!", state="error", expanded=False)
                    st.stop()
                
                # Step 2: Generate Full Rewrite
                status.write("✍️ Generating a tailored resume rewrite...")
                rewritten_resume, error = generate_full_resume_rewrite(resume_data.dict(), jd_text, raw_text)
                if error:
                    st.session_state.error_message = error
                    status.update(label="Rewrite Failed!", state="error", expanded=False)
                    st.stop()

                st.session_state.rewritten_resume = rewritten_resume.content
                
                # Clear other state
                st.session_state.parsed_resume_chunks = None
                st.session_state.headers = None
                st.session_state.jd_text = None
                st.session_state.feedback = None

                status.update(label="✅ Rewrite complete!", state="complete", expanded=False)


# --- Displaying Results ---
if st.session_state.feedback:
    st.subheader("Analysis & Suggestions")
    st.write("---")
    
    st.markdown(f"**Overall Match Score:** `{st.session_state.feedback.overall_score}`")
    st.markdown(f"**Professional Summary:** {st.session_state.feedback.tailored_summary}")
    
    st.markdown("**Missing Skills:**")
    if st.session_state.feedback.missing_skills:
        st.warning(f"**Missing Keywords/Skills:**")
        st.markdown(", ".join(f"`{s}`" for s in st.session_state.feedback.missing_skills))
        st.markdown("---")
        
    # st.markdown("**Suggested Bullet Point Rewrites:**")
    # for rewrite in st.session_state.feedback.suggested_rewrites:
    #     st.markdown(f"**Original:** {rewrite.original}")
    #     st.markdown(f"**Rewrite:** {rewrite.rewritten}")
    #     st.markdown(f"**Reason:** {rewrite.reasoning}")
    if st.session_state.feedback.suggested_rewrites:
        st.markdown("**:bulb: Rewritten Bullet Points:**")
        for rewrite in st.session_state.feedback.suggested_rewrites:
            st.text_area("Original:", value=rewrite.original, height=50)
            st.text_area("Rewritten:", value=rewrite.rewritten, height=50)
            st.caption(f"Reasoning: {rewrite.reasoning}")
            st.markdown("---")

# Display the chat assistant
if st.session_state.parsed_resume_chunks:
    st.subheader("Interactive Resume Assistant")
    st.markdown("Ask me questions about the resume or job description.")

    # Re-display messages from chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the documents..."):
        # Display user message in chat message container
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, error = answer_query_with_rag(
                    prompt, 
                    st.session_state.parsed_resume_chunks, 
                    st.session_state.headers,
                    st.session_state.jd_text
                )
                if error:
                    st.session_state.error_message = error
                    st.stop()
                else:
                    st.markdown(response.content)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.content})

# Display the rewritten resume
if st.session_state.rewritten_resume:
    st.subheader("Tailored Resume Rewrite")
    st.markdown("Your professionally rewritten resume is ready. Copy and paste the text below.")
    st.text_area("Rewritten Resume", value=st.session_state.rewritten_resume, height=600)

# Add a reset button at the end
if st.session_state.feedback or st.session_state.rewritten_resume:
    st.markdown("---")
    st.button("Start Over", on_click=reset_app)