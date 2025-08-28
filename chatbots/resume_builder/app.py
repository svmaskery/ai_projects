import streamlit as st
from dotenv import load_dotenv
from core.model import ResumeModel, JobDescriptionModel
from core.processing import load_pdf_content, parse_document, generate_feedback
from core.prompts import RESUME_PARSER_PROMPT, JD_PARSER_PROMPT

# # Load environment variables
# load_dotenv()

st.set_page_config(page_title="AI-Powered Resume Builder", layout="wide")

# Initialize state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = None

st.title("👨‍💼 AI-Powered Resume Builder & Coach")
st.markdown("Upload your resume and a job description to get personalized, actionable feedback and content to improve your resume.")

# User inputs in a single form
with st.form("input_form"):
    st.subheader("Your Inputs")
    uploaded_resume = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])
    jd_text = st.text_area("Paste the Job Description Text", height=300)
    submitted = st.form_submit_button("Analyze Resume", type="primary")

# Main processing logic is now a simple, linear block
if submitted:
    # Clear previous messages and feedback
    st.session_state.messages = []
    st.session_state.feedback = None

    if not uploaded_resume or not jd_text:
        st.session_state.messages.append({"role": "assistant", "content": "❌ Please upload a resume and paste a job description to analyze."})
    else:
        with st.spinner("Analyzing your resume..."):
            # Step 1: Load resume content from PDF
            st.session_state.messages.append({"role": "assistant", "content": "✅ Resume file uploaded successfully."})
            resume_text = load_pdf_content(uploaded_resume)

            # print("--------------------------")
            # print("Resume Text: ", resume_text)
            # print("--------------------------")
            
            if not resume_text:
                st.session_state.messages.append({"role": "assistant", "content": "❌ Failed to read the PDF. Please ensure it is a valid PDF file."})
                st.stop()
            
            # Step 2: Parse Resume Data
            st.session_state.messages.append({"role": "assistant", "content": "🔍 Parsing resume content..."})
            resume_data = parse_document(resume_text, ResumeModel, RESUME_PARSER_PROMPT)

            # print("--------------------------")
            # print("Resume Data: ", resume_data)
            # print("--------------------------")
            if resume_data:
                with st.expander("Show Parsed Resume Data"):
                    st.json(resume_data.dict())
            else:
                st.session_state.messages.append({"role": "assistant", "content": "❌ Failed to parse your resume..."})
                st.stop()

            if not resume_data:
                st.session_state.messages.append({"role": "assistant", "content": "❌ Failed to parse your resume. The content may be too complex for the model."})
                st.stop()
            
            st.session_state.messages.append({"role": "assistant", "content": "✅ Resume parsed successfully."})

            # Step 3: Parse Job Description Data
            st.session_state.messages.append({"role": "assistant", "content": "🔍 Parsing job description..."})
            jd_data = parse_document(jd_text, JobDescriptionModel, JD_PARSER_PROMPT)

            # print("--------------------------")
            # print("JD Data: ", jd_data)
            # print("--------------------------")
            if jd_data:
                with st.expander("Show Parsed Job Description Data"):
                    st.json(jd_data.dict())
            else:
                st.session_state.messages.append({"role": "assistant", "content": "❌ Failed to parse the job description..."})
                st.stop()

            if not jd_data:
                st.session_state.messages.append({"role": "assistant", "content": "❌ Failed to parse the job description. Please check the format."})
                st.stop()
            
            st.session_state.messages.append({"role": "assistant", "content": "✅ Job description parsed successfully."})

            # Step 4: Generate Feedback
            st.session_state.messages.append({"role": "assistant", "content": "✨ Generating personalized feedback..."})
            feedback = generate_feedback(resume_data.dict(), jd_text)

            # print("--------------------------")
            # print("Feedback: ", feedback)
            # print("--------------------------")
            if feedback:
                with st.expander("Show Final Feedback Data (Pydantic Model)"):
                    st.json(feedback.dict())
            else:
                st.session_state.messages.append({"role": "assistant", "content": "❌ Failed to generate feedback..."})
                st.stop()

            if not feedback:
                st.session_state.messages.append({"role": "assistant", "content": "❌ Failed to generate feedback. Please try again later."})
                st.stop()
            
            st.session_state.messages.append({"role": "assistant", "content": "🎉 Analysis complete! Here is your personalized feedback."})
            st.session_state.feedback = feedback

# Display chat history in order of execution
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display structured feedback if available
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