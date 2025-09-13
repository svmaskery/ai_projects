import os
from dotenv import load_dotenv
import streamlit as st
import time
from openai import OpenAI, OpenAIError

# Initialize the LLM client
try:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("API_KEY environment variable not set/missing")
        st.stop()
    
    client = OpenAI(
        base_url=os.environ("API_URL"),
        api_key=api_key,
    )
except ValueError as e:
    st.error(f"Configuration error: {e}")
    st.stop()

# Define a robust prompt template with guardrails.
SYSTEM_PROMPT = """
You are a helpful, harmless, and unbiased AI assistant. Your purpose is to provide clear, concise, and accurate information.
You will not generate any content that is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.
If a request falls into any of these categories, you will politely decline and explain that you cannot fulfill the request due to safety guidelines.
Maintain a professional and friendly tone in all your responses.
"""

def get_openrouter_response(messages_history):
    """
    Sends the entire chat history to the language model and handles the response.

    Args:
        messages_history (list): The list of previous messages in the conversation.
        
    Returns:
        str: The model's response or an error message.
    """
    try:
        model = os.environ("MODEL")
        completion = client.chat.completions.create(
            model=model,
            messages=messages_history,
            temperature=0.7,
        )
        
        return completion.choices[0].message.content

    except OpenAIError as e:
        st.error(f"An API error occurred: {e}")
        return "Sorry, I'm having trouble connecting to the service. Please try again later."
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return "An unknown error occurred. Please try again."

def main():
    """
    Main Streamlit application function.
    """
    st.title("OpenRouter Chatbot")
    
    # Initialize chat history in Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Display existing chat messages (Skip the system prompt)
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User input and conversation history
    if user_input := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                llm_response = get_openrouter_response(st.session_state.messages)

            message_placeholder.markdown(llm_response)

        st.session_state.messages.append({"role": "assistant", "content": llm_response})

if __name__ == "__main__":
    main()
