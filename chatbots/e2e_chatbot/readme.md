# Chat with Documents
This is a Streamlit application that allows you to have a conversational chat with your PDF and TXT documents. It uses the power of LangChain for document processing and retrieval, Streamlit for the user interface, and OpenRouter to access various large language models (LLMs).

## Features
**Document Processing**: Upload and process PDF and TXT files.

**Conversational Chat**: Ask questions about your documents in a natural, conversational manner.

**Context-Aware Responses**: The model provides answers by retrieving information directly from the document.

**Customizable LLM**: Select from a range of LLMs available through OpenRouter.

**Modular Code**: The codebase is split into modular files (app.py, helper.py) for better organization and reusability.

## Prerequisites
Before you begin, ensure you have the following installed on your machine:
```
python 3.8+
git
```
## Setup
Follow these steps to get the application up and running on your local machine.

1. Clone the Repository

First, clone this repository to your local machine using Git:
```
$git clone [https://github.com/svmaskery/e2e_chatbot.git](https://github.com/svmaskery/e2e_chatbot.git)
$cd e2e_chatbot
```
2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

> $python -m venv venv

Activate the virtual environment:

On macOS and Linux:

> $source venv/bin/activate

On Windows:

> $.\venv\Scripts\activate

3. Install Dependencies

With your virtual environment active, install the required libraries using the requirements.txt file:

> $pip install -r requirements.txt

4. Configure your API Key

To use the application, you need an API key from OpenRouter.

Go to the OpenRouter website and create a free account.

Generate a new API key.

In the Streamlit application's sidebar, paste this key into the "Enter your OpenRouter API Key" text box.

5. Run the Application

Now you can start the application using the Streamlit CLI:

> $streamlit run app.py

This will launch the application in your default web browser.

## Usage
On the sidebar, enter your OpenRouter API key.

Select your preferred LLM from the dropdown menu.

Click "Browse files" or drag and drop a PDF or TXT file to upload it.

Click the "Process Document" button. Wait for the success message.

Once the document is processed, you can start asking questions in the chat box at the bottom of the page.