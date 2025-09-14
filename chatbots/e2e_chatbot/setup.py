from setuptools import setup, find_packages

setup(
    name="chat-with-docs",
    version="0.1.0",
    description="A Streamlit application to chat with documents using LangChain",
    author="Shree Vijay",
    author_email="svmaskery@gmail.com",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)