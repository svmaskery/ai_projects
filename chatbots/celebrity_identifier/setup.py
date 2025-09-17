from setuptools import setup, find_packages

with open('requirements.txt', 'r') as foo:
    requirements = foo.read().splitlines()

setup(
    name="Celebrity Identifier and Chatbot",
    version='0.1',
    author="Shree Vijay",
    author_email="svmaskery@gmail.com",
    packages=find_packages(),
    requires=requirements
)