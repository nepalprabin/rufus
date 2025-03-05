# setup.py
from setuptools import setup, find_packages

setup(
    name="Rufus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "smolagents>=0.1.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.1",
        "sentence-transformers>=2.2.2",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Rufus: An intelligent web crawler for RAG systems",
    keywords="web-crawler, rag, ai, llm",
    url="https://github.com/yourusername/rufus",
)
