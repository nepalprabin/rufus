# Rufus - An intelligent web crawler

Rufus is an AI-driven web crawler designed to intelligently navigate websites and extract relevant content based on user prompts. Its primary purpose is to synthesize structured documents from web content that can be seamlessly integrated into Retrieval Augmented Generation (RAG) pipelines.

## Key Features:

- Intelligent Crawling: Crawls websites based on user-defined prompts, following links up to a specified depth.
- Content Extraction: Removes non-essential elements like scripts, styles, and navigation to extract clean, readable content.
- Relevance Evaluation: Uses semantic similarity to evaluate content relevance to the user's prompt.
- Document Synthesis: Organizes extracted content into structured documents ready for RAG systems.
- Multiple Output Formats: Supports both markdown and JSON output formats.

## **Setup Environment**

1.  Clone the repository:

    ```
    git clone https://github.com/nepalprabin/rufus.git
    cd rufus
    ```

2.  Create a virtual environment:

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install the dependencies:

    ```
    pip install -r requirements.txt
    ```

## **Configure API Keys**

1.  Create a **`.env`** file in the project root with your API keys:

    ```
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_API_BASE=https://api.openai.com/v1
    RUFUS_API_KEY=rufus_api_key
    ```

    **Run Rufus**

<!-- -->

2.  Run the Python script:

    ```
    from client import RufusClient
    import os

    key = os.getenv('RUFUS_API_KEY')
    client = RufusClient(api_key=key)

    instructions = "Get all the blog contents"
    documents = client.scrape(instructions, "https://nepalprabin.github.io")
    ```
