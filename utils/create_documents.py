import json
from typing import List, Dict, Any
from langchain_core.documents import Document

def rufus_to_langchain_documents(rufus_result: Dict[str, Any]) -> List[Document]:
    """
    Convert Rufus scraping results to a list of Langchain Document objects.
    
    Args:
        rufus_result: The result dictionary from client.scrape()
        
    Returns:
        List of Langchain Document objects
    """
    # Get the content field which contains the JSON string
    content_str = rufus_result.get("content", "{}")
    
    # Parse the JSON string
    try:
        # If content is already a dictionary, use it directly
        if isinstance(content_str, dict):
            content_data = content_str
        else:
            content_data = json.loads(content_str)
    except json.JSONDecodeError:
        # Handle case where content isn't valid JSON
        print("Error: Could not parse JSON content")
        return []
    
    # Extract sections
    sections = content_data.get("sections", [])
    
    # Create a Document for each section
    documents = []
    for section in sections:
        # Get section metadata
        title = section.get("title", "Untitled")
        url = section.get("url", "")
        relevance_score = section.get("relevance_score", 0.0)
        
        # Get section content
        paragraphs = section.get("content", [])
        
        # Join paragraphs into a single text
        text_content = "\n\n".join(paragraphs)
        
        # Create metadata dictionary
        metadata = {
            "title": title,
            "url": url,
            "relevance_score": relevance_score,
            "source": "rufus_crawler",
            "prompt": content_data.get("prompt", ""),
            "created_at": content_data.get("created_at", "")
        }
        
        # Create Document
        doc = Document(
            page_content=text_content,
            metadata=metadata
        )
        
        documents.append(doc)
    
    return documents
