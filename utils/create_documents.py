from typing import Union, Dict, List
from langchain_core.documents import Document
import json


def rufus_to_langchain_documents(rufus_output: Union[str, Dict]) -> List[Document]:
    """
    Convert Rufus JSON output to a list of Langchain Document objects.
    Handles the nested JSON structure where content contains another JSON string.
    
    Args:
        rufus_output: The JSON output from Rufus (either as a string or parsed dict)
        
    Returns:
        A list of Langchain Document objects
    """
    # Parse JSON if it's a string
    if isinstance(rufus_output, str):
        try:
            outer_json = json.loads(rufus_output)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in input")
    else:
        outer_json = rufus_output
    
    # Extract the inner JSON from the content field
    content_str = outer_json.get("content", "{}")
    
    # Parse the inner JSON
    try:
        if isinstance(content_str, str):
            inner_json = json.loads(content_str)
        else:
            inner_json = content_str
    except json.JSONDecodeError:
        print("Error parsing inner JSON content")
        return []
    
    documents = []
    
    # Extract sections from the inner JSON
    sections = inner_json.get("sections", [])
    
    # Create a Document for each section
    for section in sections:
        # Extract section metadata
        title = section.get("title", "Untitled")
        url = section.get("url", "")
        relevance_score = section.get("relevance_score", 0.0)
        
        # Combine content paragraphs into a single text
        # Some sections might have content as a string instead of a list
        content_items = section.get("content", [])
        if isinstance(content_items, list):
            content = "\n\n".join(content_items)
        else:
            content = str(content_items)
        
        # Create metadata dictionary
        metadata = {
            "title": title,
            "url": url,
            "relevance_score": relevance_score,
            "prompt": inner_json.get("prompt", ""),
            "source": "rufus_crawler",
            "created_at": inner_json.get("created_at", "")
        }
        
        # Create Langchain Document
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        
        documents.append(doc)
    
    return documents
