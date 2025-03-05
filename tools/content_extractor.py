from smolagents import Tool
from typing import Dict
class ContentExtractionTool(Tool):
    name = "content_extraction_tool"
    description = """
    Extracts clean, readable content from HTML pages, removing navigation, ads, and other non-essential elements.
    """
    inputs = {
        "pages": {"type": "any", "description": "Dictionary of pages with HTML content"}
    }
    output_type = "any"

    def forward(self, pages: Dict) -> Dict:
        from bs4 import BeautifulSoup
        
        extracted_content = {}
        
        for url, page_data in pages.items():
            html = page_data.get("html", "")
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script, style elements and other non-content elements
            for element in soup(["script", "style", "header", "footer", "nav"]):
                element.decompose()
            
            # Extract text content
            text = soup.get_text(separator='\n')
            
            # Clean whitespace
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            # Extract metadata
            metadata = {}
            for meta in soup.find_all('meta'):
                if meta.get('name') and meta.get('content'):
                    metadata[meta['name']] = meta['content']
            
            extracted_content[url] = {
                "title": page_data.get("title", "No title"),
                "text": text,
                "metadata": metadata,
                "depth": page_data.get("depth", 0)
            }
        
        return extracted_content
