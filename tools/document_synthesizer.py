from smolagents import Tool
from typing import Dict
class DocumentSynthesizerTool(Tool):
    name = "document_synthesizer_tool"
    description = """
    Synthesizes extracted content into structured documents ready for RAG systems.
    """
    inputs = {
        "relevant_content": {"type": "any", "description": "Dictionary of relevant content"},
        "prompt": {"type": "string", "description": "User's prompt describing the information needed"},
        "output_format": {"type": "string", "description": "Desired output format (markdown, json)", "nullable": True}
    }
    output_type = "any"

    def forward(self, relevant_content: Dict, prompt: str, output_format: str = "markdown") -> Dict:
        import json
        from datetime import datetime
        import hashlib
        
        if not relevant_content:
            return {
                "title": f"No relevant content found for: {prompt}",
                "content": "No relevant content was found matching your query.",
                "format": output_format,
                "metadata": {
                    "prompt": prompt,
                    "sources": 0,
                    "created_at": datetime.now().isoformat()
                }
            }
        
        # Create title from prompt
        title = f"Information about: {prompt}"
        
        # Track sources
        sources = list(relevant_content.keys())
        
        # Build content
        content_parts = []
        seen_content = set()  # To avoid duplicates
        
        if output_format == "markdown":
            # Markdown format
            content_parts.append(f"# {title}\n")
            content_parts.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
            content_parts.append("## Sources\n")
            
            for i, url in enumerate(sources[:10]):  # Limit to top 10 sources
                content_parts.append(f"{i+1}. [{relevant_content[url]['title']}]({url})")
            
            content_parts.append("\n## Content\n")
            
            for url, data in relevant_content.items():
                content_parts.append(f"### {data['title']}\n")
                content_parts.append(f"*Relevance score: {data['relevance_score']:.2f}*\n")
                
                # Split content into paragraphs
                paragraphs = data['text'].split('\n\n')
                for paragraph in paragraphs:
                    # Skip short paragraphs
                    if len(paragraph.split()) < 10:
                        continue
                    
                    # Create hash to check for duplicates
                    p_hash = hashlib.md5(paragraph.encode()).hexdigest()
                    if p_hash not in seen_content:
                        content_parts.append(paragraph)
                        content_parts.append("\n")
                        seen_content.add(p_hash)
                
                content_parts.append("\n")
        
        elif output_format == "json":
            # Create JSON content
            json_content = {
                "title": title,
                "prompt": prompt,
                "sources": sources,
                "created_at": datetime.now().isoformat(),
                "sections": []
            }
            
            for url, data in relevant_content.items():
                # Split content into paragraphs
                paragraphs = [p for p in data['text'].split('\n\n') if len(p.split()) >= 10]
                
                # Filter duplicates
                unique_paragraphs = []
                for p in paragraphs:
                    p_hash = hashlib.md5(p.encode()).hexdigest()
                    if p_hash not in seen_content:
                        unique_paragraphs.append(p)
                        seen_content.add(p_hash)
                
                section = {
                    "title": data['title'],
                    "url": url,
                    "relevance_score": data['relevance_score'],
                    "content": unique_paragraphs
                }
                
                json_content["sections"].append(section)
            
            content_parts = [json.dumps(json_content, indent=2)]
        
        # Combine all parts
        content = "\n".join(content_parts)
        
        return {
            "title": title,
            "content": content,
            "format": output_format,
            "metadata": {
                "prompt": prompt,
                "sources": len(sources),
                "created_at": datetime.now().isoformat()
            }
        }