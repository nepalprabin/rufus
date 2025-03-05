import os
import asyncio
import urllib.parse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
# Install required packages
# !pip install smolagents requests beautifulsoup4 sentence-transformers

from smolagents import CodeAgent, Tool, HfApiModel, DuckDuckGoSearchTool, OpenAIServerModel

# Set up environment variables for API keys if needed
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



class WebCrawlerTool(Tool):
    name = "web_crawler_tool"
    description = """
    Crawls a website starting from a given URL, following links up to a specified depth.
    Handles rate limiting and respects robots.txt.
    """
    inputs = {
        "start_url": {"type": "string", "description": "The URL to start crawling from"},
        "max_pages": {"type": "integer", "description": "Maximum number of pages to crawl", "nullable": True
},
        "max_depth": {"type": "integer", "description": "Maximum depth to crawl", "nullable": False
},
        "respect_robots": {"type": "boolean", "description": "Whether to respect robots.txt", "nullable": True
},
    }
    output_type = "any"

    def forward(self, start_url: str, max_pages: int = 50, max_depth: int = 3, respect_robots: bool = True) -> Dict:
        import requests
        from bs4 import BeautifulSoup
        from urllib.robotparser import RobotFileParser
        import time
        
        results = {
            "pages": {},
            "stats": {
                "pages_discovered": 0,
                "pages_crawled": 0,
                "start_time": datetime.now().isoformat(),
            }
        }
        
        visited_urls = set()
        queue = [(start_url, 0)]  # (url, depth)
        
        # Set up robots.txt parser if needed
        rp = None
        if respect_robots:
            parsed_url = urllib.parse.urlparse(start_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            rp = RobotFileParser()
            rp.set_url(f"{base_url}/robots.txt")
            try:
                rp.read()
            except:
                print("Warning: Could not read robots.txt")
        
        while queue and len(visited_urls) < max_pages:
            url, depth = queue.pop(0)
            
            # Skip if already visited or too deep
            if url in visited_urls or depth > max_depth:
                continue
            
            # Check robots.txt
            if rp and not rp.can_fetch("*", url):
                print(f"Skipping {url} (disallowed by robots.txt)")
                continue
            
            visited_urls.add(url)
            
            try:
                # Fetch the page
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    continue
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = soup.title.string if soup.title else "No title"
                
                # Save page content
                results["pages"][url] = {
                    "title": title,
                    "html": response.text,
                    "depth": depth
                }
                
                results["stats"]["pages_crawled"] += 1
                
                # Extract links for further crawling
                if depth < max_depth:
                    links = []
                    for a_tag in soup.find_all('a', href=True):
                        href = a_tag['href']
                        if href.startswith(('http://', 'https://')):
                            links.append(href)
                        elif not href.startswith(('#', 'javascript:', 'mailto:')):
                            # Handle relative URLs
                            base_url = urllib.parse.urljoin(url, '')
                            absolute_url = urllib.parse.urljoin(base_url, href)
                            links.append(absolute_url)
                    
                    # Add new links to queue
                    for link in links:
                        if link not in visited_urls:
                            queue.append((link, depth + 1))
                    
                    results["stats"]["pages_discovered"] += len(links)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error crawling {url}: {e}")
        
        # Update stats
        results["stats"]["end_time"] = datetime.now().isoformat()
        results["stats"]["total_pages_crawled"] = len(results["pages"])
        
        return results

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

class RelevanceEvaluationTool(Tool):
    name = "relevance_evaluation_tool"
    description = """
    Evaluates the relevance of extracted content based on the user's prompt using semantic similarity.
    """
    inputs = {
        "extracted_content": {"type": "any", "description": "Dictionary of extracted page content"},
        "prompt": {"type": "string", "description": "User's prompt describing the information needed"}
    }
    output_type = "any"

    def forward(self, extracted_content: Dict, prompt: str) -> Dict:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Load sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode the prompt
        prompt_embedding = model.encode(prompt)
        
        relevant_content = {}
        
        for url, content in extracted_content.items():
            # Combine title and text for evaluation
            full_text = f"{content['title']} {content['text']}"
            
            # Encode the content
            content_embedding = model.encode(full_text)
            
            # Calculate similarity
            similarity = np.dot(prompt_embedding, content_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(content_embedding)
            )
            
            # Score the content
            content["relevance_score"] = float(similarity)
            
            # Add to relevant content if score is above threshold
            if similarity > 0.3:  # Adjustable threshold
                relevant_content[url] = content
        
        # Sort by relevance score
        sorted_content = {k: v for k, v in sorted(
            relevant_content.items(), 
            key=lambda item: item[1]["relevance_score"], 
            reverse=True
        )}
        
        return sorted_content

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


def create_rufus_agent(model_name=None):
    """
    Create the Rufus agent with all necessary tools.
    
    Args:
        model_name: The name or ID of the model to use. Defaults to using Hugging Face API.
    
    Returns:
        A configured CodeAgent instance.
    """
    # Create tools
    crawler_tool = WebCrawlerTool()
    extractor_tool = ContentExtractionTool()
    relevance_tool = RelevanceEvaluationTool()
    synthesizer_tool = DocumentSynthesizerTool()
    search_tool = DuckDuckGoSearchTool()
    
    # Set up the model
    if model_name:
        if "openai" in model_name.lower() or "gpt" in model_name.lower():
            model = OpenAIServerModel(model_id="gpt-4o", 
                                      api_base=os.getenv("OPENAI_API_BASE"),
                                      api_key=os.getenv("OPENAI_API_KEY")
                                      )
        elif "anthropic" in model_name.lower() or "claude" in model_name.lower():
            from smolagents.models import AnthropicModel
            model = AnthropicModel(model_id=model_name)
        elif "ollama" in model_name.lower():
            from smolagents.models import LiteLLMModel
            model = LiteLLMModel(model_id=f"ollama/{model_name}")
        else:
            # Default to Hugging Face model
            model = HfApiModel(model_id=model_name)
    else:
        # Default to a powerful Hugging Face model
        model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
    
    # Create the agent
    agent = CodeAgent(
        tools=[crawler_tool, extractor_tool, relevance_tool, synthesizer_tool, search_tool],
        additional_authorized_imports=['random', 're', 'statistics', 'time', 'unicodedata', 'math', 'stat', 'queue', 'itertools', 'collections', 'datetime', "json"],
        model=model,
        max_steps=10,
        # verbose=True
    )
    
    return agent

# Example function to run Rufus
def run_rufus(start_url, prompt, model_name=None, output_format="markdown", output_file=None):
    """
    Run Rufus to extract information from a website based on a prompt.
    
    Args:
        start_url: The URL to start crawling from
        prompt: User prompt describing the information needed
        model_name: The LLM to use
        output_format: "markdown" or "json"
        output_file: Optional file to save the output
    
    Returns:
        The synthesized document
    """
    # Create the agent
    agent = create_rufus_agent(model_name)
    
    # Prepare the input
    input_prompt = f"""
    Extract information from the website {start_url} based on this prompt: "{prompt}"
    
    Follow these steps:
    1. Crawl the website starting from the given URL
    2. Extract clean, readable content from the HTML
    3. Evaluate the relevance of each page to the prompt
    4. Synthesize the relevant content into a structured document in {output_format} format
    
    Return the final document.
    """
    
    # Run the agent
    result = agent.run(input_prompt)
    
    # Save output if requested
    if output_file and isinstance(result, dict) and "content" in result:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["content"])
        print(f"Output saved to {output_file}")
    
    return result


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rufus: An intelligent web crawler for RAG systems")
    parser.add_argument("--url", required=True, help="Starting URL to crawl")
    parser.add_argument("--prompt", required=True, help="User prompt describing information needed")
    parser.add_argument("--model", default=None, help="LLM model to use")
    parser.add_argument("--format", choices=["markdown", "json"], default="json", help="Output format")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    print(f"Starting Rufus crawler for URL: {args.url}")
    print(f"Prompt: {args.prompt}")
    print(f"Model: {args.model or 'Default (Qwen/Qwen2.5-Coder-32B-Instruct)'}")
    
    result = run_rufus(
        start_url=args.url,
        prompt=args.prompt,
        model_name=args.model,
        output_format=args.format,
        output_file=args.output
    )
    
    if isinstance(result, dict):
        print("\nExtraction complete!")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Sources: {result.get('metadata', {}).get('sources', 'N/A')}")
        
        if args.output:
            print(f"Full output saved to: {args.output}")
        else:
            print("\nPreview of content:")
            content = result.get("content", "")
            print(content[:500] + "..." if len(content) > 500 else content)
    else:
        print("Unexpected result format. Please check the agent's output.")
