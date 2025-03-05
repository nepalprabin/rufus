# rufus/client.py
import os
from typing import Dict, List, Optional, Union
from smolagents import CodeAgent, OpenAIServerModel
from tools import WebCrawlerTool, ContentExtractionTool, RelevanceEvaluationTool, DocumentSynthesizerTool
from  config import Config
from utils.create_documents import rufus_to_langchain_documents


class RufusClient:
    """Client for the Rufus web crawler."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Rufus client.
        
        Args:
            api_key: Rufus API key for authentication
            model: Model identifier to use
        """
        # Validate Rufus API key
        self.api_key = api_key
        if not self._validate_rufus_key(api_key):
            raise ValueError("Invalid Rufus API key")
        
        # Initialize the agent on first use (lazy loading)
        self._agent = None
    
    def _validate_rufus_key(self, key):
        """Validate the Rufus API key against configured key."""
        if key == Config.RUFUS_API_KEY:
            return True
        return False
    
    @property
    def agent(self):
        """Get or create the underlying agent."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent
    
    def _create_agent(self):
        """Create the CodeAgent with necessary tools."""
        # Create tools
        crawler_tool = WebCrawlerTool()
        extractor_tool = ContentExtractionTool()
        relevance_tool = RelevanceEvaluationTool()
        synthesizer_tool = DocumentSynthesizerTool()
        
        # Set up the model
        model = OpenAIServerModel(
                    model_id="gpt-4o",
                    api_key=Config.OPENAI_API_KEY
                )
        
        # Create the agent
        agent = CodeAgent(
            tools=[crawler_tool, extractor_tool, relevance_tool, synthesizer_tool],
            additional_authorized_imports=['datetime', 'json', 'hashlib'],
            model=model,
            max_steps=10,
        )
        
        return agent
    
    def scrape(self, url: str, instructions: str = None, max_pages: int = 50, 
               output_format: str = "json", save_to: str = "output.json") -> Dict:
        """
        Scrape a website and extract information based on instructions.
        
        Args:
            url: The URL to start crawling from
            instructions: Description of what information to find (optional)
            max_pages: Maximum number of pages to crawl
            output_format: Format of the output ("json" or "markdown")
            save_to: File path to save the results
            
        Returns:
            A dictionary containing the extracted information
        """
        prompt = instructions or "Extract all relevant information from this website"
        
        # Prepare the input for the agent
        input_prompt = f"""
        Extract information from the website {url} based on this prompt: "{prompt}"
        
        Follow these steps:
        1. Crawl the website starting from the given URL
        2. Extract clean, readable content from the HTML
        3. Evaluate the relevance of each page to the prompt
        4. Synthesize the relevant content into a structured document in {output_format} format  
        
        Return the final document.
        """
        
        # Run the agent
        result = self.agent.run(input_prompt)
        
        # Save output if requested
        if save_to and isinstance(result, dict) and "content" in result:
            with open(save_to, "w", encoding="utf-8") as f:
                f.write(result["content"])
        documents = rufus_to_langchain_documents(result)
        return documents
