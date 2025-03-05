from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool, OpenAIServerModel
import os
from typing import List
from config import Config
from tools import (
    WebCrawlerTool, 
    ContentExtractionTool, 
    RelevanceEvaluationTool, 
    DocumentSynthesizerTool
)

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
    
    tools = [crawler_tool, extractor_tool, relevance_tool, synthesizer_tool, search_tool]
    
    # Set up the model
    model = _get_model(model_name)
    
    # Authorized imports for the agent
    authorized_imports = [
        'random', 're', 'statistics', 'time', 'unicodedata', 
        'math', 'queue', 'itertools', 'collections', 'datetime', 'json'
    ]
    
    # Create the agent
    agent = CodeAgent(
        tools=tools,
        additional_authorized_imports=authorized_imports,
        model=model,
        max_steps=10,
        verbosity_level=2
    )
    
    return agent

def _get_model(model_name=None):
    """Get the appropriate model based on name."""
    if model_name:
        if "openai" in model_name.lower() or "gpt" in model_name.lower():
            return OpenAIServerModel(
                model_id="gpt-4o", 
                api_base=Config.OPENAI_API_BASE,
                api_key=Config.OPENAI_API_KEY
            )
        elif "anthropic" in model_name.lower() or "claude" in model_name.lower():
            from smolagents.models import AnthropicModel
            return AnthropicModel(model_id=model_name)
        elif "ollama" in model_name.lower():
            from smolagents.models import LiteLLMModel
            return LiteLLMModel(model_id=f"ollama/{model_name}")
        else:
            # Use specified Hugging Face model
            return HfApiModel(model_id=model_name)
    else:
        # Default to a powerful Hugging Face model
        return HfApiModel(model_id=Config.DEFAULT_MODEL)
