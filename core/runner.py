from typing import Dict, Optional
from .agent import create_rufus_agent

def run_rufus(
    start_url: str, 
    prompt: str, 
    model_name: Optional[str] = None, 
    output_format: str = "markdown", 
    output_file: Optional[str] = None
) -> Dict:
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
