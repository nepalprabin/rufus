from smolagents import Tool
from typing import Dict
from config import Config
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
        model = SentenceTransformer(Config.DEFAULT_SENTENCE_EMBEDDING_MODEL)
        
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
            if similarity > 0.1:  # Adjustable threshold
                relevant_content[url] = content
        
        # Sort by relevance score
        sorted_content = {k: v for k, v in sorted(
            relevant_content.items(), 
            key=lambda item: item[1]["relevance_score"], 
            reverse=True
        )}
        
        return sorted_content