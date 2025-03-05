#!/usr/bin/env python3
"""
Rufus: An intelligent web crawler for RAG systems.
"""

import argparse
from core.runner import run_rufus

def main():
    """Main entry point for Rufus CLI."""
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

if __name__ == "__main__":
    main()
