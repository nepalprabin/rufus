import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys and configuration settings
class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    RUFUS_API_KEY = os.getenv("RUFUS_API_KEY")
    
    # Default settings
    DEFAULT_MAX_PAGES = 50
    DEFAULT_MAX_DEPTH = 3
    DEFAULT_REQUEST_DELAY = 0.5
    DEFAULT_RESPECT_ROBOTS = True
    DEFAULT_OUTPUT_FORMAT = "markdown"
    DEFAULT_RELEVANCE_THRESHOLD = 0.3
    DEFAULT_SENTENCE_EMBEDDING_MODEL="all-MiniLM-L6-v2"


# Initialize configuration
# Config.initialize()
