import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys and configuration settings
class Config:
    # API Keys
    HF_TOKEN = os.getenv("HF_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    RUFUS_API_KEY = os.getenv("RUFUS_API_KEY")
    
    # Default settings
    DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
    DEFAULT_MAX_PAGES = 50
    DEFAULT_MAX_DEPTH = 3
    DEFAULT_REQUEST_DELAY = 0.5
    DEFAULT_RESPECT_ROBOTS = True
    DEFAULT_OUTPUT_FORMAT = "markdown"
    DEFAULT_RELEVANCE_THRESHOLD = 0.3
    DEFAULT_SENTENCE_EMBEDDING_MODEL="all-MiniLM-L6-v2"
    
    # Set environment variables
    @classmethod
    def initialize(cls):
        if cls.HF_TOKEN:
            os.environ["HF_TOKEN"] = cls.HF_TOKEN
        if cls.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY

# Initialize configuration
Config.initialize()
