# How Rufu works

Rufus use `smolagent`, a Huggingface's Agent library to crawl web content. Here, four different tools are used.

- ContentExtractionTool: this extracts clean contents from HTML pages excluding ads, navigation and non-essential elements.
- DocumentSynthesizerTool: It takes extracted documents and convert it into structured documents
- RelevanceEvaluatorTool: Uses sentence transformer to evaluate similarity between user provided prompt and the extracted results and return the results having most relevancy score
- WebCrawlerTool: It crawls websites starting from given URL and follows specific depth

I started the use case by trying to scrape the data dynamically but due to the dynamic nature of different webpages, I could not scrape the info. Next I tried using [Crawl4ai](https://crawl4ai.com/mkdocs/) using LLM based extraction, but the result was not satisfactory. Also, it was not able to extract the contents from deeplinks effectively.

So, I started off with `Smolagent` and started creating simple crawler using builtin tools. Later, I extended it to create custom tools for different purpose so that I can easily extract content on desired format.

## RAG pipeline Integration

The output from Rufu is a langchain Documents objects which we can use directly to create vectorstore. In addition to it, the application exports data in json format (file named `output.json`)

```python
from client import RufusClient
import os

key = os.getenv('RUFUS_API_KEY')
client = RufusClient(api_key=key)

instructions = "Get all the blog contents"
documents = client.scrape(instructions, "https://nepalprabin.github.io") # returns langchain Document object

# can use this document to create vectorstore

# Create a vector store from the documents
embedding_function = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embedding_function)

# Create a retriever
retriever = vectorstore.as_retriever()
...
...
```

## Technical Implementation Details

Under the hood, Rufus leverages smolagents' CodeAgent class which executes Python code snippets rather than generating text-based actions. This approach is approximately 30% more efficient than traditional tool-calling methods, requiring fewer LLM calls while maintaining higher performance on complex tasks.

The crawling process follows a sophisticated workflow:

- Initial URL Processing: The WebCrawlerTool begins by validating the starting URL and checking robots.txt compliance if enabled.

- Intelligent Depth Management: Rather than blindly crawling to a fixed depth, Rufus tracks the depth of each page and prioritizes more promising paths based on content relevance.

- HTML Cleaning and Extraction: ContentExtractionTool uses BeautifulSoup to intelligently clean HTML, removing non-content elements while preserving semantic structure.

- Relevance Scoring: The RelevanceEvaluationTool uses sentence-transformer models to create embeddings of both the user prompt and extracted content, calculating cosine similarity to determine relevance. Pages with scores above a configurable threshold are retained.

- Document Synthesis: Finally, DocumentSynthesizerTool organizes content into a cohesive structure, deduplicating content using MD5 hashing of paragraphs to avoid repetition.
