# pip install Rufus
from client import RufusClient
import os 

key = os.getenv('RUFUS_API_KEY')
client = RufusClient(api_key=key)

instructions = "Get all the blog contents"
documents = client.scrape(instructions, "https://nepalprabin.github.io")



"""
  synthesized_document = document_synthesizer_tool(                                                                                                                                                                           
      relevant_content=extracted_content,                                                                                                                                                                                     
      prompt="Get all the blog contents",                                                                                                                                                                                     
      output_format="json"                                                                                                                                                                                                    
  )                                                                                                                                                                                                                           
  final_answer(synthesized_document)   
"""