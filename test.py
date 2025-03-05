# pip install Rufus
from client import RufusClient
import os 

key = os.getenv('RUFUS_API_KEY')
client = RufusClient(api_key=key)

instructions = "Get all the blog contents"
documents = client.scrape(instructions, "https://nepalprabin.github.io")