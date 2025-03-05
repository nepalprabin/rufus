from smolagents import Tool
from typing import Dict
from datetime import datetime
import urllib.parse

class WebCrawlerTool(Tool):
    name = "web_crawler_tool"
    description = """
    Crawls a website starting from a given URL, following links up to a specified depth.
    Handles rate limiting and respects robots.txt.
    """
    inputs = {
        "start_url": {"type": "string", "description": "The URL to start crawling from"},
        "max_pages": {"type": "integer", "description": "Maximum number of pages to crawl", "nullable": True
},
        "max_depth": {"type": "integer", "description": "Maximum depth to crawl", "nullable": False
},
        "respect_robots": {"type": "boolean", "description": "Whether to respect robots.txt", "nullable": True
},
    }
    output_type = "any"

    def forward(self, start_url: str, max_pages: int = 50, max_depth: int = 3, respect_robots: bool = True) -> Dict:
        import requests
        from bs4 import BeautifulSoup
        from urllib.robotparser import RobotFileParser
        import time
        
        results = {
            "pages": {},
            "stats": {
                "pages_discovered": 0,
                "pages_crawled": 0,
                "start_time": datetime.now().isoformat(),
            }
        }
        
        visited_urls = set()
        queue = [(start_url, 0)]  # (url, depth)
        
        # Set up robots.txt parser if needed
        rp = None
        if respect_robots:
            parsed_url = urllib.parse.urlparse(start_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            rp = RobotFileParser()
            rp.set_url(f"{base_url}/robots.txt")
            try:
                rp.read()
            except:
                print("Warning: Could not read robots.txt")
        
        while queue and len(visited_urls) < max_pages:
            url, depth = queue.pop(0)
            
            # Skip if already visited or too deep
            if url in visited_urls or depth > max_depth:
                continue
            
            # Check robots.txt
            if rp and not rp.can_fetch("*", url):
                print(f"Skipping {url} (disallowed by robots.txt)")
                continue
            
            visited_urls.add(url)
            
            try:
                # Fetch the page
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    continue
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = soup.title.string if soup.title else "No title"
                
                # Save page content
                results["pages"][url] = {
                    "title": title,
                    "html": response.text,
                    "depth": depth
                }
                
                results["stats"]["pages_crawled"] += 1
                
                # Extract links for further crawling
                if depth < max_depth:
                    links = []
                    for a_tag in soup.find_all('a', href=True):
                        href = a_tag['href']
                        if href.startswith(('http://', 'https://')):
                            links.append(href)
                        elif not href.startswith(('#', 'javascript:', 'mailto:')):
                            # Handle relative URLs
                            base_url = urllib.parse.urljoin(url, '')
                            absolute_url = urllib.parse.urljoin(base_url, href)
                            links.append(absolute_url)
                    
                    # Add new links to queue
                    for link in links:
                        if link not in visited_urls:
                            queue.append((link, depth + 1))
                    
                    results["stats"]["pages_discovered"] += len(links)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error crawling {url}: {e}")
        
        # Update stats
        results["stats"]["end_time"] = datetime.now().isoformat()
        results["stats"]["total_pages_crawled"] = len(results["pages"])
        
        return results