import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import config as config

class CollegeWebsiteLoader:
    def __init__(self, base_url=config.WEBSITE_URL):
        self.base_url = base_url
        self.visited_urls = set()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def get_all_links(self, url):
        """Extract all links from a webpage that belong to the same domain."""
        if url in self.visited_urls:
            return []
        
        self.visited_urls.add(url)
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/'):
                    full_url = f"{self.base_url}{href}"
                    if full_url not in self.visited_urls:
                        links.append(full_url)
                elif href.startswith(self.base_url):
                    if href not in self.visited_urls:
                        links.append(href)
            
            return links
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return []
    
    def crawl_website(self, max_pages=50):
        """Crawl the website starting from the base URL."""
        pages_to_visit = [self.base_url]
        documents = []
        
        while pages_to_visit and len(self.visited_urls) < max_pages:
            url = pages_to_visit.pop(0)
            try:
                loader = WebBaseLoader(url)
                page_docs = loader.load()
                documents.extend(page_docs)
                
                new_links = self.get_all_links(url)
                pages_to_visit.extend(new_links)
            except Exception as e:
                print(f"Failed to load {url}: {e}")
        
        return self.text_splitter.split_documents(documents)