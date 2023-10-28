import datetime
import hashlib
import logging
import os
import pickle
import re
from urllib.parse import urljoin

from scrapy.spiders import Spider
from scrapy import Request

from helpdesk_scraper.items import ScraperItem


class HelpdeskSpider(Spider):
    name = 'helpdesk'
    allowed_domains = ['helpdesk.ugent.be']
    start_urls = ['https://helpdesk.ugent.be']

    # Define the deny patterns for the LinkExtractor
    deny_patterns = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load existing data if available
        file_path = 'scraped_data.pkl'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.visited_urls = set(pickle.load(f).keys())
                logging.info(f"Loaded {len(self.visited_urls)} existing URLs")
        else:
            self.visited_urls = set()

    def get_key_from_url(self, url):
        # Generate a key from the URL
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def parse_headers(self, response):
        content_type = response.headers.get('Content-Type')
        if content_type and 'text' in content_type.decode('utf-8'):
            yield response.request.replace(callback=self.parse, method='GET')            
        else:            
            logging.info(f"Skipping non-text URL: {response.url}")

    def parse(self, response):
        # Find all links on the page
        base_url = response.url        
        links_to_follow = []
        for link in response.css('a::attr(href)').getall():
            if link.startswith("mailto:"):
                continue

            if not link.startswith("https://"):
                link = urljoin(base_url, link)            

            if self.get_key_from_url(link) in self.visited_urls:
                continue
                
            links_to_follow.append(link)

        metadata = {
            "title": response.css('title::text').get(),
            "url": response.url,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        key = self.get_key_from_url(response.url)
        content = response.text

        # Yield the scraped data
        yield ScraperItem(key=key, content=content, metadata=metadata)
        
        # Add the URL's key to the visited set
        self.visited_urls.add(key)

        for link in links_to_follow:
            yield Request(link, callback=self.parse_headers, method='HEAD')        