import os
import pickle
from pathlib import Path

DATA_FOLDER = Path("../tmp/data")
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

class HelpdeskScraperPipeline:

    def __init__(self):
        # Create a dictionary to store items temporarily
        self.items = {}
        # Define a checkpoint interval
        self.checkpoint_interval = 500  # Store after every 500 items
        self.items_scraped = 0

    def process_item(self, item, spider):
        key = item['key']

        # If the item's key is already in the dictionary, it means we've visited this page
        if key in self.items:
            return item

        # Otherwise, add the item to the dictionary
        self.items[key] = item.to_dict()
        self.items_scraped += 1

        # Check if it's time to store the items
        if self.items_scraped % self.checkpoint_interval == 0:
            self.store_items()

        return item


    def store_items(self):
        # Define the file path        
        file_path = DATA_FOLDER / 'scraped_data.pkl'

        # Check if the file already exists
        if os.path.exists(file_path):
            # If it exists, merge the new data with the existing data
            with open(file_path, 'rb') as f:
                existing_data = pickle.load(f)
            existing_data.update(self.items)
            with open(file_path, 'wb') as f:
                pickle.dump(existing_data, f)
        else:
            # If it doesn't exist, just store the current data
            with open(file_path, 'wb') as f:
                pickle.dump(self.items, f)

        # Clear the current dictionary of items
        self.items = {}

    def close_spider(self, spider):
        # Store any remaining items when the spider closes
        if self.items:
            self.store_items()