import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib.parse
import os

def get_scraperapi_url(url):
    """Constructs the ScraperAPI URL for a given URL."""
    api_key = os.getenv("SCRAPERAPI_KEY")
    if not api_key:
        raise ValueError("SCRAPERAPI_KEY environment variable not set.")
    payload = {'api_key': api_key, 'url': url}
    return 'http://api.scraperapi.com?' + urllib.parse.urlencode(payload)

def scrape_patagonia(search_query, num_products=50):
    """
    Scrapes product data from Patagonia for a given search query using ScraperAPI.
    """
    encoded_query = urllib.parse.quote_plus(search_query)
    search_url = f"https://www.patagonia.com/search/?q={encoded_query}"
    
    product_urls = []
    all_details = []

    print(f"Navigating to search results for '{search_query}'...")
    response = requests.get(get_scraperapi_url(search_url))
    soup = BeautifulSoup(response.text, 'html.parser')

    # Collect product URLs
    print("Collecting product URLs...")
    product_links = soup.select("a.product-tile__link")
    for link in product_links:
        if len(product_urls) < num_products:
            url = "https://www.patagonia.com" + link['href']
            if url not in product_urls:
                product_urls.append(url)

    print(f"Found {len(product_urls)} product URLs.")

    # Scrape details for each product
    for url in product_urls:
        try:
            print(f"Scraping details from: {url}")
            response = requests.get(get_scraperapi_url(url))
            soup = BeautifulSoup(response.text, 'html.parser')
            
            details = {'url': url}
            details['name'] = soup.select_one("h1.product-name").text.strip()
            details['price'] = soup.select_one("span.price").text.strip()
            
            # Extract materials
            materials_section = soup.find('div', id='product-materials')
            materials = [li.text.strip() for li in materials_section.select("li")] if materials_section else []
            details['materials'] = materials
            
            all_details.append(details)
        except Exception as e:
            print(f"Could not scrape details from {url}: {e}")

    return pd.DataFrame(all_details)

if __name__ == '__main__':
    # You will need to set the SCRAPERAPI_KEY environment variable to run this
    if os.getenv("SCRAPERAPI_KEY"):
        df = scrape_patagonia("jackets", num_products=20)
        df.to_csv('data/patagonia_products.csv', index=False)
        print("Scraped data saved to data/patagonia_products.csv")
    else:
        print("Please set the SCRAPERAPI_KEY environment variable to run this scraper.")
