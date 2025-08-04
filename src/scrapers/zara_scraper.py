from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import pandas as pd
import time
import urllib.parse

def scrape_zara(search_query, num_products=50):
    """
    Scrapes product data from Zara for a given search query.
    """
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 10)
    
    encoded_query = urllib.parse.quote_plus(search_query)
    search_url = f"https://www.zara.com/us/en/search?searchTerm={encoded_query}"
    
    product_urls = []
    all_details = []

    try:
        print(f"Navigating to search results for '{search_query}'...")
        driver.get(search_url)

        # Handle cookie consent
        try:
            cookie_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept All')]")))
            cookie_button.click()
            print("Accepted cookie consent.")
        except TimeoutException:
            print("No cookie consent button found or it was not clickable.")

        # Wait for product grid to be visible
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.product-grid__product-list")))
        
        # Scroll to load products
        print("Collecting product URLs...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        while len(product_urls) < num_products:
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            links = soup.select("li.product-grid__product-list-item > a")
            
            for link in links:
                url = link['href']
                if url not in product_urls:
                    product_urls.append(url)
                    if len(product_urls) >= num_products:
                        break
            
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3) # Give time for new products to load
            
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        print(f"Found {len(product_urls)} product URLs.")

        # Scrape details for each product
        for url in product_urls:
            try:
                print(f"Scraping details from: {url}")
                driver.get(url)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                details = {'url': url}
                details['name'] = soup.select_one("h1.product-detail-info__header-name").text.strip()
                details['price'] = soup.select_one("p.product-detail-info__price > span").text.strip()
                
                # Extract materials
                materials = []
                for p in soup.select("p"):
                    if "Composition" in p.text:
                        materials = [li.text.strip() for li in p.find_next_sibling("ul").select("li")]
                details['materials'] = materials
                
                all_details.append(details)
            except Exception as e:
                print(f"Could not scrape details from {url}: {e}")

    finally:
        driver.quit()

    return pd.DataFrame(all_details)

if __name__ == '__main__':
    df = scrape_zara("dresses", num_products=20)
    df.to_csv('data/zara_products.csv', index=False)
    print("Scraped data saved to data/zara_products.csv")
