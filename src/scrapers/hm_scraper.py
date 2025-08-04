from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import pandas as pd
import time
import urllib.parse

def scrape_hm(search_query, num_products=50):
    """
    Scrapes product data from H&M for a given search query.
    """
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 10)
    
    # Construct the search URL directly
    encoded_query = urllib.parse.quote_plus(search_query)
    search_url = f"https://www2.hm.com/en_us/search-results.html?q={encoded_query}"
    
    product_urls = []
    all_details = []

    try:
        print(f"Navigating to search results for '{search_query}'...")
        driver.get(search_url)

        # Handle cookie consent
        try:
            cookie_button = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
            cookie_button.click()
            print("Accepted cookie consent.")
        except TimeoutException:
            print("No cookie consent button found or it was not clickable.")

        # Collect product URLs
        print("Collecting product URLs...")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.products-listing")))
        
        while len(product_urls) < num_products:
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            links = soup.select("li.product-item > a")
            
            if not links and not product_urls:
                print("No products found on the page.")
                break

            for link in links:
                if len(product_urls) < num_products:
                    url = "https://www2.hm.com" + link['href']
                    if url not in product_urls:
                        product_urls.append(url)
            
            if len(product_urls) >= num_products:
                break
                
            # Scroll to load more products
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2) # Wait for new products to load

        print(f"Found {len(product_urls)} product URLs.")

        # Scrape details for each product
        for url in product_urls:
            try:
                print(f"Scraping details from: {url}")
                driver.get(url)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                details = {'url': url}
                details['name'] = soup.select_one("h1").text.strip()
                details['price'] = soup.select_one(".price-value").text.strip()
                
                # Extract materials
                materials = []
                for dt in soup.select("dt"):
                    if "Composition" in dt.text:
                        materials = [li.text.strip() for li in dt.find_next_sibling("dd").select("li")]
                details['materials'] = materials
                
                all_details.append(details)
            except Exception as e:
                print(f"Could not scrape details from {url}: {e}")

    finally:
        driver.quit()

    return pd.DataFrame(all_details)

if __name__ == '__main__':
    df = scrape_hm("dresses", num_products=20)
    df.to_csv('data/hm_products.csv', index=False)
    print("Scraped data saved to data/hm_products.csv")
