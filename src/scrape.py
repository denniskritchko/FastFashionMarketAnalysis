import argparse
import os
from scrapers.hm_scraper import scrape_hm
from scrapers.zara_scraper import scrape_zara
from scrapers.patagonia_scraper import scrape_patagonia
# Import other scrapers as they are developed

def main():
    parser = argparse.ArgumentParser(description="Scrape product data from fashion websites.")
    parser.add_argument('--brand', type=str, required=True, help="The brand to scrape (e.g., 'hm', 'zara', 'patagonia').")
    parser.add_argument('--query', type=str, required=True, help="The search query for products (e.g., 'dresses').")
    parser.add_argument('--num_products', type=int, default=50, help="The number of products to scrape.")
    args = parser.parse_args()

    # Check for ScraperAPI key if using a scraper that requires it
    if args.brand.lower() in ['patagonia']: # Add other ScraperAPI brands here
        if not os.getenv("SCRAPERAPI_KEY"):
            print("Error: SCRAPERAPI_KEY environment variable not set.")
            print("Please set it to your ScraperAPI key to run this scraper.")
            return

    if args.brand.lower() == 'hm':
        print("Starting H&M scraper...")
        df = scrape_hm(args.query, args.num_products)
        output_path = f"data/{args.brand.lower()}_{args.query.lower()}_products.csv"
        df.to_csv(output_path, index=False)
        print(f"Scraped data saved to {output_path}")
    elif args.brand.lower() == 'zara':
        print("Starting Zara scraper...")
        df = scrape_zara(args.query, args.num_products)
        output_path = f"data/{args.brand.lower()}_{args.query.lower()}_products.csv"
        df.to_csv(output_path, index=False)
        print(f"Scraped data saved to {output_path}")
    elif args.brand.lower() == 'patagonia':
        print("Starting Patagonia scraper...")
        df = scrape_patagonia(args.query, args.num_products)
        output_path = f"data/{args.brand.lower()}_{args.query.lower()}_products.csv"
        df.to_csv(output_path, index=False)
        print(f"Scraped data saved to {output_path}")
    # Add other brands here
    else:
        print(f"Scraper for brand '{args.brand}' not implemented yet.")

if __name__ == '__main__':
    main()
