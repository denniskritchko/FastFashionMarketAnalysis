import os
import re
import urllib.parse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from random import uniform
from dotenv import load_dotenv
import argparse

load_dotenv()

API_KEY_ENV = 'SCRAPERAPI_KEY'
SEARCH_BASE = 'https://www.thereformation.com/search?q={}&page={}'

DEFAULT_QUERIES = [
	'dress','jeans','tops','sweater','jacket','coat','skirt','pants','t-shirt','shirt','blouse','activewear','swim','shoes','accessories'
]

SESSION = requests.Session()


def _wrap(url: str) -> str:
	a = os.getenv(API_KEY_ENV)
	if not a:
		raise RuntimeError('SCRAPERAPI_KEY not set')
	params = {'api_key': a, 'url': url, 'render': 'true', 'country_code': 'us', 'retry': 'true'}
	return 'http://api.scraperapi.com?' + urllib.parse.urlencode(params)


def _get(url: str) -> str:
	backoff = 1.0
	for attempt in range(6):
		try:
			resp = SESSION.get(_wrap(url), timeout=60)
			if resp.ok and resp.text:
				return resp.text
		except requests.RequestException:
			pass
		sleep(backoff + uniform(0, 0.7))
		backoff = min(backoff * 2.0, 8.0)
	raise RuntimeError(f'Failed to fetch: {url}')


def gather_product_urls(query: str, max_pages: int = 20) -> list:
	urls = []
	for page in range(1, max_pages + 1):
		html = _get(SEARCH_BASE.format(urllib.parse.quote_plus(query), page))
		soup = BeautifulSoup(html, 'html.parser')
		page_links = []
		for a in soup.select('a'):
			href = a.get('href', '')
			if '/products/' in href:
				if href.startswith('/'):
					page_links.append('https://www.thereformation.com' + href)
				else:
					page_links.append(href)
		before = len(urls)
		urls.extend(page_links)
		urls = list(dict.fromkeys(urls))
		if len(urls) == before:
			break
		sleep(0.3)
	return urls


def parse_product(url: str) -> dict:
	try:
		html = _get(url)
		s = BeautifulSoup(html, 'html.parser')
		title_el = s.find('h1')
		name = title_el.get_text(strip=True) if title_el else None
		price_el = s.select_one('[data-testid="product-price"], .price, [class*="Price"]')
		price_text = price_el.get_text(strip=True) if price_el else None
		price_num = None
		if price_text:
			m = re.search(r'(\d+\.?\d*)', price_text)
			price_num = float(m.group(1)) if m else None
		materials = None
		for sel in ['[data-testid="accordion"]','[data-testid*="details"]','[class*="details"]','[class*="accordion"]']:
			blk = s.select_one(sel)
			if blk:
				materials = blk.get_text(" ", strip=True)
				break
		imgs = [img.get('src') for img in s.select('img') if img.get('src','').startswith('http')]
		return {'url': url, 'name': name, 'price_raw': price_text, 'price': price_num, 'materials': materials, 'images': ','.join(dict.fromkeys(imgs))[:2000], 'brand': 'Reformation'}
	except Exception:
		return {'url': url, 'name': None, 'price_raw': None, 'price': None, 'materials': None, 'images': None, 'brand': 'Reformation'}


def scrape_reformation_queries(queries=None, max_pages: int = 20, max_workers: int = 8, out_csv: str | None = None) -> pd.DataFrame:
	if not queries:
		queries = DEFAULT_QUERIES
	all_urls = []
	for q in queries:
		all_urls.extend(gather_product_urls(q, max_pages=max_pages))
	all_urls = list(dict.fromkeys(all_urls))
	rows = []
	written = 0
	def flush():
		nonlocal rows, written
		if out_csv and rows:
			df_chunk = pd.DataFrame(rows)
			df_chunk.to_csv(out_csv, mode='a', index=False, header=(written == 0))
			written += len(rows)
			rows = []
	with ThreadPoolExecutor(max_workers=max_workers) as ex:
		futs = [ex.submit(parse_product, u) for u in all_urls]
		for i, f in enumerate(as_completed(futs), 1):
			rows.append(f.result())
			if i % 200 == 0:
				flush()
	flush()
	return pd.read_csv(out_csv) if out_csv and os.path.exists(out_csv) else pd.DataFrame(rows)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--queries', nargs='*', default=DEFAULT_QUERIES)
	ap.add_argument('--max_pages', type=int, default=30)
	ap.add_argument('--workers', type=int, default=10)
	ap.add_argument('--output', default='data/reformation_products.csv')
	args = ap.parse_args()
	if not os.getenv(API_KEY_ENV):
		print('Set SCRAPERAPI_KEY to run this scraper')
		return
	_ = scrape_reformation_queries(args.queries, args.max_pages, args.workers, out_csv=args.output)
	print(f'Saved {args.output}')

if __name__ == '__main__':
	main()
