import os
import json
from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Simple in-process loaders; in production consider caching or a DB

def load_parquet_dir(path: str) -> pd.DataFrame:
	try:
		return pd.read_parquet(path)
	except Exception:
		return pd.DataFrame()

@app.get('/api/health')
def health():
	return jsonify({'status': 'ok'})

@app.get('/api/brand-metrics')
def brand_metrics():
	# Merge product data (price etc.) and reddit sentiment outputs if available
	frames = []
	for p in [
		'data/zara_cleaned.parquet',
		'data/shein_cleaned.parquet',
		'data/reddit_sentiment.csv',
	]:
		if os.path.exists(p):
			try:
				if p.endswith('.csv'):
					frames.append(pd.read_csv(p))
				else:
					frames.append(pd.read_parquet(p))
			except Exception:
				pass
	if not frames:
		return jsonify({'brands': []})
	df = pd.concat(frames, ignore_index=True, sort=False)
	# Example aggregation: average price and avg vader_compound per brand
	if 'brand' not in df.columns:
		df['brand'] = df.get('_source', '').astype(str)
	agg = df.groupby('brand', dropna=False).agg({
		'price': 'mean',
		'vader_compound': 'mean',
	}).reset_index().rename(columns={'price': 'avg_price', 'vader_compound': 'avg_sentiment'})
	return agg.to_json(orient='records')

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5001')), debug=True)
