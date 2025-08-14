import os
import re
from typing import List, Dict
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download as nltk_download

try:
	import google.generativeai as genai
	_HAS_GEMINI = True
except Exception:
	_HAS_GEMINI = False


def _ensure_vader():
	try:
		SentimentIntensityAnalyzer()
	except Exception:
		nltk_download('vader_lexicon')


def load_texts_from_csvs(csv_paths: List[str], text_cols: List[str]) -> pd.DataFrame:
	frames = []
	for p in csv_paths:
		df = pd.read_csv(p)
		present_cols = [c for c in text_cols if c in df.columns]
		if not present_cols:
			continue
		df['_source'] = os.path.basename(p)
		df['_text'] = df[present_cols].astype(str).agg(' '.join, axis=1)
		frames.append(df[['_source', '_text']])
	return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['_source','_text'])


def vader_scores(texts: List[str]) -> List[Dict[str, float]]:
	_ensure_vader()
	sia = SentimentIntensityAnalyzer()
	return [sia.polarity_scores(t if isinstance(t, str) else '') for t in texts]


def _gemini_init():
	if not _HAS_GEMINI:
		return None
	api_key = os.getenv('GEMINI_API_KEY')
	if not api_key:
		return None
	genai.configure(api_key=api_key)
	try:
		model = genai.GenerativeModel('gemini-1.5-flash')
		return model
	except Exception:
		return None


def gemini_sentiment_batch(texts: List[str]) -> List[float]:
	model = _gemini_init()
	if model is None:
		return []
	scores = []
	prompt = "Classify overall sentiment as a real number in [-1,1] (negative to positive). Text: \n\n{}\n\nScore:" 
	for t in texts:
		try:
			resp = model.generate_content(prompt.format(t[:5000]))
			val = re.findall(r"-?\d+\.?\d*", resp.text or '')
			scores.append(float(val[0]) if val else 0.0)
		except Exception:
			scores.append(0.0)
	return scores


def run_sentiment(csv_paths: List[str], text_cols: List[str]) -> pd.DataFrame:
	data = load_texts_from_csvs(csv_paths, text_cols)
	if data.empty:
		return data
	v_scores = vader_scores(data['_text'].tolist())
	data['vader_neg'] = [s['neg'] for s in v_scores]
	data['vader_pos'] = [s['pos'] for s in v_scores]
	data['vader_compound'] = [s['compound'] for s in v_scores]
	g_scores = gemini_sentiment_batch(data['_text'].tolist())
	if g_scores:
		data['gemini_score'] = g_scores
	return data

if __name__ == '__main__':
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('--inputs', nargs='+', required=True)
	p.add_argument('--text_cols', nargs='+', default=['body','title','selftext'])
	p.add_argument('--output', required=True)
	args = p.parse_args()
	out = run_sentiment(args.inputs, args.text_cols)
	out.to_csv(args.output, index=False)
	print(f"Wrote {len(out)} rows -> {args.output}")
