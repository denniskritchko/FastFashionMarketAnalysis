import sys
import argparse
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download as nltk_download
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from math import erf, sqrt, log
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_INPUTS = [
    'data/everlane_products.csv',
    'data/reformation_products.csv',
    'data/Patagonia_WebScrape_ClothingItems_v1.csv',
    'data/store_zara.csv',
    'data/shein_sample.csv',
    'data/handm.csv',
]


MATERIAL_QUALITY_SCORES: Dict[str, float] = {
    # High quality / sustainable
    'tencel': 9.0,
    'lyocell': 9.0,
    'linen': 8.5,
    'hemp': 9.0,
    'organic cotton': 8.5,
    'recycled cotton': 8.0,
    'recycled polyester': 7.5,
    'merino wool': 8.0,
    'wool': 7.5,
    # Mid-tier
    'cotton': 6.0,
    'viscose': 5.0,
    'rayon': 5.0,
    # Lower quality / less sustainable
    'polyester': 3.0,
    'virgin polyester': 2.0,
    'acrylic': 2.5,
    'nylon': 3.5,
    'spandex': 3.0,
    'elastane': 3.0,
}


SUSTAINABILITY_KEYWORDS = [
    'eco', 'eco-friendly', 'conscious', 'sustainable', 'responsible', 'recycled', 'organic', 'green', 'earth', 'low-impact'
]


DURABLE_POS = [
    'durable', 'long lasting', 'long-lasting', 'well made', 'well-made', 'held up', 'high quality', 'sturdy', 'robust', 'lasted', 'lasts'
]
DURABLE_NEG = [
    'fell apart', 'tore', 'ripped', 'cheaply made', 'wore out', 'wore-out', 'broke', 'hole', 'pilled', 'shrunk', 'threads came loose'
]


FAST_FASHION_BRANDS = {'shein', 'zara', 'h&m', 'hm', 'h & m'}
SUSTAINABLE_BRANDS = {'patagonia', 'everlane', 'reformation'}


def _ensure_vader():
    try:
        SentimentIntensityAnalyzer()
    except Exception:
        nltk_download('vader_lexicon')


def _read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _infer_brand_from_path(path: str) -> Optional[str]:
    base = os.path.basename(path).lower()
    for b in list(FAST_FASHION_BRANDS | SUSTAINABLE_BRANDS):
        if b.replace(' ', '') in base.replace(' ', ''):
            return 'h&m' if b in {'hm', 'h & m'} else b
    return None


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_products(paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = _read_csv_safe(p)
        if df.empty:
            continue

        brand_col = _first_present(df, ['brand', 'Brand', 'seller', 'store'])
        price_col = _first_present(df, ['price', 'Price', 'product_price', 'sale_price'])
        materials_col = _first_present(df, ['materials', 'Materials', 'material', 'Material', 'composition', 'Composition'])
        desc_col = _first_present(df, ['description', 'Description', 'details', 'Details', 'product_description'])
        title_col = _first_present(df, ['title', 'Title', 'name', 'Name', 'product_name'])
        reviews_col = _first_present(df, ['review_text', 'reviews', 'Review', 'Comments', 'comment', 'Comment'])
        id_col = _first_present(df, ['id', 'ID', 'sku', 'product_id'])

        out = pd.DataFrame()
        out['source_file'] = os.path.basename(p)
        out['brand'] = df[brand_col] if brand_col else _infer_brand_from_path(p)
        out['title'] = df[title_col] if title_col else None
        out['description'] = df[desc_col] if desc_col else None
        out['materials'] = df[materials_col] if materials_col else None
        out['review_text'] = df[reviews_col] if reviews_col else None
        out['product_id'] = df[id_col] if id_col else None

        if price_col:
            prices = df[price_col].astype(str).str.replace(r"[^0-9\.]", '', regex=True)
            out['price'] = pd.to_numeric(prices, errors='coerce')
        else:
            out['price'] = np.nan

        # Fill brand if missing
        if 'brand' in out.columns:
            _inferred = _infer_brand_from_path(p)
            if _inferred is not None:
                out['brand'] = out['brand'].fillna(_inferred)

        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=['brand', 'title', 'description', 'materials', 'review_text', 'product_id', 'price', 'source_file'])
    data = pd.concat(frames, ignore_index=True)
    return data


def _durability_score(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    t = text.lower()
    pos = sum(1 for k in DURABLE_POS if k in t)
    neg = sum(1 for k in DURABLE_NEG if k in t)
    return pos - neg


def _parse_materials(composition: str) -> List[Tuple[str, float]]:
    if not isinstance(composition, str) or not composition.strip():
        return []
    comp = composition.lower()
    parts = re.split(r"[,/;+]", comp)
    results: List[Tuple[str, float]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.search(r"(\d{1,3})\s*%\s*([a-z\s]+)", part)
        if m:
            pct = min(max(float(m.group(1)) / 100.0, 0.0), 1.0)
            mat = m.group(2).strip()
            results.append((mat, pct))
        else:
            # No percentage: try to extract a material token; assume small share
            mat = re.sub(r"[^a-z\s]", '', part).strip()
            if mat:
                results.append((mat, 0.0))
    # Normalize percentages if some given; otherwise distribute equally
    total = sum(p for _, p in results)
    if total > 0:
        norm = [(mat, p / total) for mat, p in results]
    else:
        n = len(results) if results else 0
        norm = [(mat, 1.0 / n) for mat, _ in results] if n > 0 else []
    return norm


def _material_quality_score(comp: str) -> float:
    parsed = _parse_materials(comp)
    if not parsed:
        return np.nan
    score = 0.0
    for mat, frac in parsed:
        mkey = mat.strip()
        # map variants
        mkey = mkey.replace('elastane', 'spandex')
        if 'organic cotton' in mkey:
            base = 'organic cotton'
        elif 'cotton' in mkey:
            base = 'cotton'
        elif 'virgin polyester' in mkey:
            base = 'virgin polyester'
        elif 'recycled polyester' in mkey:
            base = 'recycled polyester'
        else:
            base = mkey
        # find best matching key
        best_key = None
        for k in MATERIAL_QUALITY_SCORES:
            if k in base:
                best_key = k
                break
        s = MATERIAL_QUALITY_SCORES.get(best_key, 5.0)  # neutral default
        score += frac * s
    return score


def _brand_category(brand: Optional[str]) -> Optional[str]:
    if not isinstance(brand, str) or not brand:
        return None
    b = brand.lower()
    if b in FAST_FASHION_BRANDS:
        return 'fast_fashion'
    if b in SUSTAINABLE_BRANDS:
        return 'sustainable'
    return None


def _sentiment_scores(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    _ensure_vader()
    sia = SentimentIntensityAnalyzer()
    return float(sia.polarity_scores(text)['compound'])


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Compose a text field for sentiment and durability from review_text if present, else fallback to description
    df['text_for_sentiment'] = df['review_text'].fillna('') + ' ' + df['description'].fillna('')
    df['durability_score'] = df['text_for_sentiment'].apply(_durability_score)
    df['objective_quality_score'] = df['materials'].apply(_material_quality_score)
    df['sentiment_score'] = df['text_for_sentiment'].apply(_sentiment_scores)
    df['brand_category'] = df['brand'].apply(_brand_category)

    # Proxy cost per wear; clip to avoid division by zero or negative
    eps = 1e-6
    denom = df['durability_score'].astype(float)
    denom = denom.where(denom > 0.0, np.nan)
    df['proxy_cpw'] = df['price'] / denom

    return df


def _analyze_cpw(df: pd.DataFrame) -> pd.DataFrame:
    # Average proxy CPW by category
    subset = df[['brand_category', 'proxy_cpw']].dropna()
    return subset.groupby('brand_category', as_index=False)['proxy_cpw'].mean().rename(columns={'proxy_cpw': 'avg_proxy_cpw'})


def _analyze_correlation(df: pd.DataFrame) -> Tuple[float, float, pd.DataFrame]:
    """Return Pearson r and p-value for quality vs sentiment, plus the subset used.
    Tries scipy.stats.pearsonr if available; otherwise uses Fisher z approximation
    with normal distribution (adequate for large n).
    """
    sub = df[['objective_quality_score', 'sentiment_score']].dropna()
    if len(sub) == 0:
        return float('nan'), float('nan'), sub
    x = sub['objective_quality_score'].to_numpy()
    y = sub['sentiment_score'].to_numpy()
    try:
        from scipy.stats import pearsonr  # type: ignore
        r, p = pearsonr(x, y)
        return float(r), float(p), sub
    except Exception:
        # Fallback: compute r and approximate p via Fisher z-transform
        r = float(sub['objective_quality_score'].corr(sub['sentiment_score']))
        n = len(sub)
        if n > 3 and abs(r) < 0.999999:
            z = 0.5 * log((1 + r) / (1 - r))
            se = 1.0 / sqrt(max(n - 3, 1))
            z0 = z / se
            # Two-sided p from normal approximation
            Phi = lambda t: 0.5 * (1 + erf(t / sqrt(2)))
            p = 2 * (1 - Phi(abs(z0)))
        else:
            p = float('nan')
        return r, float(p), sub


def _analyze_claims(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    def has_claim(text: str) -> bool:
        if not isinstance(text, str):
            return False
        t = text.lower()
        return any(k in t for k in SUSTAINABILITY_KEYWORDS)

    sub = df[['description', 'objective_quality_score']].copy()
    sub['has_claim'] = df['description'].apply(has_claim)
    sub = sub.dropna(subset=['objective_quality_score'])
    sub['low_quality'] = sub['objective_quality_score'].apply(lambda x: bool(x <= 4.0))
    # Rates among claimed
    result = sub.groupby('has_claim', as_index=False)['low_quality'].mean().rename(columns={'low_quality': 'share_low_quality'})
    # Point-biserial correlation between claim (binary) and quality (continuous)
    if not sub.empty:
        corr = float(sub['has_claim'].astype(int).corr(sub['objective_quality_score']))
    else:
        corr = float('nan')
    return result, sub[['has_claim', 'objective_quality_score']], corr


def _cluster_products(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    features = df[['price', 'sentiment_score', 'durability_score', 'objective_quality_score']].copy()
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        df['cluster'] = np.nan
        df['cluster_name'] = np.nan
        return df
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = km.fit_predict(X)
    # Map labels back to df index alignment
    features['cluster'] = labels
    df = df.join(features['cluster'], how='left')

    # Create descriptive cluster names based on per-cluster means vs global quantiles
    global_q = {
        'price': df['price'].quantile([0.33, 0.66]).to_dict(),
        'sentiment_score': df['sentiment_score'].quantile([0.33, 0.66]).to_dict(),
        'durability_score': df['durability_score'].quantile([0.33, 0.66]).to_dict(),
        'objective_quality_score': df['objective_quality_score'].quantile([0.33, 0.66]).to_dict(),
    }

    def _level(val: float, q: Dict[float, float]) -> str:
        if pd.isna(val):
            return 'n/a'
        lo = q.get(0.33, val)
        hi = q.get(0.66, val)
        if val < lo:
            return 'low'
        if val > hi:
            return 'high'
        return 'mid'

    cmeans = df.groupby('cluster')[['price', 'sentiment_score', 'durability_score', 'objective_quality_score']].mean()
    cluster_name_map: Dict[int, str] = {}
    for cid, row in cmeans.iterrows():
        desc = (
            f"price:{_level(row['price'], global_q['price'])}, "
            f"quality:{_level(row['objective_quality_score'], global_q['objective_quality_score'])}, "
            f"durability:{_level(row['durability_score'], global_q['durability_score'])}, "
            f"sentiment:{_level(row['sentiment_score'], global_q['sentiment_score'])}"
        )
        cluster_name_map[cid] = f"C{cid} ({desc})"

    df['cluster_name'] = df['cluster'].map(cluster_name_map)
    return df


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_plots(df: pd.DataFrame, corr_df: pd.DataFrame, corr_p: float, claims_df: pd.DataFrame, claims_corr: float, out_dir: str):
    _ensure_dir(out_dir)

    sns.set_theme(style='whitegrid', context='talk')

    # Scatter: objective_quality_score vs sentiment_score with colored regression line
    if not corr_df.empty:
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = sns.regplot(
            data=corr_df,
            x='objective_quality_score',
            y='sentiment_score',
            scatter_kws={'alpha': 0.4, 's': 24, 'color': '#4C72B0'},
            line_kws={'color': '#C44E52', 'lw': 2.5}
        )
        title = 'Sentiment vs Objective Material Quality'
        if pd.notnull(corr_p):
            title += f" (p={corr_p:.3g})"
        ax1.set_title(title)
        ax1.set_xlabel('Objective Material Quality (weighted)')
        ax1.set_ylabel('VADER Sentiment (compound)')
        plt.tight_layout()
        fig1.savefig(os.path.join(out_dir, 'sentiment_vs_quality.png'), dpi=220)
        plt.close(fig1)

    # Bar: CPW by category (ensure both categories present)
    cpw = df[['brand_category', 'proxy_cpw']].copy()
    cpw = cpw[cpw['proxy_cpw'].notna()]
    categories = ['fast_fashion', 'sustainable']
    cpw_means = cpw.groupby('brand_category')['proxy_cpw'].mean()
    cpw_counts = cpw.groupby('brand_category')['proxy_cpw'].size()
    cpw_plot = pd.DataFrame({
        'brand_category': categories,
        'avg_proxy_cpw': [cpw_means.get(cat, np.nan) for cat in categories],
        'n': [int(cpw_counts.get(cat, 0)) for cat in categories],
    })
    fig2 = plt.figure(figsize=(7, 5))
    ax2 = sns.barplot(data=cpw_plot, x='brand_category', y='avg_proxy_cpw', palette=['#55A868', '#8172B2'])
    ax2.set_title('Average Proxy Cost-Per-Wear by Category')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Avg Proxy CPW (lower is better)')
    for i, row in cpw_plot.iterrows():
        y = 0.0 if pd.isna(row['avg_proxy_cpw']) else float(row['avg_proxy_cpw'])
        ax2.text(i, y if y is not None else 0.0, f"n={row['n']}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'avg_proxy_cpw_by_category.png'), dpi=220)
    plt.close(fig2)

    # Cluster scatter with descriptive legend
    cl = df.dropna(subset=['cluster_name'])
    if not cl.empty:
        fig3 = plt.figure(figsize=(8, 6))
        ax3 = sns.scatterplot(
            data=cl,
            x='objective_quality_score',
            y='sentiment_score',
            hue='cluster_name',
            palette='tab10',
            alpha=0.6,
            s=30,
        )
        ax3.set_title('Clusters (projected)')
        ax3.set_xlabel('Objective Material Quality (weighted)')
        ax3.set_ylabel('VADER Sentiment (compound)')
        ax3.legend(title='Cluster summary', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig3.savefig(os.path.join(out_dir, 'clusters_scatter.png'), dpi=220, bbox_inches='tight')
        plt.close(fig3)

    # Claims vs material quality: distribution comparison
    if claims_df is not None and not claims_df.empty:
        claims_plot = claims_df.copy()
        claims_plot['claim'] = claims_plot['has_claim'].map({True: 'Claimed sustainable', False: 'No claim'})
        fig4 = plt.figure(figsize=(8, 6))
        ax4 = sns.violinplot(
            data=claims_plot,
            x='claim', y='objective_quality_score',
            inner='quartile', palette=['#4C72B0', '#55A868']
        )
        n0 = int((claims_plot['claim'] == 'No claim').sum())
        n1 = int((claims_plot['claim'] == 'Claimed sustainable').sum())
        title = f"Material Quality by Sustainability Claim (r={claims_corr:.2f}, n_no={n0}, n_claim={n1})"
        ax4.set_title(title)
        ax4.set_xlabel('Marketing claim')
        ax4.set_ylabel('Objective Material Quality (weighted)')
        plt.tight_layout()
        fig4.savefig(os.path.join(out_dir, 'claims_vs_material_quality.png'), dpi=220)
        plt.close(fig4)

        # Simple-titled variant per request
        fig5 = plt.figure(figsize=(8, 6))
        ax5 = sns.violinplot(
            data=claims_plot,
            x='claim', y='objective_quality_score',
            inner='quartile', palette=['#4C72B0', '#55A868']
        )
        ax5.set_title('Material Quality by Sustainability Claim')
        ax5.set_xlabel('Marketing claim')
        ax5.set_ylabel('Objective Material Quality (weighted)')
        plt.tight_layout()
        fig5.savefig(os.path.join(out_dir, 'claims_vs_material_quality_simple.png'), dpi=220)
        plt.close(fig5)



def _write_outputs(df: pd.DataFrame, cpw: pd.DataFrame, claims: pd.DataFrame, corr_val: float, corr_p: float, corr_n: int, claims_corr: float, out_dir: str):
    _ensure_dir(out_dir)
    df.to_csv(os.path.join(out_dir, 'products_enriched.csv'), index=False)
    cpw.to_csv(os.path.join(out_dir, 'proxy_cpw_by_category.csv'), index=False)
    claims.to_csv(os.path.join(out_dir, 'claims_vs_quality.csv'), index=False)
    with open(os.path.join(out_dir, 'correlation.txt'), 'w') as f:
        f.write(f"pearson(objective_quality_score, sentiment_score): r={corr_val:.4f}, p={corr_p:.4g}, n={corr_n}\n")
        f.write(f"point_biserial(has_claim, objective_quality_score) = {claims_corr:.4f}\n")


def _parse_args(argv):
    p = argparse.ArgumentParser(description='Research questions analysis over product CSVs')
    p.add_argument('--inputs', nargs='*', default=DEFAULT_INPUTS)
    p.add_argument('--k', type=int, default=4, help='KMeans clusters')
    p.add_argument('--output_base', default='artifacts/research_analysis')
    p.add_argument('--run_tag', default=None)
    p.add_argument('--no_suffix', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv or [])
    out_base = args.output_base
    if not args.no_suffix:
        tag = args.run_tag if args.run_tag else datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        out_base = f"{args.output_base}/run={tag}"

    # Load and normalize
    products = _normalize_products(args.inputs)
    if products.empty:
        print('No product data found. Check input paths.')
        return 1

    enriched = _prepare_features(products)

    # CPW by category
    cpw = _analyze_cpw(enriched)

    # Claims vs quality
    claims, claims_detail, claims_corr = _analyze_claims(enriched)

    # Correlation sentiment vs quality
    corr_val, corr_p, corr_df = _analyze_correlation(enriched)

    # Clustering
    clustered = _cluster_products(enriched, k=args.k)

    # Outputs
    _write_outputs(clustered, cpw, claims, corr_val, corr_p, len(corr_df), claims_corr, os.path.join(out_base, 'tables'))
    _save_plots(clustered, corr_df, corr_p, claims_detail, claims_corr, os.path.join(out_base, 'plots'))

    print(f"Wrote analysis outputs to {out_base}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


