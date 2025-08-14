import sys
import argparse
import re
from datetime import datetime
from typing import Optional, Dict

from pyspark.sql import SparkSession, functions as F, types as T


def _build_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def _parse_args(argv):
    p = argparse.ArgumentParser(description='Spark-based cleaning and sentiment for reddit-brands submissions')
    p.add_argument('--input', default='reddit-brands/submissions', help='Path to local parquet submissions folder')
    p.add_argument('--output_base', default='artifacts/reddit_brands_sentiment', help='Base output directory')
    p.add_argument('--run_tag', default=None, help='Optional run tag; defaults to UTC timestamp')
    p.add_argument('--no_suffix', action='store_true', help='Do not append run suffix to output path')
    p.add_argument('--coalesce', type=int, default=4, help='Coalesce factor for outputs (0 to skip)')
    p.add_argument('--min_len', type=int, default=5, help='Minimum cleaned text token length to keep')
    return p.parse_args(argv)


def _clean_text_column(df, text_col: str, min_len: int):
    url_pattern = r"https?://\S+|www\.\S+"
    user_pattern = r"/u/\S+|u/\S+|@\S+"
    html_pattern = r"<[^>]+>"
    md_link_pattern = r"\[[^\]]+\]\([^\)]+\)"  # [text](url)

    cleaned = F.lower(F.col(text_col))
    cleaned = F.regexp_replace(cleaned, md_link_pattern, ' ')
    cleaned = F.regexp_replace(cleaned, url_pattern, ' ')
    cleaned = F.regexp_replace(cleaned, user_pattern, ' ')
    cleaned = F.regexp_replace(cleaned, html_pattern, ' ')
    cleaned = F.regexp_replace(cleaned, r"[^a-z0-9\s]", ' ')  # keep alnum and space
    cleaned = F.regexp_replace(cleaned, r"\s+", ' ')
    cleaned = F.trim(cleaned)

    # Keep only rows with enough tokens
    token_count = F.size(F.split(cleaned, r"\s+"))
    return df.withColumn('text_clean', cleaned).where(token_count >= F.lit(min_len))


def _vader_udf():
    # Lazy init per executor
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk import download as nltk_download

    sia_ref: Dict[str, Optional[SentimentIntensityAnalyzer]] = {'sia': None}

    def ensure_sia():
        if sia_ref['sia'] is None:
            try:
                sia_ref['sia'] = SentimentIntensityAnalyzer()
            except Exception:
                nltk_download('vader_lexicon')
                sia_ref['sia'] = SentimentIntensityAnalyzer()
        return sia_ref['sia']

    def score(text: Optional[str]):
        if text is None:
            return (0.0, 0.0, 0.0, 0.0)
        sia = ensure_sia()
        s = sia.polarity_scores(text)
        return (float(s.get('neg', 0.0)), float(s.get('neu', 0.0)), float(s.get('pos', 0.0)), float(s.get('compound', 0.0)))

    return F.udf(score, T.StructType([
        T.StructField('neg', T.DoubleType()),
        T.StructField('neu', T.DoubleType()),
        T.StructField('pos', T.DoubleType()),
        T.StructField('compound', T.DoubleType()),
    ]))


def main(argv=None):
    args = _parse_args(argv or [])
    spark = _build_spark('reddit-brands sentiment')

    base_output = args.output_base
    if not args.no_suffix:
        suffix = args.run_tag if args.run_tag else datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        base_output = f"{args.output_base}/run={suffix}"

    # Read local parquet submissions
    df = spark.read.parquet(args.input)

    # Construct raw text field from available columns
    title = F.coalesce(F.col('title'), F.lit(''))
    selftext = F.coalesce(F.col('selftext'), F.lit(''))
    text = F.concat_ws(' ', title, selftext)
    df = df.withColumn('text_raw', F.trim(text))

    # Clean text
    df = _clean_text_column(df, 'text_raw', args.min_len)

    # Sentiment
    vader = _vader_udf()
    df = df.withColumn('vader', vader(F.col('text_clean')))
    df = df.withColumn('vader_neg', F.col('vader.neg')) \
           .withColumn('vader_neu', F.col('vader.neu')) \
           .withColumn('vader_pos', F.col('vader.pos')) \
           .withColumn('vader_compound', F.col('vader.compound')) \
           .drop('vader')

    # Write detailed results as parquet
    detailed = df.select(
        'id', 'subreddit', 'year', 'month', 'title', 'selftext', 'text_clean',
        'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound'
    )
    if args.coalesce and args.coalesce > 0:
        detailed = detailed.coalesce(args.coalesce)
    detailed.write.mode('error').parquet(f"{base_output}/detailed_parquet")

    # Aggregated CSV by year/month/subreddit
    agg = (
        df.groupBy('year', 'month', 'subreddit')
          .agg(
              F.count(F.lit(1)).alias('n_posts'),
              F.avg('vader_compound').alias('avg_compound'),
              F.avg('vader_pos').alias('avg_pos'),
              F.avg('vader_neg').alias('avg_neg'),
              F.avg('vader_neu').alias('avg_neu'),
          )
          .orderBy('year', 'month', 'subreddit')
    )
    agg_out = agg.coalesce(1) if (args.coalesce and args.coalesce > 0) else agg
    agg_out.write.mode('error').option('header', True).csv(f"{base_output}/aggregates_csv")

    spark.stop()


if __name__ == '__main__':
    main(sys.argv[1:])


