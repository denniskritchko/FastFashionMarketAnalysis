import sys
import argparse
from typing import List
from pyspark.sql import SparkSession, functions as F, types as T
from datetime import datetime

# HDFS base paths (Hive-partitioned by year/month)
REDDIT_SUBMISSIONS_PATH = '/courses/datasets/reddit_submissions_repartitioned/'
REDDIT_COMMENTS_PATH = '/courses/datasets/reddit_comments_repartitioned/'

DEFAULT_OUTPUT = 'reddit-subset'
DEFAULT_SUBS = [
    'mensfashion',
    'malefashionadvice',
    'womensfashion',
    'femalefashionadvice',
]

# Full schemas from template (enables fast read and avoids inference)
COMMENTS_SCHEMA = T.StructType([
    T.StructField('archived', T.BooleanType()),
    T.StructField('author', T.StringType()),
    T.StructField('author_flair_css_class', T.StringType()),
    T.StructField('author_flair_text', T.StringType()),
    T.StructField('body', T.StringType()),
    T.StructField('controversiality', T.LongType()),
    T.StructField('created_utc', T.StringType()),
    T.StructField('distinguished', T.StringType()),
    T.StructField('downs', T.LongType()),
    T.StructField('edited', T.StringType()),
    T.StructField('gilded', T.LongType()),
    T.StructField('id', T.StringType()),
    T.StructField('link_id', T.StringType()),
    T.StructField('name', T.StringType()),
    T.StructField('parent_id', T.StringType()),
    T.StructField('retrieved_on', T.LongType()),
    T.StructField('score', T.LongType()),
    T.StructField('score_hidden', T.BooleanType()),
    T.StructField('subreddit', T.StringType()),
    T.StructField('subreddit_id', T.StringType()),
    T.StructField('ups', T.LongType()),
    T.StructField('year', T.IntegerType()),
    T.StructField('month', T.IntegerType()),
])

SUBMISSIONS_SCHEMA = T.StructType([
    T.StructField('archived', T.BooleanType()),
    T.StructField('author', T.StringType()),
    T.StructField('author_flair_css_class', T.StringType()),
    T.StructField('author_flair_text', T.StringType()),
    T.StructField('created', T.LongType()),
    T.StructField('created_utc', T.StringType()),
    T.StructField('distinguished', T.StringType()),
    T.StructField('domain', T.StringType()),
    T.StructField('downs', T.LongType()),
    T.StructField('edited', T.BooleanType()),
    T.StructField('from', T.StringType()),
    T.StructField('from_id', T.StringType()),
    T.StructField('from_kind', T.StringType()),
    T.StructField('gilded', T.LongType()),
    T.StructField('hide_score', T.BooleanType()),
    T.StructField('id', T.StringType()),
    T.StructField('is_self', T.BooleanType()),
    T.StructField('link_flair_css_class', T.StringType()),
    T.StructField('link_flair_text', T.StringType()),
    T.StructField('media', T.StringType()),
    T.StructField('name', T.StringType()),
    T.StructField('num_comments', T.LongType()),
    T.StructField('over_18', T.BooleanType()),
    T.StructField('permalink', T.StringType()),
    T.StructField('quarantine', T.BooleanType()),
    T.StructField('retrieved_on', T.LongType()),
    T.StructField('saved', T.BooleanType()),
    T.StructField('score', T.LongType()),
    T.StructField('secure_media', T.StringType()),
    T.StructField('selftext', T.StringType()),
    T.StructField('stickied', T.BooleanType()),
    T.StructField('subreddit', T.StringType()),
    T.StructField('subreddit_id', T.StringType()),
    T.StructField('thumbnail', T.StringType()),
    T.StructField('title', T.StringType()),
    T.StructField('ups', T.LongType()),
    T.StructField('url', T.StringType()),
    T.StructField('year', T.IntegerType()),
    T.StructField('month', T.IntegerType()),
])


def parse_args(argv: List[str]):
    p = argparse.ArgumentParser(description='Extract subreddit/year/month filtered Reddit data (fast).')
    p.add_argument('--subs', nargs='*', default=DEFAULT_SUBS, help='Subreddits (names without r/)')
    p.add_argument('--start_year', type=int, default=2023)
    p.add_argument('--end_year', type=int, default=2024)
    p.add_argument('--months', nargs='*', type=int, default=[1, 2, 3], help='Months 1-12')
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--include_submissions', action='store_true', help='Include submissions')
    p.add_argument('--include_comments', action='store_true', help='Include comments')
    p.add_argument('--run_tag', default=None, help='Optional tag for run subdir; default is UTC timestamp')
    p.add_argument('--no_suffix', action='store_true', help='Do not append run suffix to output path')
    p.add_argument('--coalesce', type=int, default=8, help='Number of output files per dataset (0 to skip)')
    args = p.parse_args(argv)
    if not (args.include_comments or args.include_submissions):
        args.include_comments = True
        args.include_submissions = True
    return args


def main(argv: List[str]):
    args = parse_args(argv)

    spark = SparkSession.builder.appName('reddit extracter').getOrCreate()

    subs_lc = [s.lower() for s in args.subs]
    months = list(set(args.months))

    # Compute a temporary unique output directory to avoid overwrites by default
    base_output = args.output
    if not args.no_suffix:
        suffix = args.run_tag if args.run_tag else datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        base_output = f"{args.output}/run={suffix}"

    if args.include_submissions:
        reddit_submissions = spark.read.json(REDDIT_SUBMISSIONS_PATH, schema=SUBMISSIONS_SCHEMA)
        sub_df = (
            reddit_submissions
            .where((F.col('year') >= F.lit(args.start_year)) & (F.col('year') <= F.lit(args.end_year)))
            .where(F.col('month').isin(months))
            .where(F.lower(F.col('subreddit')).isin(subs_lc))
        )
        if args.coalesce and args.coalesce > 0:
            sub_df = sub_df.coalesce(args.coalesce)
        (sub_df.write
            .mode('error')
            .json(f"{base_output}/submissions", compression='gzip')
        )

    if args.include_comments:
        reddit_comments = spark.read.json(REDDIT_COMMENTS_PATH, schema=COMMENTS_SCHEMA)
        com_df = (
            reddit_comments
            .where((F.col('year') >= F.lit(args.start_year)) & (F.col('year') <= F.lit(args.end_year)))
            .where(F.col('month').isin(months))
            .where(F.lower(F.col('subreddit')).isin(subs_lc))
        )
        if args.coalesce and args.coalesce > 0:
            com_df = com_df.coalesce(args.coalesce)
        (com_df.write
            .mode('error')
            .json(f"{base_output}/comments", compression='gzip')
        )

    spark.stop()


if __name__ == '__main__':
    main(sys.argv[1:])