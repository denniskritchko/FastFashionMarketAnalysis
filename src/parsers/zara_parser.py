from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

def parse_zara_csv_with_spark(input_path, output_path):
    """
    Parses the Zara CSV using Spark, cleans the data,
    and saves it to a Parquet file.
    """
    spark = SparkSession.builder \
        .appName("ZaraDataParser") \
        .getOrCreate()

    # Read the CSV
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Select and rename columns for consistency with our data model
    df_cleaned = df.select(
        col('url'),
        col('name'),
        col('sku'),
        col('price'),
        col('description').alias('materials'), # Using description as a proxy for materials for now
        col('brand')
    )

    # Save as Parquet for efficient processing
    df_cleaned.write.parquet(output_path, mode='overwrite')
    print(f"Cleaned Zara data saved to {output_path}")

    spark.stop()

if __name__ == '__main__':
    # Construct the full path to the input file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, '..', '..', 'data', 'store_zara.csv')
    output_dir = os.path.join(base_dir, '..', '..', 'data', 'zara_cleaned.parquet')
    
    parse_zara_csv_with_spark(input_file, output_dir)
