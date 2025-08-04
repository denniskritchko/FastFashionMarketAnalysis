from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, udf, when
from pyspark.sql.types import StringType
import json

def parse_shein_csv_with_spark(input_path, output_path):
    """
    Parses the semicolon-delimited SHEIN CSV using Spark, cleans the data,
    and saves it to a Parquet file.
    """
    spark = SparkSession.builder \
        .appName("SheinDataParser") \
        .getOrCreate()

    # Read the CSV with iso-8859-1 encoding to handle special characters
    df = spark.read.csv(input_path, sep=';', header=True, encoding='iso-8859-1')

    # Extract numeric part of price
    df = df.withColumn('price_numeric_string', regexp_extract(col('price'), r'(\d+\.?\d*)', 1))

    # Cast to float, replacing empty/malformed strings with null
    df = df.withColumn(
        'price_float',
        when(col('price_numeric_string') != '', col('price_numeric_string').cast('float'))
        .otherwise(None)
    ).drop('price_numeric_string')

    # Define a UDF to extract material composition
    def extract_composition(description):
        if not isinstance(description, str):
            return None
        try:
            # The description is a string representation of a list of dicts
            s = description.strip()
            if not (s.startswith('[') and s.endswith(']')):
                return None
                
            desc_list = json.loads(s.replace("'", '"'))
            for item in desc_list:
                if 'Composition' in item:
                    return item['Composition']
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None
        return None

    extract_composition_udf = udf(extract_composition, StringType())
    df = df.withColumn('materials', extract_composition_udf(df['description']))

    # Select and rename columns
    df_cleaned = df.select(
        col('url'),
        col('name'),
        col('sku'),
        col('price_float').alias('price'),
        col('size'),
        col('brand'),
        col('materials')
    )

    # Save as Parquet for efficient processing
    df_cleaned.write.parquet(output_path, mode='overwrite')
    print(f"Cleaned SHEIN data saved to {output_path}")

    spark.stop()

if __name__ == '__main__':
    parse_shein_csv_with_spark('data/shein_sample.csv', 'data/shein_cleaned.parquet')
