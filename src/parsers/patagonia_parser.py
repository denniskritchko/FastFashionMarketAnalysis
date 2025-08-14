from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, lit
import os

def parse_patagonia_csv_with_spark(input_path, output_path):
	spark = SparkSession.builder.appName('PatagoniaDataParser').getOrCreate()
	df = spark.read.csv(input_path, header=True, inferSchema=True)
	# Expected columns likely: Name, Price, URL, Description, etc. Make robust
	name_col = 'name' if 'name' in df.columns else ('Name' if 'Name' in df.columns else None)
	price_col = 'price' if 'price' in df.columns else ('Price' if 'Price' in df.columns else None)
	url_col = 'url' if 'url' in df.columns else ('URL' if 'URL' in df.columns else None)
	desc_col = 'description' if 'description' in df.columns else ('Description' if 'Description' in df.columns else None)
	selected = df
	if price_col:
		selected = selected.withColumn('price_str', col(price_col).cast('string'))
		selected = selected.withColumn('price', regexp_extract(col('price_str'), r'(\d+\.?\d*)', 1).cast('float')).drop('price_str')
	else:
		selected = selected.withColumn('price', lit(None).cast('float'))
	selected = selected.withColumn('name', col(name_col)) if name_col else selected.withColumn('name', lit(None).cast('string'))
	selected = selected.withColumn('materials', col(desc_col)) if desc_col else selected.withColumn('materials', lit(None).cast('string'))
	selected = selected.withColumn('sku', lit(None).cast('string'))
	selected = selected.withColumn('url', col(url_col)) if url_col else selected.withColumn('url', lit(None).cast('string'))
	selected = selected.withColumn('brand', lit('Patagonia'))
	cleaned = selected.select('url','name','sku','price','materials','brand')
	cleaned.write.parquet(output_path, mode='overwrite')
	print(f"Cleaned Patagonia data saved to {output_path}")
	spark.stop()

if __name__ == '__main__':
	base_dir = os.path.dirname(os.path.abspath(__file__))
	input_file = os.path.join(base_dir, '..', '..', 'data', 'Patagonia_WebScrape_ClothingItems_v1.csv')
	output_dir = os.path.join(base_dir, '..', '..', 'data', 'patagonia_cleaned.parquet')
	parse_patagonia_csv_with_spark(input_file, output_dir)
