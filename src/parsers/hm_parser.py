from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, lit, when
import os

def parse_hm_csv_with_spark(input_path, output_path):
	spark = SparkSession.builder.appName('HMDataParser').getOrCreate()
	df = spark.read.csv(input_path, header=True, inferSchema=True)
	# Try to standardize columns: url, name, sku, price, description/materials, brand
	price_col = 'price' if 'price' in df.columns else 'Price' if 'Price' in df.columns else None
	name_col = 'name' if 'name' in df.columns else ('product_name' if 'product_name' in df.columns else None)
	desc_col = 'description' if 'description' in df.columns else ('product_description' if 'product_description' in df.columns else None)
	sku_col = 'sku' if 'sku' in df.columns else ('product_id' if 'product_id' in df.columns else None)
	url_col = 'url' if 'url' in df.columns else ('product_url' if 'product_url' in df.columns else None)
	selected = df
	if price_col:
		selected = selected.withColumn('price_str', col(price_col).cast('string'))
		selected = selected.withColumn('price_num_str', regexp_extract(col('price_str'), r'(\d+\.?\d*)', 1))
		selected = selected.withColumn('price', when(col('price_num_str') == '', lit(None).cast('float')).otherwise(col('price_num_str').cast('float')))
		selected = selected.drop('price_str', 'price_num_str')
	else:
		selected = selected.withColumn('price', lit(None).cast('float'))
	selected = selected.withColumn('name', col(name_col)) if name_col else selected.withColumn('name', lit(None).cast('string'))
	selected = selected.withColumn('materials', col(desc_col)) if desc_col else selected.withColumn('materials', lit(None).cast('string'))
	selected = selected.withColumn('sku', col(sku_col)) if sku_col else selected.withColumn('sku', lit(None).cast('string'))
	selected = selected.withColumn('url', col(url_col)) if url_col else selected.withColumn('url', lit(None).cast('string'))
	selected = selected.withColumn('brand', lit('H&M'))
	cleaned = selected.select('url','name','sku','price','materials','brand')
	cleaned.write.parquet(output_path, mode='overwrite')
	print(f"Cleaned H&M data saved to {output_path}")
	spark.stop()

if __name__ == '__main__':
	base_dir = os.path.dirname(os.path.abspath(__file__))
	input_file = os.path.join(base_dir, '..', '..', 'data', 'handm.csv')
	output_dir = os.path.join(base_dir, '..', '..', 'data', 'hm_cleaned.parquet')
	parse_hm_csv_with_spark(input_file, output_dir)
