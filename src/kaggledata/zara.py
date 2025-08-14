import os
import ast
import pandas as pd 
from PIL import Image

BASE_DIR = os.path.join('/kaggle', 'input', 'zara-products')

df = pd.read_csv(os.path.join(BASE_DIR, 'store_zara.csv'))

df.head()

first_product = df.iloc[0]
first_product

