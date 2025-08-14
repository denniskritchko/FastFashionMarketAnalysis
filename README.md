# Fast Fashion Market Analysis

An empirical analysis of fast fashion's effects on the market using PySpark and scraped web data. This project compares fast fashion brands (H&M, Zara, SHEIN) with sustainable brands (Patagonia, Everlane, Reformation) to analyze price, quality, consumer satisfaction, and environmental impact relationships.

Note: React and Flask interactive dashboard is still in progress.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Data Sources](#data-sources)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Keys Required](#api-keys-required)
- [Dependencies](#dependencies)

## Project Overview

This project analyzes the relationship between price, quality, consumer satisfaction, and environmental impact for fast fashion versus sustainable clothing brands. It uses multiple data sources including:

- **Product Data**: Web scraping from brand websites and Kaggle datasets
- **Consumer Sentiment**: Reddit posts and comments mentioning target brands
- **Big Data Processing**: Apache Spark for data cleaning and analysis
- **Sentiment Analysis**: VADER for text analysis

### Target Brands
- **Fast Fashion**: H&M, Zara, SHEIN
- **Sustainable**: Patagonia, Everlane, Reformation

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/denniskritchko/FastFashionMarketAnalysis.git
cd FastFashionMarketAnalysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys
Create a `.env` file in the project root with your API keys:
```bash
# Required API Key
SCRAPERAPI_KEY=your_scraperapi_key_here

# Optional: Kaggle API (for dataset downloads)
# Create ~/.kaggle/kaggle.json with your Kaggle credentials
```

### 4. Set Up Kaggle API (Optional)
If you want to download datasets directly:
```bash
mkdir ~/.kaggle
# Add your kaggle.json file to ~/.kaggle/
```

## Data Sources

### Product Data
- **SHEIN**: Kaggle dataset (`shein_sample.csv`)
- **Zara**: Kaggle dataset (`store_zara.csv`)
- **H&M**: Kaggle dataset (`handm.csv`)
- **Patagonia**: Kaggle dataset (`Patagonia_WebScrape_ClothingItems_v1.csv`)
- **Reformation**: Web scraping via ScraperAPI
- **Everlane**: Web scraping via ScraperAPI

### Consumer Sentiment Data
- **Reddit Data**: SFU HDFS cluster (submissions and comments)
- **Subreddits**: r/mensfashion, r/malefashionadvice, r/womensfashion, r/femalefashionadvice

## Usage

### 1. Data Collection

#### Web Scraping (Reformation & Everlane)
```bash
# Scrape Reformation products
python3 src/scrapers/reformation_scraper.py --max_pages 30 --workers 8 --output data/reformation_products.csv

# Scrape Everlane products
python3 src/scrapers/everlane_scraper.py --max_pages 30 --workers 8 --output data/everlane_products.csv
```

**Arguments:**
- `--queries`: List of search terms (default: dress, jeans, tops, etc.)
- `--max_pages`: Maximum pages to scrape per query (default: 30)
- `--workers`: Number of concurrent workers (default: 10)
- `--output`: Output CSV file path

#### Download Kaggle Datasets
```bash
kaggle datasets download -d user/dataset --unzip -p data/
```

### 2. Data Processing with Spark

#### Parse and Clean Product Data
```bash
# Process SHEIN data
python3 src/parsers/shein_parser.py

# Process Zara data
python3 src/parsers/zara_parser.py

# Process H&M data
python3 src/parsers/hm_parser.py

# Process Patagonia data
python3 src/parsers/patagonia_parser.py
```

#### Extract Reddit Data (SFU Cluster)
```bash
# Extract Reddit submissions and comments mentioning target brands
# Must be SSHed into the cluster 
ssh -p24 <USERID>@cluster.cs.sfu.ca 

scp src/scrapers/extract.py cluster.cs.sfu.ca:

spark-submit src/scrapers/extract.py \
  --start_year 2023 \
  --end_year 2024 \
  --months 01,02,03,04,05,06,07,08,09,10,11,12 \
  --subreddits mensfashion,malefashionadvice,womensfashion,femalefashionadvice \
  --include_comments \
  --include_submissions \
  --output /path/to/output/directory
```

### 3. Sentiment Analysis
```bash
# Analyze sentiment on Reddit data
python3 src/sentiment/analyze.py \
  --input data/reddit_brands_filtered.csv \
  --text_column body \
  --output data/reddit_sentiment_analysis.csv \
  --use_gemini  # Optional: use Google Gemini for enhanced analysis
```

### 4. Run Analysis
```bash
# Run the main analysis pipeline
python3 src/main.py
```


## Project Structure

```
FastFashionMarketAnalysis/
├── data/                          # Data files (CSV, Parquet)
│   ├── shein_sample.csv
│   ├── store_zara.csv
│   ├── handm.csv
│   ├── Patagonia_WebScrape_ClothingItems_v1.csv
│   ├── reformation_products.csv
│   ├── everlane_products.csv
│   └── *_cleaned.parquet          # Spark-processed data
├── src/
│   ├── scrapers/                  # Web scraping modules
│   │   ├── reformation_scraper.py
│   │   ├── everlane_scraper.py
│   │   ├── extract.py             # Reddit data extraction
│   │   └── ...
│   ├── parsers/                   # Spark data processing
│   │   ├── shein_parser.py
│   │   ├── zara_parser.py
│   │   ├── hm_parser.py
│   │   └── patagonia_parser.py
│   ├── sentiment/                 # Sentiment analysis
│   │   └── analyze.py
│   ├── api/                       # Flask backend
│   │   └── app.py
│   ├── dashboard/                 # React frontend
│   │   ├── package.json
│   │   ├── index.html
│   │   └── src/
│   └── main.py                    # Main analysis pipeline
├── docs/                          # Documentation
│   ├── REPORT.md                  # Research report
│   ├── RUBRIC.md                  # Project rubric
│   └── EXAMPLE.md                 # Example specifications
├── artifacts/                     # Analysis outputs (gitignored)
├── requirements.txt               # Python dependencies
├── .env                          # API keys (gitignored)
└── README.md                     # This file
```

## 🔑 API Keys Required

### ScraperAPI
- **Purpose**: Web scraping Reformation and Everlane websites
- **Setup**: Sign up at [scraperapi.com](https://scraperapi.com)
- **Environment Variable**: `SCRAPERAPI_KEY`

### Kaggle
- **Purpose**: Download pre-existing datasets
- **Setup**: Create account at [kaggle.com](https://kaggle.com) and download API token
- **File**: `~/.kaggle/kaggle.json`

## Dependencies

### Python Packages
```
pandas              # Data manipulation
numpy               # Numerical computing
scikit-learn        # Machine learning
pyspark             # Big data processing
beautifulsoup4      # HTML parsing
requests            # HTTP requests
selenium            # Web automation (legacy)
nltk                # Natural language processing
vaderSentiment      # Sentiment analysis
google-generativeai # Google Gemini API
Flask               # Web API framework
python-dotenv       # Environment variable management
kaggle              # Kaggle API client
matplotlib          # Data visualization
seaborn             # Statistical visualization
plotly              # Interactive plots
```

### System Requirements
- **Python**: 3.8+
- **Java**: 8+ (for Apache Spark)
- **Node.js**: 16+ (for React dashboard)
- **Memory**: 8GB+ recommended for Spark processing

## Troubleshooting

### Common Issues

1. **ScraperAPI Timeouts**: Reduce `--workers` and `--max_pages` parameters
2. **Spark Memory Issues**: Increase Spark driver memory or reduce data size
3. **Git Large File Errors**: Large datasets are excluded via `.gitignore`
4. **API Key Errors**: Ensure `.env` file is properly configured

### Performance Tips

- Use fewer workers for web scraping to avoid rate limiting
- Process data in chunks for large datasets
- Use Parquet format for efficient Spark processing
- Enable Spark caching for repeated operations

## License

This project is for academic research purposes. Please respect the terms of service for all data sources used.

## Contributing

This is an academic project. For questions or issues, please message me on the provided links on my profile.