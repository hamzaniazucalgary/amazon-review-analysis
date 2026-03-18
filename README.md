# Amazon Reviews Sentiment Analysis

**Python 3.10+ | PySpark 3.5 | License: MIT**

Large-scale sentiment analysis on Amazon product reviews using PySpark distributed computing and transformer models.

## Highlights

- **8 model configurations** benchmarked on 3.6M Amazon reviews
- **Best model**: Logistic Regression + CountVectorizer -- 90.2% accuracy, 96.2% AUC
- **PySpark** distributed processing for scalable training
- **DistilBERT** transformer baseline for comparison
- **MLflow** experiment tracking with full metric logging
- **Streamlit** interactive demo app
- **Scalability analysis** from 36K to 3.6M training samples


## Quick Start

### Option 1: Google Colab (Easiest)

Open the notebooks directly in Colab -- they auto-install dependencies and download data.

### Option 2: Local (WSL/Linux)

```bash
# Setup
git clone https://github.com/<your-username>/amazon-reviews-sentiment-analysis.git
cd amazon-reviews-sentiment-analysis
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Download data
python data/download_data.py

# Train all models
python scripts/train_all_models.py

# Launch Streamlit demo
streamlit run app/streamlit_app.py
```

### Option 3: Docker

```bash
docker-compose up
# Streamlit: http://localhost:8501
# MLflow:    http://localhost:5000
```

See [docs/how_to_run.md](docs/how_to_run.md) for detailed instructions.

## Project Structure

```
amazon-reviews-sentiment-analysis/
|-- app/
|   |-- streamlit_app.py              # Interactive demo
|   |-- model_artifacts/              # Exported models
|-- data/
|   |-- download_data.py              # Dataset download script
|-- docs/
|   |-- how_to_run.md                 # Detailed setup guide
|   |-- *.png                         # Generated visualizations
|-- notebooks/
|   |-- 01_eda.ipynb                  # Exploratory data analysis
|   |-- 02_pyspark_models.ipynb       # Multi-model benchmarking
|   |-- 03_transformer_baseline.ipynb # DistilBERT baseline
|   |-- 04_error_analysis.ipynb       # Error analysis & failure patterns
|   |-- 05_scalability.ipynb          # Scalability analysis
|-- scripts/
|   |-- train_all_models.py           # CLI training script
|   |-- export_model_for_app.py       # Export model for Streamlit
|   |-- run_scalability_bench.py      # Scalability benchmarks
|-- src/
|   |-- config.py                     # Project configuration
|   |-- data_loader.py                # Spark session & data loading
|   |-- feature_engineering.py        # Feature extraction (CV, TF-IDF, n-grams)
|   |-- models.py                     # Model definitions & pipeline builder
|   |-- evaluation.py                 # Metrics, plots, comparison tables
|   |-- experiment_tracker.py         # MLflow integration
|   |-- utils.py                      # Timer, helpers
|-- tests/
|   |-- test_data_loader.py
|   |-- test_feature_engineering.py
|   |-- test_evaluation.py
|-- requirements.txt
|-- requirements-transformer.txt
|-- Makefile
|-- docker-compose.yml
|-- Dockerfile
|-- README.md
```

## Technologies

- **PySpark 3.5** -- Distributed data processing and ML pipelines
- **scikit-learn** -- Model export for lightweight inference
- **HuggingFace Transformers** -- DistilBERT sentiment baseline
- **MLflow** -- Experiment tracking and model registry
- **Streamlit** -- Interactive web demo
- **Matplotlib / Seaborn** -- Visualizations
- **Pandas / NumPy** -- Data manipulation

## Dataset

**Amazon Reviews Polarity** (Zhang et al., 2015)

- 3,600,000 training reviews + 400,000 test reviews
- Binary sentiment: positive (4-5 stars) / negative (1-2 stars)
- Source: [HuggingFace Datasets](https://huggingface.co/datasets/amazon_polarity)

## License

MIT
