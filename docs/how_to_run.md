# How to Run This Project

## Google Colab (Recommended for Quick Start)

1. **Upload project** to Google Drive or clone from GitHub:
   ```
   !git clone https://github.com/<your-username>/amazon-reviews-sentiment-analysis.git
   %cd amazon-reviews-sentiment-analysis
   ```

2. **Open notebooks** in Colab -- they auto-detect the environment and install dependencies.

3. **Download data**:
   ```
   !python data/download_data.py --source huggingface
   ```

4. **Run notebooks in order**: `01_eda.ipynb` -> `02_pyspark_models.ipynb` -> `03_transformer_baseline.ipynb` -> `04_error_analysis.ipynb` -> `05_scalability.ipynb`

5. **Notes**:
   - Use **GPU runtime** for notebook 03 (DistilBERT)
   - Set `sample_fraction` in config for RAM-constrained environments
   - Colab sessions timeout after ~90 minutes of inactivity
   - Free tier has ~12GB RAM; use `sample_fraction=0.1` if you hit memory limits

---

## WSL / Linux (Local Development)

### Prerequisites

- Python 3.10+
- Java 11 (required by PySpark):
  ```bash
  sudo apt install openjdk-11-jre-headless
  ```
- 16GB+ RAM recommended (8GB minimum with sampling)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/amazon-reviews-sentiment-analysis.git
cd amazon-reviews-sentiment-analysis

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Download Data

```bash
python data/download_data.py
```

This downloads the Amazon Reviews Polarity dataset (~2GB) and converts it to Parquet format.

### Run Notebooks

```bash
jupyter lab notebooks/
```

Run notebooks in order: 01 -> 05.

### CLI Alternative

```bash
# Train all 8 model configurations
make train
# or
python scripts/train_all_models.py

# Run scalability benchmarks
python scripts/run_scalability_bench.py
```

### Streamlit App

```bash
# Export model for the app (after training)
python scripts/export_model_for_app.py

# Launch the demo
streamlit run app/streamlit_app.py
```

### MLflow UI

```bash
mlflow ui
# Open http://localhost:5000
```

### Transformer Notebook

```bash
# Install additional dependencies for notebook 03
uv pip install -r requirements-transformer.txt
# GPU recommended -- runs on CPU but much slower
```

---

## Docker

### Quick Start

```bash
docker-compose up
```

This starts:
- **Streamlit app** on `http://localhost:8501`
- **MLflow UI** on `http://localhost:5000`

### Mount Data Volume

```bash
docker-compose up -v ./data:/app/data
```

---

## Troubleshooting

### Java not found / PySpark fails to start
- Ensure Java 11 is installed: `java -version`
- Set `JAVA_HOME` if needed: `export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`

### Out of memory errors
- Reduce `sample_fraction` in `src/config.py` (e.g., `0.1` for 10% of data)
- Increase Spark driver memory in config: `driver_memory = "12g"`
- Close other applications to free RAM

### Parquet file not found
- Run `python data/download_data.py` first
- Check that `data/train.parquet` and `data/test.parquet` exist

### MLflow tracking errors
- Ensure `mlruns/` directory exists (created automatically on first run)
- Check that port 5000 is not in use: `lsof -i :5000`

### Streamlit model not found
- Run `python scripts/export_model_for_app.py` after training a model
- Check that `app/model_artifacts/sklearn_model.pkl` exists

### CUDA / GPU issues (Notebook 03)
- DistilBERT runs on CPU by default; GPU is optional but faster
- Install PyTorch with CUDA: `uv pip install torch --index-url https://download.pytorch.org/whl/cu118`
