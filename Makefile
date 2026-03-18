.PHONY: setup setup-transformer data train scalability export-model app mlflow docker test clean

setup:
	uv venv .venv
	uv pip install -r requirements.txt

setup-transformer: setup
	uv pip install -r requirements-transformer.txt

data:
	python data/download_data.py --source huggingface --format parquet

train:
	python scripts/train_all_models.py

scalability:
	python scripts/run_scalability_bench.py

export-model:
	python scripts/export_model_for_app.py

app:
	streamlit run app/streamlit_app.py

mlflow:
	mlflow ui --port 5000

docker:
	docker-compose up --build

test:
	pytest tests/ -v

clean:
	rm -rf spark-warehouse metastore_db derby.log mlruns __pycache__ .ipynb_checkpoints
	find . -type d -name __pycache__ -exec rm -rf {} +
