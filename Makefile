.PHONY: train clean test mlflow-ui

train:
	python src/run_pipeline.py

clean:
	rm -f models/*.joblib
	rm -f reports/*.png reports/*.csv reports/*.md

# Run unit tests quietly
test:
	pytest -q

# Launch local MLflow UI (uses MLFLOW_BACKEND or defaults to ./mlruns)
mlflow-ui:
	mlflow ui --backend-store-uri $${MLFLOW_BACKEND:-mlruns}
