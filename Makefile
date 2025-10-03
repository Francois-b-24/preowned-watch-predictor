.PHONY: train clean app
train:
	python src/run_pipeline.py
clean:
	rm -f models/*.joblib
	rm -f reports/*.png reports/*.csv reports/*.md
app:
	streamlit run src/streamlit_app.py
