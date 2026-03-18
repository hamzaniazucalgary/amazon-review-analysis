"""Streamlit demo for Amazon Reviews Sentiment Analysis."""

import streamlit as st
import os
import pickle

st.set_page_config(page_title="Amazon Review Sentiment Analyzer", page_icon="📊", layout="wide")

st.title("Amazon Review Sentiment Analyzer")
st.markdown("Predict sentiment (positive/negative) for Amazon product reviews.")


@st.cache_resource
def load_sklearn_model():
    """Load the exported sklearn model."""
    model_path = os.path.join(os.path.dirname(__file__), "model_artifacts", "sklearn_model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_transformer_model():
    """Load DistilBERT model from HuggingFace."""
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception:
        return None


# Sidebar
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Choose model:", ["Logistic Regression (Fast)", "DistilBERT (Accurate)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Highlights")
st.sidebar.markdown("- **8 model configurations** benchmarked")
st.sidebar.markdown("- **90.2% accuracy** (best PySpark model)")
st.sidebar.markdown("- **3.6M reviews** in training data")
st.sidebar.markdown("- **PySpark + DistilBERT** pipeline")

# Main content
review_text = st.text_area("Paste your review here:", height=150,
                           placeholder="This product is amazing! Great quality and fast shipping...")

if st.button("Analyze Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing..."):
            if model_choice == "Logistic Regression (Fast)":
                model = load_sklearn_model()
                if model is None:
                    st.error("sklearn model not found. Run `python scripts/export_model_for_app.py` first.")
                else:
                    prediction = model.predict([review_text.lower()])[0]
                    proba = model.predict_proba([review_text.lower()])[0]
                    confidence = max(proba)
                    sentiment = "Positive" if prediction == 1 else "Negative"
                    color = "green" if prediction == 1 else "red"

                    st.markdown(f"### Sentiment: :{color}[{sentiment}]")
                    st.progress(confidence, text=f"Confidence: {confidence:.1%}")
            else:
                model = load_transformer_model()
                if model is None:
                    st.error("Transformer model not available. Install: `pip install transformers torch`")
                else:
                    result = model(review_text[:512])[0]
                    sentiment = "Positive" if result["label"] == "POSITIVE" else "Negative"
                    confidence = result["score"]
                    color = "green" if sentiment == "Positive" else "red"

                    st.markdown(f"### Sentiment: :{color}[{sentiment}]")
                    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

# Model details
with st.expander("Model Details"):
    if model_choice == "Logistic Regression (Fast)":
        st.markdown("""
        **Logistic Regression + CountVectorizer**
        - Trained on 3.6M Amazon reviews using PySpark
        - Features: CountVectorizer (65,536 vocab, minDF=5)
        - Test accuracy: ~90.2%
        - Inference: < 100ms
        """)
    else:
        st.markdown("""
        **DistilBERT (Transformer)**
        - Pre-trained on SST-2 sentiment data
        - 66M parameters, 6 transformer layers
        - Handles nuanced language and context
        - Inference: ~500ms on CPU
        """)
