import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import requests
import streamlit as st

from src.fnd.models.pipeline import FakeNewsPipeline, PipelineArtifacts

st.set_page_config(page_title="News Detector", layout="wide")

def get_api_url():
    """Try deployed API first, fallback to localhost if not reachable."""
    api_url = st.secrets.get("API_URL", "http://localhost:8000")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return api_url
    except:
        pass
    return "http://localhost:8000"

API_URL = get_api_url()
MODEL_ARTIFACTS = PipelineArtifacts(
    model_path=Path(os.getenv("FND_MODEL_PATH", "models/baseline.joblib")),
    vectorizer_path=Path(os.getenv("FND_VECTOR_PATH", "models/vectorizer.joblib")),
    label_path=Path(os.getenv("FND_LABELS_PATH", "models/labels.json")),
    metrics_path=Path("models/metrics.json"),
)


@st.cache_resource(show_spinner=False)
def load_local_pipeline() -> Optional[FakeNewsPipeline]:
    if MODEL_ARTIFACTS.model_path.exists():
        return FakeNewsPipeline.load(MODEL_ARTIFACTS)
    return None


def run_local(text: str) -> dict:
    pipeline = load_local_pipeline()
    if not pipeline:
        raise FileNotFoundError("Model artifacts missing. Train a model or set FND_API_URL.")
    label = pipeline.predict([text])[0]
    confidence = float(pipeline.predict_proba([text])[0].max())
    return {"label": label, "confidence": confidence, "top_tokens": {}}


def run_api(text: str) -> dict:
    response = requests.post(API_URL, json={"text": text}, timeout=30)
    response.raise_for_status()
    return response.json()


st.title("Fake News Detection Dashboard")
mode = st.radio("Inference Mode", ["API", "Local"], horizontal=True)
text = st.text_area(
    "Paste article or claim",
    height=220,
    placeholder="Paste a news article, statement, or social media post...",
)
threshold = st.slider("Alert threshold", 0.5, 0.99, 0.8, 0.01)

if st.button("Analyze", type="primary"):
    if not text or len(text.split()) < 5:
        st.warning("Please enter a longer text snippet.")
    else:
        try:
            result = run_api(text) if mode == "API" else run_local(text)
        except Exception as exc:  # pragma: no cover - user feedback only
            st.error(f"Inference failed: {exc}")
        else:
            label = result["label"].title()
            confidence = result["confidence"]
            alert = confidence >= threshold and label.lower() != "real"
            st.metric("Predicted Label", f"{'⚠️' if alert else '✅'} {label}")
            st.metric("Confidence", f"{confidence:.2%}")

            if tokens := result.get("top_tokens"):
                st.subheader("Top Tokens")
                st.table({"token": list(tokens.keys()), "weight": list(tokens.values())})

            st.subheader("Guidance")
            st.write("Use this tool as a triage step. Always verify with trusted fact-checking sources.")
