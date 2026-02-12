import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.fnd.models.pipeline import FakeNewsPipeline


def test_pipeline_train_predict_cycle():
    texts = ["real news article", "fake conspiracy", "authentic report", "fabricated story"]
    labels = ["real", "fake", "real", "fake"]
    pipeline = FakeNewsPipeline(TfidfVectorizer(), LogisticRegression(max_iter=50))
    pipeline.fit(texts, labels, calibration=False)
    preds = pipeline.predict(["real report"])
    probs = pipeline.predict_proba(["real report"])
    assert preds[0] in {"real", "fake"}
    assert np.isclose(probs.sum(), 1.0)
