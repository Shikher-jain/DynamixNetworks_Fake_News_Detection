from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ..features.text_cleaner import batch_normalize


@dataclass
class PipelineArtifacts:
    model_path: Path
    vectorizer_path: Path
    label_path: Path
    metrics_path: Path


class FakeNewsPipeline:
    def __init__(self, vectorizer: TfidfVectorizer, classifier: Any):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.labels_: list[str] | None = None

    def fit(self, texts: Iterable[str], labels: Iterable[str], calibration: bool = True) -> Dict[str, Any]:
        texts = batch_normalize(texts)
        x = self.vectorizer.fit_transform(texts)
        y = np.array(list(labels))
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
        if calibration:
            self.classifier = CalibratedClassifierCV(self.classifier, method="sigmoid", cv=5)
        self.classifier.fit(train_x, train_y)
        preds = self.classifier.predict(test_x)
        probs = self.classifier.predict_proba(test_x)
        metrics = classification_report(test_y, preds, output_dict=True)
        metrics["confidences"] = {
            "mean_max": float(np.mean(np.max(probs, axis=1))),
            "std_max": float(np.std(np.max(probs, axis=1))),
        }
        self.labels_ = sorted(set(y.tolist()))
        return metrics

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.vectorizer.transform(batch_normalize(texts))
        return self.classifier.predict(vectors)

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.vectorizer.transform(batch_normalize(texts))
        return self.classifier.predict_proba(vectors)

    def save(self, artifacts: PipelineArtifacts) -> None:
        artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, artifacts.model_path)
        joblib.dump(self.vectorizer, artifacts.vectorizer_path)
        artifacts.label_path.write_text("\n".join(self.labels_ or []), encoding="utf-8")

    @classmethod
    def load(cls, artifacts: PipelineArtifacts) -> "FakeNewsPipeline":
        classifier = joblib.load(artifacts.model_path)
        vectorizer = joblib.load(artifacts.vectorizer_path)
        pipeline = cls(vectorizer=vectorizer, classifier=classifier)
        if artifacts.label_path.exists():
            pipeline.labels_ = artifacts.label_path.read_text(encoding="utf-8").splitlines()
        return pipeline


def build_baseline(cfg: Dict[str, Any]) -> FakeNewsPipeline:
    vectorizer_params = cfg["features"]["vectorizer"]["params"].copy()
    # Convert ngram_range from list to tuple if present
    if "ngram_range" in vectorizer_params and isinstance(vectorizer_params["ngram_range"], list):
        vectorizer_params["ngram_range"] = tuple(vectorizer_params["ngram_range"])
    vectorizer = TfidfVectorizer(**vectorizer_params)
    model_params = cfg["model"]["params"] | {"max_iter": cfg["training"]["max_iter"]}
    classifier = LogisticRegression(**model_params)
    return FakeNewsPipeline(vectorizer=vectorizer, classifier=classifier)
