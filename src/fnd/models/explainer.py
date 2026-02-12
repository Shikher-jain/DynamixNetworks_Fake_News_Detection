from __future__ import annotations

from typing import Iterable, List

import numpy as np
import shap

from ..features.text_cleaner import batch_normalize
from .pipeline import FakeNewsPipeline


class ExplanationService:
    def __init__(self, pipeline: FakeNewsPipeline, background_size: int = 200):
        self.pipeline = pipeline
        self.background_size = background_size
        self._explainer: shap.Explainer | None = None

    def _ensure_explainer(self, background_texts: Iterable[str]) -> shap.Explainer:
        if self._explainer is None:
            normalized = batch_normalize(background_texts)
            vectors = self.pipeline.vectorizer.transform(normalized)
            dense = vectors[: self.background_size].toarray()
            self._explainer = shap.KernelExplainer(self.pipeline.classifier.predict_proba, dense)
        return self._explainer

    def explain(self, samples: List[str], background_texts: Iterable[str]) -> list[dict[str, float]]:
        explainer = self._ensure_explainer(background_texts)
        normalized = batch_normalize(samples)
        vectors = self.pipeline.vectorizer.transform(normalized).toarray()
        shap_values = explainer.shap_values(vectors)
        explanations = []
        for value_matrix in shap_values:
            importance = np.abs(value_matrix).mean(axis=0)
            feature_names = self.pipeline.vectorizer.get_feature_names_out()
            pairs = sorted(zip(feature_names, importance), key=lambda item: item[1], reverse=True)[:10]
            explanations.append({token: float(score) for token, score in pairs})
        return explanations
