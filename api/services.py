from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from src.fnd.models.pipeline import FakeNewsPipeline, PipelineArtifacts

load_dotenv()


@lru_cache(maxsize=1)
def pipeline() -> FakeNewsPipeline:
    artifacts = PipelineArtifacts(
        model_path=Path(os.getenv("FND_MODEL_PATH", "models/baseline.joblib")),
        vectorizer_path=Path(os.getenv("FND_VECTOR_PATH", "models/vectorizer.joblib")),
        label_path=Path(os.getenv("FND_LABELS_PATH", "models/labels.json")),
        metrics_path=Path("models/metrics.json"),
    )
    return FakeNewsPipeline.load(artifacts)
