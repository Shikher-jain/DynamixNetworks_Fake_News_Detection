from __future__ import annotations

import json
from pathlib import Path
from typing import List

import typer

from .config import load_config
from .models.pipeline import FakeNewsPipeline, PipelineArtifacts

app = typer.Typer(help="CLI inference for news detector")


def load_pipeline(cfg_path: Path) -> FakeNewsPipeline:
    cfg = load_config(cfg_path)
    artifacts = PipelineArtifacts(
        model_path=Path(cfg["artifacts"]["model_path"]),
        vectorizer_path=Path(cfg["artifacts"]["vectorizer_path"]),
        label_path=Path(cfg["artifacts"]["label_path"]),
        metrics_path=Path(cfg.get("metrics_path", "models/metrics.json")),
    )
    return FakeNewsPipeline.load(artifacts)


@app.command()
def text(
    cfg: Path = typer.Option(..., exists=True, help="Path to training config"),
    samples: List[str] = typer.Argument(..., help="Texts to classify"),
) -> None:
    pipeline = load_pipeline(cfg)
    preds = pipeline.predict(samples)
    probs = pipeline.predict_proba(samples)
    results = []
    for sample, label, prob in zip(samples, preds, probs):
        results.append({"text": sample, "label": label, "confidence": float(prob.max())})
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
