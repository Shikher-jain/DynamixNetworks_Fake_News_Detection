from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from .config import load_config
from .data.loader import DatasetConfig, load_dataset
from .models.pipeline import FakeNewsPipeline, PipelineArtifacts, build_baseline

app = typer.Typer(help="Training entrypoints for news detector")

def _prepare_texts(df: pd.DataFrame, cfg: DatasetConfig) -> pd.Series:
    texts = df[cfg.text_column].fillna("")
    if cfg.title_column and cfg.title_column in df.columns:
        texts = df[cfg.title_column].fillna("") + " " + texts
    return texts.str.strip()


def _map_labels(labels: pd.Series) -> pd.Series:
    """
    Map labels to binary classification.
    Since this dataset contains only various types of misinformation,
    we create two classes:
    - 'unreliable': bs, satire (less harmful, clickbait, parody)
    - 'fake': bias, conspiracy, hate, junksci, fake, state (more harmful misinformation)
    """
    normalized = labels.str.lower().str.strip()
    
    unreliable_types = {'bs', 'satire'}
    fake_types = {'bias', 'conspiracy', 'hate', 'junksci', 'fake', 'state'}
    
    def map_label(x):
        if x in unreliable_types:
            return 'unreliable'
        elif x in fake_types:
            return 'fake'
        else:
            # Default to fake for unknown types
            return 'fake'
    
    mapped = normalized.apply(map_label)
    typer.echo(f"Label distribution: {mapped.value_counts().to_dict()}")
    return mapped


@app.command()
def main(config: Path = typer.Option(..., exists=True, help="Path to YAML config")) -> None:
    cfg = load_config(config)
    dataset_cfg = DatasetConfig(
        csv_path=Path(cfg["data"]["input_csv"]),
        text_column=cfg["data"].get("text_column", "text"),
        title_column=cfg["data"].get("title_column"),
        label_column=cfg["data"].get("label_column", "label"),
        language_column=cfg["data"].get("language_column"),
        language_filter=cfg["data"].get("language_filter"),
        sample_frac=cfg["data"].get("sample_frac", 1.0),
    )

    df = load_dataset(dataset_cfg)
    texts = _prepare_texts(df, dataset_cfg)
    labels = _map_labels(df[dataset_cfg.label_column])

    pipeline: FakeNewsPipeline = build_baseline(cfg)
    metrics = pipeline.fit(texts, labels, calibration=cfg["training"].get("calibration", True))

    artifacts = PipelineArtifacts(
        model_path=Path(cfg["artifacts"]["model_path"]),
        vectorizer_path=Path(cfg["artifacts"]["vectorizer_path"]),
        label_path=Path(cfg["artifacts"]["label_path"]),
        metrics_path=Path(cfg.get("metrics_path", "models/metrics.json")),
    )
    pipeline.save(artifacts)
    artifacts.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"Saved model to {artifacts.model_path}")


if __name__ == "__main__":
    app()
