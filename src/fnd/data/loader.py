from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DatasetConfig:
    csv_path: Path
    text_column: str = "text"
    title_column: Optional[str] = "title"
    label_column: str = "label"
    language_column: Optional[str] = None
    language_filter: Optional[str] = None
    sample_frac: float = 1.0


def load_dataset(cfg: DatasetConfig) -> pd.DataFrame:
    if not cfg.csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {cfg.csv_path}. Download the Kaggle fake news dataset and place it under data/raw."
        )
    df = pd.read_csv(cfg.csv_path)
    if cfg.language_column and cfg.language_filter:
        df = df[df[cfg.language_column].str.lower() == cfg.language_filter.lower()]
    df = df.dropna(subset=[cfg.text_column, cfg.label_column])
    if cfg.title_column:
        df = df.dropna(subset=[cfg.title_column])
    if not 0 < cfg.sample_frac <= 1:
        raise ValueError("sample_frac must be between 0 and 1")
    if cfg.sample_frac < 1:
        df = df.sample(frac=cfg.sample_frac, random_state=42)
    return df.reset_index(drop=True)
