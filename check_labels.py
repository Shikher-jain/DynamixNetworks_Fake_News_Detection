import pandas as pd
from pathlib import Path

csv_path = Path('data/raw/fake_news.csv')
if not csv_path.exists():
    print(f"File not found: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path, usecols=['type'])
print("Label distribution:")
print(df['type'].value_counts())
print("\nUnique labels (sorted):")
print(sorted(df['type'].dropna().unique()))
