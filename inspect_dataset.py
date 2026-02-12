import pandas as pd
import sys

# Load just the type column
try:
    df = pd.read_csv('data/raw/fake_news.csv')
    print(f"Total rows: {len(df)}")
    print(f"\nColumns available: {list(df.columns)}")
    print(f"\nType column unique values:")
    print(df['type'].value_counts())
    print(f"\nSample rows:")
    print(df[['title', 'type']].head(20))
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
