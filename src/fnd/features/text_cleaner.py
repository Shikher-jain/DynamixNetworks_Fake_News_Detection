from __future__ import annotations

import re
from typing import Iterable

import nltk
from nltk.corpus import stopwords

TOKEN_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


def _load_stopwords() -> set[str]:  # pragma: no cover - download is side effect
	try:
		return set(stopwords.words("english"))
	except LookupError:
		nltk.download("stopwords")
		return set(stopwords.words("english"))


STOP_WORDS = _load_stopwords()


def normalize(text: str, lowercase: bool = True, remove_stopwords: bool = True) -> str:
	if lowercase:
		text = text.lower()
	tokens = TOKEN_PATTERN.split(text)
	if remove_stopwords:
		tokens = [token for token in tokens if token and token not in STOP_WORDS]
	return " ".join(tokens)


def batch_normalize(texts: Iterable[str], **kwargs) -> list[str]:
	return [normalize(t, **kwargs) for t in texts]
