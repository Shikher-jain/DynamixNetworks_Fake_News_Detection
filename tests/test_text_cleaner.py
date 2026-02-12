from src.fnd.features.text_cleaner import normalize


def test_normalize_removes_stopwords():
    text = "This is a SAMPLE sentence with Noise!"
    normalized = normalize(text)
    assert "sample" in normalized
    assert "this" not in normalized
