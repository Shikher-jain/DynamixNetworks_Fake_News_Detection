from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import HealthResponse, PredictionRequest, PredictionResponse
from .services import pipeline

app = FastAPI(title="News Detector API", version="0.1.0")

allowed_origins = os.getenv("FND_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _top_tokens(text: str, k: int = 5) -> dict[str, float]:
    model = pipeline()
    vectorizer = model.vectorizer
    classifier = model.classifier
    vector = vectorizer.transform([text])
    
    # Handle CalibratedClassifierCV by accessing base_estimator
    if hasattr(classifier, 'calibrated_classifiers_'):
        # For calibrated classifiers, get the base estimator
        base_estimator = classifier.calibrated_classifiers_[0].estimator
        if hasattr(base_estimator, 'coef_'):
            weights = base_estimator.coef_[0]
        else:
            return {}
    elif hasattr(classifier, 'coef_'):
        weights = classifier.coef_[0]
    else:
        return {}
    
    indices = vector.nonzero()[1]
    contributions = {vectorizer.get_feature_names_out()[i]: float(weights[i]) for i in indices}
    return dict(sorted(contributions.items(), key=lambda pair: abs(pair[1]), reverse=True)[:k])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/")
@app.post("/")
def root():
    return {"message": "Welcome to News Detector API. Use POST /predict with {'text': 'your news'} to analyze."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model = pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        probs = model.predict_proba([request.text])[0]
        label = model.predict([request.text])[0]
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    return PredictionResponse(
        label=label,
        confidence=float(max(probs)),
        top_tokens=_top_tokens(request.text),
    )
