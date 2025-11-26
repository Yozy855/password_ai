"""Utilities to train and use a password strength classifier.

The module is intentionally lightweight so it can be imported without
triggering training. Call `train_model()` once to build a model, then
pass the returned `(model, vectorizer)` pair to `predict_strength()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Strength label mapping used across the project
LABELS = {0: "Weak", 1: "Medium", 2: "Strong"}

# Cache the most recently trained model to avoid retraining within a process
_MODEL_CACHE: Optional[Tuple[LogisticRegression, TfidfVectorizer]] = None


def load_dataset(csv_path: str = "data.csv") -> pd.DataFrame:
    """Load and clean the password dataset."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(path, on_bad_lines="skip")
    df = df.dropna(subset=["password", "strength"])
    df = df[df["password"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    df["strength"] = df["strength"].astype(int)
    return df


def train_model(
    csv_path: str = "data.csv",
    sample_n: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
    verbose: bool = False,
) -> Tuple[LogisticRegression, TfidfVectorizer]:
    """
    Train the password strength model.

    Returns a `(model, vectorizer)` tuple. The trained objects are cached for
    reuse in this process; call again to retrain with different options.
    """
    global _MODEL_CACHE

    df = load_dataset(csv_path)
    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=random_state)

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    X = vectorizer.fit_transform(df["password"])
    y = df["strength"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=max_iter, n_jobs=1)
    model.fit(X_train, y_train)

    if verbose:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=LABELS.values())
        print("Validation report:")
        print(report)

    _MODEL_CACHE = (model, vectorizer)
    return model, vectorizer


def _resolve_model_and_vectorizer(
    model: Optional[LogisticRegression], vectorizer: Optional[TfidfVectorizer]
) -> Tuple[LogisticRegression, TfidfVectorizer]:
    if model and vectorizer:
        return model, vectorizer
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    raise RuntimeError("Model not provided and no cached model available. Call train_model() first.")


def predict_strength(
    password: str,
    model: Optional[LogisticRegression] = None,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> str:
    """Predict password strength label."""
    if not isinstance(password, str) or password.strip() == "":
        raise ValueError("Password must be a non-empty string.")

    model, vectorizer = _resolve_model_and_vectorizer(model, vectorizer)
    vec = vectorizer.transform([password])
    pred = int(model.predict(vec)[0])
    return LABELS.get(pred, str(pred))


if __name__ == "__main__":
    # Minimal demo for manual runs
    model, vectorizer = train_model(verbose=True)
    sample_passwords = ["password123", "Summer2025!", "Tg!93xQ#zA"]
    for pw in sample_passwords:
        label = predict_strength(pw, model, vectorizer)
        print(f"{pw:15s} -> {label}")
