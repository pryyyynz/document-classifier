"""Dataset utilities: stratified splits for data preparation."""

from __future__ import annotations

from typing import Sequence, Tuple

try:
    from sklearn.model_selection import train_test_split  # type: ignore
except Exception:
    train_test_split = None  # type: ignore


def stratified_split(
    X: Sequence, y: Sequence, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42
) -> Tuple[Tuple, Tuple, Tuple]:
    """Split into train/val/test with stratification.

    Returns: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")

    if train_test_split is None:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_size), stratify=y, random_state=random_state
    )

    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, stratify=y_temp, random_state=random_state
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


__all__ = [
    "stratified_split",
]
