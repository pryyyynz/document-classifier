import numpy as np
import pandas as pd
from unittest import mock


def test_plot_confusion_matrix_saves_and_shows(tmp_path, monkeypatch):
    from src.models import plot_confusion_matrix
    cm = np.array([[5, 1], [2, 7]])
    classes = ['a', 'b']
    out_path = tmp_path / 'cm.png'

    # Avoid opening a window
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    plot_confusion_matrix(cm, classes, title='CM', save_path=str(out_path))
    assert out_path.exists()


def test_plot_feature_importance_saves_and_shows(tmp_path, monkeypatch):
    from src.models import plot_feature_importance
    df = pd.DataFrame(
        {'feature': ['f1', 'f2', 'f3'], 'importance': [0.3, 0.6, 0.1]})
    out_path = tmp_path / 'fi.png'

    # Avoid opening a window
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    plot_feature_importance(df, title='FI', save_path=str(out_path))
    assert out_path.exists()
