import sys
from types import ModuleType
import builtins
import io
import pickle
import pytest


class DummyClassifier:
    def __init__(self, classes):
        self.classes_ = classes

    def predict_proba(self, X):
        import numpy as np
        # Return a deterministic distribution: all weight to last class
        probs = np.zeros((len(getattr(X, 'rows', [0])), len(self.classes_))) if hasattr(
            X, 'rows') else np.zeros((1, len(self.classes_)))
        probs[...] = 0.0
        probs[..., -1] = 1.0
        return probs


class DummyVectorizer:
    def transform(self, texts):
        class _X:
            rows = [0] * len(list(texts))
        return _X()


class DummyExplainer:
    def __init__(self, *args, **kwargs):
        pass

    def explain_prediction(self, text, num_features=1):
        return {
            'text': text[:200],
            'full_text': text,
            'prediction': 'vendor',
            'confidence': 1.0,
            'important_features': [('dummy phrase with three', 0.9)],
            'explanation_html': '<div></div>',
            'num_features': num_features,
            'success': True,
        }


@pytest.fixture(autouse=True)
def ensure_src_explainability_stub(monkeypatch):
    """Provide a lightweight stub for src.explainability.ContractExplainer when needed."""
    module_name = 'src.explainability'
    if module_name not in sys.modules:
        m = ModuleType('src.explainability')
        setattr(m, 'ContractExplainer', DummyExplainer)
        sys.modules[module_name] = m
    yield


@pytest.fixture
def fake_model_pickle_bytes():
    classes = ['employment_contract', 'nda',
               'partnership', 'service', 'vendor']
    model_dict = {
        'classifier': DummyClassifier(classes),
        'vectorizer': DummyVectorizer(),
        'class_names': classes,
        'feature_selector': None,
    }
    return pickle.dumps(model_dict)


@pytest.fixture
def test_client(monkeypatch):
    from fastapi.testclient import TestClient
    import importlib
    api_module = importlib.import_module('api.main')

    # Bypass heavy load by setting globals directly
    api_module.MODEL = DummyClassifier(
        ['employment_contract', 'nda', 'partnership', 'service', 'vendor'])
    api_module.VECTORIZER = DummyVectorizer()
    api_module.EXPLAINER = DummyExplainer()
    api_module.CLASS_NAMES = ['employment_contract',
                              'nda', 'partnership', 'service', 'vendor']
    api_module.MODEL_LOADED = True

    client = TestClient(api_module.app)
    return client
