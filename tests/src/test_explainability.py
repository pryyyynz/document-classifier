from unittest import mock
import pytest


def test_contract_explainer_init_requires_lime(monkeypatch):
    import importlib
    m = importlib.import_module('src.explainability')
    # Force init path to see LIME as unavailable
    monkeypatch.setattr(m, 'LIME_AVAILABLE', False, raising=False)
    # Ensure ContractExplainer raises based on that flag
    from tests.conftest import DummyClassifier, DummyVectorizer
    # When LIME_AVAILABLE is False, __init__ should raise ImportError
    with pytest.raises(ImportError):
        m.ContractExplainer(DummyClassifier(['a']), DummyVectorizer(), ['a'])


def test_explain_prediction_happy_path(monkeypatch):
    import importlib
    m = importlib.import_module('src.explainability')
    monkeypatch.setattr(m, 'LIME_AVAILABLE', True, raising=False)

    class DummyExp:
        def as_list(self, label):
            return [('long phrase with words', 0.8), ('token', -0.2)]

        def as_html(self):
            return '<div></div>'

    class DummyLime:
        def __init__(self, *a, **k):
            pass

        def explain_instance(self, text, predict_proba, num_features, num_samples, top_labels):
            return DummyExp()

    # If LimeTextExplainer is not present as attribute under import hook, set with raising=False
    monkeypatch.setattr(m, 'LimeTextExplainer', DummyLime, raising=False)
    from tests.conftest import DummyClassifier, DummyVectorizer

    classes = ['employment_contract', 'nda',
               'partnership', 'service', 'vendor']
    explainer = m.ContractExplainer(
        DummyClassifier(classes), DummyVectorizer(), classes)
    out = explainer.explain_prediction(
        'some test text that is definitely long enough', num_features=3)
    assert out['success'] is True
    assert out['prediction'] in classes
    assert isinstance(out['confidence'], float)
    assert len(out['important_features']) >= 1
    assert isinstance(out['explanation_html'], str)


def test_get_best_phrase_feature_selects_phrase(monkeypatch):
    import importlib
    m = importlib.import_module('src.explainability')
    monkeypatch.setattr(m, 'LIME_AVAILABLE', True, raising=False)
    monkeypatch.setattr(m, 'LimeTextExplainer',
                        mock.MagicMock(), raising=False)
    from tests.conftest import DummyClassifier, DummyVectorizer
    classes = ['a', 'b']
    e = m.ContractExplainer(DummyClassifier(
        classes), DummyVectorizer(), classes)
    feats = [('token', 0.5), ('another', 0.4)]
    text = 'context around token that makes a longer phrase candidate here'
    best = m.ContractExplainer._get_best_phrase_feature(e, feats, text)
    assert len(best) == 1
    assert len(best[0][0].split()) >= 3
