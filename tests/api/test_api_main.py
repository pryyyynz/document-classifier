import io
import os
import sys
import importlib
from unittest import mock


def test_load_model_adds_project_root_to_sys_path(tmp_path, monkeypatch, fake_model_pickle_bytes):
    mod = importlib.import_module('api.main')
    model_path = tmp_path / 'enhanced_models_output' / 'models'
    model_path.mkdir(parents=True)
    pkl_file = model_path / 'enhanced_tfidf_gradient_boosting_model.pkl'
    pkl_file.write_bytes(fake_model_pickle_bytes)

    monkeypatch.setattr(mod, 'PROJECT_ROOT', tmp_path)
    # Remove path if present
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))

    # Force reload function logic
    # Provide real explainability import via our conftest stub
    mod.MODEL = None
    mod.VECTORIZER = None
    mod.CLASS_NAMES = None
    mod.EXPLAINER = None
    mod.MODEL_LOADED = False

    mod.load_model()

    assert str(tmp_path) in sys.path
    assert mod.MODEL is not None
    assert mod.VECTORIZER is not None
    assert mod.CLASS_NAMES == ['employment_contract',
                               'nda', 'partnership', 'service', 'vendor']
    assert mod.MODEL_LOADED is True


def test_load_model_missing_file_raises(tmp_path, monkeypatch):
    mod = importlib.import_module('api.main')
    monkeypatch.setattr(mod, 'PROJECT_ROOT', tmp_path)
    with mock.patch.dict('sys.modules', {'src.explainability': mock.MagicMock()}):
        with mock.patch.object(mod, 'logger'):
            try:
                mod.load_model()
            except FileNotFoundError:
                pass
            else:
                assert False, 'Expected FileNotFoundError'


def test_health_endpoint_loaded(test_client):
    r = test_client.get('/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'healthy'
    assert body['model_loaded'] is True
    assert body['model_info']['num_classes'] == 5
    assert isinstance(body['model_info']['classes'], list)


def test_health_endpoint_unloaded(monkeypatch):
    import api.main as mod
    from fastapi.testclient import TestClient
    mod.MODEL_LOADED = False
    client = TestClient(mod.app)
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'unhealthy'


def test_predict_file_accepts_text(test_client, tmp_path):
    content = b"This is a long enough text content to classify as vendor."
    file = {'file': ('sample.txt', io.BytesIO(content), 'text/plain')}
    r = test_client.post('/predict/file', files=file,
                         params={'num_features': 1})
    assert r.status_code == 200
    body = r.json()
    assert body['success'] is True
    assert body['prediction']
    assert body['confidence'] >= 0.0


def test_predict_file_rejects_unsupported_mime(test_client):
    file = {'file': ('sample.bin', io.BytesIO(
        b'abc'), 'application/octet-stream')}
    r = test_client.post('/predict/file', files=file)
    assert r.status_code == 400


def test_predict_file_routes_pdf_docx_doc_via_extractors(test_client, monkeypatch):
    import api.main as mod
    monkeypatch.setattr(mod, 'extract_text_from_pdf',
                        lambda p: 'pdf text content more than ten')
    monkeypatch.setattr(mod, 'extract_text_from_docx',
                        lambda p: 'docx text content more than ten')
    monkeypatch.setattr(mod, 'extract_text_from_doc',
                        lambda p: 'doc text content more than ten')

    for name, mime in [('a.pdf', 'application/pdf'), ('b.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'), ('c.doc', 'application/msword')]:
        r = test_client.post(
            '/predict/file', files={'file': (name, io.BytesIO(b'ignored'), mime)})
        assert r.status_code == 200
        assert r.json()['success'] is True


def test_batch_limits_and_aggregates(test_client):
    files = []
    for i in range(3):
        files.append(('files', (f's{i}.txt', io.BytesIO(
            b"This is a sufficiently long text for classification."), 'text/plain')))
    r = test_client.post('/batch', files=files, params={'num_features': 1})
    assert r.status_code == 200
    body = r.json()
    assert body['total_documents'] == 3
    assert body['successful_predictions'] + body['failed_predictions'] == 3
    assert isinstance(body['results'], list)

    # Check limit 100
    too_many = [('files', (f'f{i}.txt', io.BytesIO(
        b"This is a sufficiently long text."), 'text/plain')) for i in range(101)]
    r2 = test_client.post('/batch', files=too_many)
    assert r2.status_code == 400
