import os
from pathlib import Path


def create_dataset(tmp_path: Path, classes_files: dict):
    for cls, files in classes_files.items():
        d = tmp_path / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(files):
            (d / f'd{i}.pdf').write_bytes(b'%PDF-1.4 fake')


def test_class_mappings_detect_existing_classes(tmp_path, monkeypatch):
    from src.data_loader import ContractDataLoader
    create_dataset(tmp_path, {'nda': 2, 'vendor': 0, 'service': 1})
    dl = ContractDataLoader(str(tmp_path))
    names = dl.get_class_names()
    assert 'nda' in names
    assert 'service' in names
    assert 'vendor' not in names  # no PDFs


def test_load_documents_uses_extractor_and_skips_empty(tmp_path, monkeypatch):
    from src.data_loader import ContractDataLoader
    create_dataset(tmp_path, {'nda': 2})
    dl = ContractDataLoader(str(tmp_path), max_docs_per_class=5)
    # Return text for first, empty for second
    monkeypatch.setattr(dl, 'extract_text_from_pdf',
                        lambda p: 'some text extracted' if 'd0' in str(p) else '')
    texts, labels = dl.load_documents()
    assert len(texts) == 1
    assert labels == ['nda']


def test_preprocess_texts_filters_short(monkeypatch):
    from src.data_loader import ContractDataLoader
    dl = ContractDataLoader(str(tmp_path := Path('.')))  # not used
    # Mock normalize_text to identity
    monkeypatch.setitem(__import__('sys').modules,
                        'preprocessing.text_processing', None)
    monkeypatch.setattr('src.data_loader.normalize_text', lambda t: t)
    processed = dl.preprocess_texts(
        ['short words', 'this text has definitely more than ten distinct tokens present repeatedly here now'])
    assert len(processed) == 1


def test_create_tfidf_features_requires_min_docs(monkeypatch):
    from src.data_loader import ContractDataLoader
    dl = ContractDataLoader(str(Path('.')))
    import pytest
    with pytest.raises(ValueError):
        dl.create_tfidf_features([])
    with pytest.raises(ValueError):
        dl.create_tfidf_features(['only one document with enough tokens here'])
