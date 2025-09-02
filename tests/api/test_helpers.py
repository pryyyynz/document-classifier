from unittest import mock
import io
import sys


def test_preprocess_text_happy_path():
    import api.main as mod
    text = '  Hello   world   this is   spaced  '
    out = mod.preprocess_text(text)
    assert out == 'Hello world this is spaced'


def test_preprocess_text_errors():
    import api.main as mod
    import pytest
    with pytest.raises(Exception):
        mod.preprocess_text('')
    with pytest.raises(Exception):
        mod.preprocess_text('short')


def test_extract_text_from_pdf_direct(monkeypatch):
    import api.main as mod

    class DummyPage:
        def extract_text(self):
            return 'page text'

    class DummyPDF:
        pages = [DummyPage(), DummyPage()]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr('pdfplumber.open', lambda path: DummyPDF())
    text = mod.extract_text_from_pdf('/tmp/fake.pdf')
    assert 'page text' in text


def test_extract_text_from_pdf_ocr_fallback(monkeypatch):
    import api.main as mod

    class DummyPage:
        def extract_text(self):
            return None

    class DummyPDF:
        pages = [DummyPage()]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr('pdfplumber.open', lambda path: DummyPDF())
    monkeypatch.setattr(mod, 'extract_text_with_ocr',
                        lambda p: 'ocr text found')
    text = mod.extract_text_from_pdf('/tmp/scan.pdf')
    assert text == 'ocr text found'


def test_extract_text_with_ocr_mocked(monkeypatch):
    import api.main as mod
    # Just ensure wrapper raises or returns; here we mock success path via fitz and pytesseract

    class DummyDoc:
        def __len__(self):
            return 1

        def load_page(self, i):
            class P:
                def get_pixmap(self, matrix=None):
                    class Pix:
                        def tobytes(self, fmt):
                            return b'PNGbytes'
                    return Pix()
            return P()

        def close(self):
            pass

    mock_fitz = mock.MagicMock()
    mock_fitz.open = lambda p: DummyDoc()
    # Ensure Matrix is available for page.get_pixmap(matrix=fitz.Matrix(...))

    class _Matrix:
        def __init__(self, *a, **k):
            pass
    mock_fitz.Matrix = _Matrix
    monkeypatch.setitem(sys.modules, 'fitz', mock_fitz)
    monkeypatch.setitem(sys.modules, 'pytesseract', mock.MagicMock(
        image_to_string=lambda img, lang=None: 'ocr text'))

    # Patch PIL Image.open to return a valid-like PIL Image object
    import PIL.Image as PILImage

    class DummyImage(PILImage.Image):
        pass
    monkeypatch.setattr(
        PILImage, 'open', lambda b: PILImage.new('RGB', (10, 10)))

    text = mod.extract_text_with_ocr('/tmp/scan.pdf')
    assert 'ocr text' in text


def test_extract_text_from_docx(monkeypatch):
    import api.main as mod

    class Para:
        def __init__(self, t):
            self.text = t

    class DummyDoc:
        paragraphs = [Para('hello'), Para('world')]
    monkeypatch.setitem(sys.modules, 'docx', mock.MagicMock(
        Document=lambda p: DummyDoc()))
    # Call underlying function directly which imports Document at module import; patch constructor instead
    monkeypatch.setattr('api.main.Document', lambda p: DummyDoc())
    text = mod.extract_text_from_docx('/tmp/f.docx')
    assert text == 'hello\nworld'


def test_extract_text_from_doc_success(monkeypatch):
    import api.main as mod
    from types import SimpleNamespace as NS
    monkeypatch.setitem(sys.modules, 'subprocess', mock.MagicMock(
        run=lambda args, capture_output, text: NS(returncode=0, stdout='doc content')))
    out = mod.extract_text_from_doc('/tmp/f.doc')
    assert out == 'doc content'


def test_extract_text_from_doc_failure(monkeypatch):
    import api.main as mod
    from types import SimpleNamespace as NS
    monkeypatch.setitem(sys.modules, 'subprocess', mock.MagicMock(
        run=lambda args, capture_output, text: NS(returncode=1, stdout='')))
    import pytest
    with pytest.raises(Exception):
        mod.extract_text_from_doc('/tmp/f.doc')
