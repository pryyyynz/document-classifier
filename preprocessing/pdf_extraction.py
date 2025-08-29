"""Utilities for extracting text from PDFs.

This module provides two main entry points:
- extract_text_digital_pdf: uses pdfplumber for text-based PDFs
- extract_text_scanned_pdf: renders pages to images and runs OCR (pytesseract + OpenCV)

Notes:
- Tesseract OCR binary must be installed on the system for OCR to work.
  Set the TESSERACT_CMD environment variable if it's not in PATH.
"""

from __future__ import annotations

import os
from typing import List
import numpy as np

try:
    import pdfplumber  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency handling
    pdfplumber = None  # type: ignore

try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover - optional dependency handling
    fitz = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency handling
    cv2 = None  # type: ignore

try:
    import pytesseract  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency handling
    pytesseract = None  # type: ignore


def _ensure_dependency(condition: bool, message: str) -> None:
    if not condition:
        raise ImportError(message)


def extract_text_digital_pdf(pdf_path: str) -> str:
    """Extract text from a digital (text-based) PDF using pdfplumber.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        The concatenated text from all pages.
    """
    _ensure_dependency(pdfplumber is not None,
                       "pdfplumber is required: pip install pdfplumber")
    texts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text.strip())
    return "\n\n".join(t for t in texts if t)


def _render_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    """Render PDF pages to images using PyMuPDF.

    Returns a list of OpenCV BGR images.
    """
    _ensure_dependency(
        fitz is not None, "PyMuPDF is required: pip install PyMuPDF")
    _ensure_dependency(
        cv2 is not None, "opencv-python is required: pip install opencv-python")

    images: List[np.ndarray] = []
    with fitz.open(pdf_path) as doc:  # type: ignore[attr-defined]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)  # type: ignore[attr-defined]
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img = cv2.imdecode(
                memoryview(pix.tobytes("png")),
                cv2.IMREAD_COLOR,
            )
            images.append(img)
    return images


def _preprocess_for_ocr(bgr_img: np.ndarray) -> np.ndarray:
    """Apply basic preprocessing to improve OCR results."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    # Slight denoise
    denoised = cv2.medianBlur(thresh, 3)
    return denoised


def extract_text_scanned_pdf(pdf_path: str, ocr_lang: str = "eng") -> str:
    """Extract text from a scanned PDF using OCR with Tesseract and OpenCV.

    Args:
        pdf_path: Absolute path to the PDF file.
        ocr_lang: Language code(s) for tesseract, e.g., "eng" or "eng+deu".

    Returns:
        The concatenated OCR text from all pages.
    """
    _ensure_dependency(pytesseract is not None,
                       "pytesseract is required: pip install pytesseract")
    _ensure_dependency(
        cv2 is not None, "opencv-python is required: pip install opencv-python")

    if os.getenv("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.getenv(
            "TESSERACT_CMD")  # type: ignore[attr-defined]

    images = _render_pdf_to_images(pdf_path)
    texts: List[str] = []
    for bgr in images:
        pre = _preprocess_for_ocr(bgr)
        text = pytesseract.image_to_string(
            pre, lang=ocr_lang)  # type: ignore[attr-defined]
        texts.append(text.strip())
    return "\n\n".join(t for t in texts if t)


__all__ = [
    "extract_text_digital_pdf",
    "extract_text_scanned_pdf",
]
