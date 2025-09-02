"""
FastAPI application for contract classification with explainability.

Features:
- Single document prediction with LIME explanations
- Batch processing for multiple documents
- Support for text, PDF, DOC, DOCX files
- OCR processing for scanned documents
- Model loading once at startup
- Comprehensive error handling
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import io
import csv

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Document processing
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Machine learning and explainability
import pickle
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure project root is on sys.path for absolute imports like `src.*`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Global model variables (loaded once at startup)
MODEL = None
VECTORIZER = None
CLASS_NAMES = None
EXPLAINER = None
MODEL_LOADED = False

# Global storage for batch results (in production, use a database or cache)
BATCH_RESULTS = {}

# Pydantic models for API requests/responses


class TextRequest(BaseModel):
    text: str = Field(...,
                      description="Contract text to classify", min_length=10)
    num_features: int = Field(
        default=1, description="Number of LIME features to return", ge=1, le=50)


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    important_features: List[Dict[str, Any]]
    processing_time: float
    text_preview: str
    success: bool
    error_message: Optional[str] = None


class BatchResponse(BaseModel):
    total_documents: int
    successful_predictions: int
    failed_predictions: int
    results: List[Dict[str, Any]]
    processing_time: float
    csv_download_url: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="Contract Classification API",
    description="AI-powered contract classification with LIME explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    """Load the trained model and explainer once at startup."""
    global MODEL, VECTORIZER, CLASS_NAMES, EXPLAINER, MODEL_LOADED

    try:
        logger.info("🔄 Loading contract classification model...")

        # Ensure project root is available for imports
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        # Path to the best model (resolve relative to project root)
        model_path = str(
            PROJECT_ROOT / "enhanced_models_output/models/enhanced_tfidf_gradient_boosting_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load model data
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        MODEL = model_data['classifier']
        VECTORIZER = model_data['vectorizer']
        # Prefer the classifier's own class order to avoid any index shift issues
        if hasattr(MODEL, 'classes_'):
            CLASS_NAMES = list(MODEL.classes_)
        else:
            CLASS_NAMES = model_data['class_names']
        FEATURE_SELECTOR = model_data.get('feature_selector', None)

        # Import and create explainer
        from src.explainability import ContractExplainer
        EXPLAINER = ContractExplainer(
            MODEL, VECTORIZER, CLASS_NAMES, FEATURE_SELECTOR)

        MODEL_LOADED = True
        logger.info(f"✅ Model loaded successfully! Classes: {CLASS_NAMES}")
        logger.info(f"📊 Model type: {type(MODEL).__name__}")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        MODEL_LOADED = False
        raise


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file, with OCR fallback for scanned documents."""
    try:
        # Try text extraction first
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        # If no text extracted, try OCR
        if not text.strip():
            logger.info("No text found in PDF, attempting OCR...")
            text = extract_text_with_ocr(file_path)

        return text.strip()

    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")


def extract_text_with_ocr(file_path: str) -> str:
    """Extract text from scanned PDF using OCR."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Convert page to image
            # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")

            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))

            # OCR processing
            page_text = pytesseract.image_to_string(img, lang='eng')
            text += page_text + "\n"

        doc.close()
        return text.strip()

    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise HTTPException(
            status_code=400, detail=f"OCR processing failed: {str(e)}")


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()

    except Exception as e:
        logger.error(f"DOCX text extraction failed: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to extract text from DOCX: {str(e)}")


def extract_text_from_doc(file_path: str) -> str:
    """Extract text from DOC file (requires antiword or similar)."""
    try:
        # Try using antiword if available
        import subprocess
        result = subprocess.run(['antiword', file_path],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            raise Exception("antiword not available")

    except Exception as e:
        logger.error(f"DOC text extraction failed: {e}")
        raise HTTPException(
            status_code=400, detail="DOC file processing not supported. Please convert to DOCX or PDF.")


def preprocess_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        raise HTTPException(
            status_code=400, detail="No text content found in document")

    # Basic cleaning
    text = text.strip()
    text = ' '.join(text.split())  # Remove extra whitespace

    if len(text) < 10:
        raise HTTPException(
            status_code=400, detail="Text too short for classification")

    return text


def classify_contract(text: str, num_features: int = 15) -> Dict[str, Any]:
    """Classify contract text and generate LIME explanation."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get LIME explanation
        explanation = EXPLAINER.explain_prediction(
            text, num_features=num_features)

        if not explanation['success']:
            raise Exception(explanation.get('error', 'Explanation failed'))

        return explanation

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Classification failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model when application starts."""
    load_model()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Contract Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_info = {}
    if MODEL_LOADED:
        model_info = {
            "model_type": type(MODEL).__name__,
            "classes": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
            "vectorizer_type": type(VECTORIZER).__name__
        }

    return HealthResponse(
        status="healthy" if MODEL_LOADED else "unhealthy",
        model_loaded=MODEL_LOADED,
        model_info=model_info,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text(request: TextRequest):
    """Classify contract from raw text."""
    start_time = datetime.now()

    try:
        # Preprocess text
        text = preprocess_text(request.text)

        # Classify
        explanation = classify_contract(text, request.num_features)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        return PredictionResponse(
            prediction=explanation['prediction'],
            confidence=explanation['confidence'],
            important_features=[
                {"feature": feat, "score": score}
                for feat, score in explanation['important_features']
            ],
            processing_time=processing_time,
            text_preview=text[:200] + "..." if len(text) > 200 else text,
            success=True
        )

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        return PredictionResponse(
            prediction="",
            confidence=0.0,
            important_features=[],
            processing_time=processing_time,
            text_preview="",
            success=False,
            error_message=str(e)
        )


@app.post("/predict/file", response_model=PredictionResponse)
async def predict_file(
    file: UploadFile = File(...),
    num_features: int = 1
):
    """Classify contract from uploaded file (PDF, DOC, DOCX)."""
    start_time = datetime.now()

    try:
        # Validate file type strictly by MIME to avoid ambiguous handling
        allowed_types = {
            'application/pdf': '.pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/msword': '.doc',
            'text/plain': '.txt'
        }
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Supported: {list(allowed_types.keys())}"
            )

        # Create temporary file
        # Choose suffix based on MIME (validated above)
        chosen_suffix = allowed_types[file.content_type]
        with tempfile.NamedTemporaryFile(delete=False, suffix=chosen_suffix) as temp_file:
            # Write uploaded file to temp file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        try:
            # Extract text based on file type
            if file.content_type == 'application/pdf':
                text = extract_text_from_pdf(temp_file_path)
            elif file.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                text = extract_text_from_docx(temp_file_path)
            elif file.content_type == 'application/msword':
                text = extract_text_from_doc(temp_file_path)
            else:  # text/plain
                with open(temp_file_path, 'r', encoding='utf-8') as fh:
                    text = fh.read()

            # Preprocess text
            text = preprocess_text(text)

            # Classify
            explanation = classify_contract(text, num_features)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            return PredictionResponse(
                prediction=explanation['prediction'],
                confidence=explanation['confidence'],
                important_features=[
                    {"feature": feat, "score": score}
                    for feat, score in explanation['important_features']
                ],
                processing_time=processing_time,
                text_preview=text[:200] + "..." if len(text) > 200 else text,
                success=True
            )

        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        return PredictionResponse(
            prediction="",
            confidence=0.0,
            important_features=[],
            processing_time=processing_time,
            text_preview="",
            success=False,
            error_message=str(e)
        )


@app.post("/batch", response_model=BatchResponse)
async def batch_predict(
    files: List[UploadFile] = File(...),
    num_features: int = 1
):
    """Process multiple documents in batch."""
    start_time = datetime.now()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 100:  # Limit batch size
        raise HTTPException(
            status_code=400, detail="Batch size limited to 100 files")

    results = []
    successful = 0
    failed = 0

    for i, file in enumerate(files):
        try:
            logger.info(f"Processing file {i+1}/{len(files)}: {file.filename}")

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name

            try:
                # Extract text based on file type
                if file.content_type == 'application/pdf':
                    text = extract_text_from_pdf(temp_file_path)
                elif file.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    text = extract_text_from_docx(temp_file_path)
                elif file.content_type == 'application/msword':
                    text = extract_text_from_doc(temp_file_path)
                else:
                    text = file.file.read().decode('utf-8')

                # Preprocess text
                text = preprocess_text(text)

                # Classify
                explanation = classify_contract(text, num_features)

                if explanation['success']:
                    successful += 1
                    results.append({
                        "filename": file.filename,
                        "prediction": explanation['prediction'],
                        "confidence": explanation['confidence'],
                        "text_preview": text[:100] + "..." if len(text) > 100 else text,
                        "top_features": explanation['important_features'][:5],
                        "status": "success"
                    })
                else:
                    failed += 1
                    results.append({
                        "filename": file.filename,
                        "prediction": "",
                        "confidence": 0.0,
                        "text_preview": "",
                        "top_features": [],
                        "status": "failed",
                        "error": explanation.get('error', 'Unknown error')
                    })

            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

        except Exception as e:
            failed += 1
            results.append({
                "filename": file.filename,
                "prediction": "",
                "confidence": 0.0,
                "text_preview": "",
                "top_features": [],
                "status": "failed",
                "error": str(e)
            })

    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds()

    # Store results for CSV download
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    BATCH_RESULTS[timestamp] = results

    return BatchResponse(
        total_documents=len(files),
        successful_predictions=successful,
        failed_predictions=failed,
        results=results,
        processing_time=processing_time,
        csv_download_url=f"/batch/download/{timestamp}"
    )


@app.get("/batch/download/{timestamp}")
async def download_batch_results(timestamp: str):
    """Download batch results as CSV."""
    if timestamp not in BATCH_RESULTS:
        raise HTTPException(status_code=404, detail="Batch results not found")

    results = BATCH_RESULTS[timestamp]

    # Create CSV data with all relevant information
    csv_data = [
        ["filename", "prediction", "confidence", "status", "error", "key_phrase"]
    ]

    for result in results:
        # Extract single key phrase
        features = result.get("top_features", [])
        key_phrase = features[0][0] if len(features) > 0 else ""

        csv_data.append([
            result["filename"],
            result["prediction"],
            f"{result['confidence']:.3f}",
            result["status"],
            result.get("error", ""),
            key_phrase
        ])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(csv_data)

    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=batch_results_{timestamp}.csv"}
    )


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": type(MODEL).__name__,
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "vectorizer_type": type(VECTORIZER).__name__,
        "model_loaded_at": "startup",
        "explainer_available": EXPLAINER is not None
    }

if __name__ == "__main__":
    # Load model before starting server
    load_model()

    # Start server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to prevent model reloading
        log_level="info"
    )
