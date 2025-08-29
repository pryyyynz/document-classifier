# Legacy Preprocessing Scripts

This directory contains preprocessing scripts that were created for initial data preparation but are not part of the main machine learning pipeline.

## Scripts

- **download.py**: Downloads SEC documents and organizes them by type
- **generate_synthetic_docs.py**: Generates synthetic contract documents using reportlab
- **copy_synthetic_docs.py**: Copies existing PDF files to create synthetic documents
- **convert_to_pdf_alt.py**: Converts HTML/text files to PDF using reportlab

## Why These Are Legacy

These scripts were created for:
1. Initial dataset preparation
2. Data augmentation
3. Format conversion

However, they are not used by the main ML pipeline because:
- The main pipeline focuses on PDF text extraction and classification
- Synthetic document generation is not needed for the core classification task
- The pipeline uses the existing PDF dataset directly

## Usage

If you need to regenerate the dataset or create synthetic documents, you can run these scripts from this directory. However, for normal ML training and inference, use the main pipeline in the `src/` directory.

## Dependencies

These scripts require additional dependencies not in the main requirements.txt:
- reportlab (for PDF generation)
- Additional data sources for synthetic generation
