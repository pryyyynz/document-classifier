"""Data loading and preprocessing for contract classification."""

# Standard library imports
import os
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Add project root to path to enable preprocessing imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Dynamic imports to handle auto-formatting issues
try:
    from preprocessing.text_processing import normalize_text
    from preprocessing.pdf_extraction import extract_text_digital_pdf, extract_text_scanned_pdf
except ImportError:
    # If direct import fails, try with updated path
    try:
        from preprocessing.text_processing import normalize_text
        from preprocessing.pdf_extraction import extract_text_digital_pdf, extract_text_scanned_pdf
    except ImportError:
        raise ImportError(
            "Could not import preprocessing modules. "
            "Please ensure the project structure is correct and preprocessing modules exist."
        )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContractDataLoader:
    """Handles loading and preprocessing of contract documents."""

    def __init__(self, dataset_path: str, max_docs_per_class: Optional[int] = None):
        """
        Initialize the data loader.

        Args:
            dataset_path: Path to the dataset directory
            max_docs_per_class: Maximum number of documents to load per class (for testing)
        """
        self.dataset_path = Path(dataset_path)
        self.max_docs_per_class = max_docs_per_class
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = None

        # Print debug info
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Dataset path exists: {self.dataset_path.exists()}")
        if self.dataset_path.exists():
            logger.info(f"Contents: {list(self.dataset_path.iterdir())}")

        # Contract class mappings - only include classes that exist
        self.class_mappings = {}
        for class_name in ['nda', 'vendor', 'service', 'partnership', 'employment_contract']:
            class_path = self.dataset_path / class_name
            logger.info(f"Checking class {class_name} at path: {class_path}")
            logger.info(f"  Path exists: {class_path.exists()}")

            if class_path.exists():
                pdf_files = list(class_path.glob("*.pdf")) + \
                    list(class_path.glob("*.PDF"))
                logger.info(f"  Found {len(pdf_files)} PDF files")
                if pdf_files:
                    self.class_mappings[class_name] = class_name
                    logger.info(f"  Added {class_name} to class mappings")
                else:
                    logger.warning(f"  No PDF files found in {class_path}")
            else:
                logger.warning(f"  Directory {class_path} does not exist")

        logger.info(f"Final class mappings: {self.class_mappings}")

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file, trying digital extraction first, then OCR.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text string
        """
        try:
            # Try digital extraction first
            text = extract_text_digital_pdf(str(pdf_path))
            if text.strip():
                return text
        except Exception as e:
            logger.debug(f"Digital extraction failed for {pdf_path}: {e}")

        try:
            # Fall back to OCR
            text = extract_text_scanned_pdf(str(pdf_path))
            return text
        except Exception as e:
            logger.warning(f"OCR extraction failed for {pdf_path}: {e}")
            return ""

    def load_documents(self) -> Tuple[List[str], List[str]]:
        """
        Load all documents from the dataset.

        Returns:
            Tuple of (texts, labels)
        """
        texts = []
        labels = []

        for class_name, class_dir in self.class_mappings.items():
            class_path = self.dataset_path / class_dir
            if not class_path.exists():
                logger.warning(f"Class directory {class_path} does not exist")
                continue

            logger.info(f"Loading documents from {class_name} class...")

            # Get all PDF files (both .pdf and .PDF extensions)
            pdf_files = list(class_path.glob("*.pdf")) + \
                list(class_path.glob("*.PDF"))
            if self.max_docs_per_class:
                pdf_files = pdf_files[:self.max_docs_per_class]

            logger.info(f"Found {len(pdf_files)} PDF files in {class_name}")

            for pdf_file in pdf_files:
                try:
                    text = self.extract_text_from_pdf(pdf_file)
                    if text.strip():  # Only add non-empty texts
                        texts.append(text)
                        labels.append(class_name)
                    else:
                        logger.warning(f"Empty text extracted from {pdf_file}")
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")

        logger.info(f"Total documents loaded: {len(texts)}")
        logger.info(
            f"Class distribution: {pd.Series(labels).value_counts().to_dict()}")

        return texts, labels

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess the extracted texts.

        Args:
            texts: List of raw text strings

        Returns:
            List of preprocessed text strings
        """
        logger.info("Preprocessing texts...")

        # Basic text normalization
        processed_texts = []
        for text in texts:
            # Remove excessive whitespace and normalize
            processed = normalize_text(text)
            # Remove very short texts (likely extraction errors)
            if len(processed.split()) > 10:
                processed_texts.append(processed)
            else:
                logger.debug(f"Skipping short text: {processed[:100]}...")

        logger.info(f"Preprocessed {len(processed_texts)} texts")
        return processed_texts

    def create_tfidf_features(self, texts: List[str], max_features: int = 10000) -> np.ndarray:
        """
        Create TF-IDF features from the texts.

        Args:
            texts: List of preprocessed text strings
            max_features: Maximum number of features for TF-IDF

        Returns:
            TF-IDF feature matrix
        """
        logger.info("Creating TF-IDF features...")

        # Check if we have any texts to process
        if not texts:
            raise ValueError(
                "No texts available for feature extraction. Please check that the dataset path is correct and contains PDF files.")

        if len(texts) < 2:
            raise ValueError(
                f"Need at least 2 documents for TF-IDF feature extraction, but only found {len(texts)}. Please check your dataset.")

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
            min_df=1,  # Changed from 2 to 1 to handle small datasets
            max_df=0.95,  # Maximum document frequency
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )

        features = self.tfidf_vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF features shape: {features.shape}")

        return features

    def prepare_dataset(self, test_size: float = 0.2, val_size: float = 0.2,
                        random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Prepare the complete dataset with train/validation/test splits.

        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing train/val/test splits
        """
        logger.info("Preparing dataset...")

        # Load and preprocess documents
        texts, labels = self.load_documents()

        # Check minimum document requirements
        min_docs_per_class = 5  # Minimum documents per class for meaningful training
        min_total_docs = 20     # Minimum total documents

        if len(texts) < min_total_docs:
            raise ValueError(
                f"Insufficient documents loaded: {len(texts)} < {min_total_docs}. "
                "Please check that PDF files can be processed successfully."
            )

        # Check class balance
        class_counts = pd.Series(labels).value_counts()
        min_class_count = class_counts.min()
        if min_class_count < min_docs_per_class:
            logger.warning(
                f"Some classes have very few documents. Minimum per class: {min_class_count} < {min_docs_per_class}. "
                "This may affect model performance."
            )

        processed_texts = self.preprocess_texts(texts)

        # Create TF-IDF features
        features = self.create_tfidf_features(processed_texts)

        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Split into train and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, encoded_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=encoded_labels
        )

        # Split remaining data into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.tfidf_vectorizer.get_feature_names_out() if hasattr(self.tfidf_vectorizer, 'get_feature_names_out') else None
        }

    def get_class_names(self) -> List[str]:
        """Get the list of class names."""
        return list(self.class_mappings.keys())

    def get_feature_names(self) -> Optional[List[str]]:
        """Get the feature names from TF-IDF vectorizer."""
        if self.tfidf_vectorizer:
            return self.tfidf_vectorizer.get_feature_names_out().tolist()
        return None
