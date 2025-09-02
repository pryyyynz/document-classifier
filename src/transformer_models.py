"""Transformer models for contract classification with fine-tuning capabilities."""

import os
import logging
import json
import pickle
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Core ML and data science
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Transformers imports
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback
    )
    from transformers.trainer_utils import set_seed
    import torch.nn.functional as F

    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers library available")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"⚠️ Transformers library not available: {e}")
    logger.warning("Install with: pip install torch transformers accelerate")

# Check for device availability with CUDA priority for GPU environments
if TRANSFORMERS_AVAILABLE:
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        logger.info(f"Using device: {DEVICE} (CUDA GPU)")
        # Set CUDA memory management
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("✅ Flash attention enabled for CUDA")
        except:
            logger.info(
                "Flash attention not available, using standard attention")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        logger.info(f"Using device: {DEVICE} (Metal Performance Shaders)")
    else:
        DEVICE = torch.device('cpu')
        logger.info(f"Using device: {DEVICE}")

    # Memory management utilities
    def cleanup_memory():
        """Clean up GPU memory."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("CUDA memory cleaned and synchronized")
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.debug("MPS memory cleaned")
        except Exception as e:
            logger.warning(f"Memory cleanup warning: {e}")
        import gc
        gc.collect()

    def get_gpu_info():
        """Get detailed GPU information for optimization."""
        if not torch.cuda.is_available():
            return None

        try:
            gpu_props = torch.cuda.get_device_properties(0)
            return {
                'name': gpu_props.name,
                'memory_total': gpu_props.total_memory / (1024**3),  # GB
                'memory_free': torch.cuda.memory_reserved(0) / (1024**3),  # GB
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                'multi_processor_count': gpu_props.multi_processor_count,
                'supports_bf16': torch.cuda.is_bf16_supported(),
                'supports_fp16': True,  # All CUDA GPUs support fp16
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return None


class ContractDataset(Dataset):
    """Dataset class for contract texts and labels."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize the dataset.

        Args:
            texts: List of contract texts
            labels: List of integer labels
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TransformerClassifier:
    """Base class for transformer-based contract classification."""

    def __init__(self, model_name: str, num_labels: int, max_length: int = 512, random_state: int = 42):
        """
        Initialize the transformer classifier.

        Args:
            model_name: Name of the pre-trained model
            num_labels: Number of classification labels
            max_length: Maximum sequence length
            random_state: Random seed
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library is required but not available. "
                "Install with: pip install torch transformers accelerate"
            )

        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.random_state = random_state

        # Set seed for reproducibility
        set_seed(random_state)

        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.trainer = None

        self.is_trained = False
        self.class_names = None

        # Get GPU info for optimization
        self.gpu_info = get_gpu_info()
        if self.gpu_info:
            logger.info(
                f"GPU detected: {self.gpu_info['name']} ({self.gpu_info['memory_total']:.1f}GB)")
            logger.info(
                f"Compute capability: {self.gpu_info['compute_capability']}")
            logger.info(
                f"bfloat16 support: {'✅' if self.gpu_info['supports_bf16'] else '❌'}")

        logger.info(
            f"Initialized {self.__class__.__name__} with model: {model_name}")

    def _load_model(self):
        """Load tokenizer and model."""
        try:
            logger.info(f"Loading tokenizer and model: {self.model_name}")

            # Load tokenizer with error handling
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,
                    trust_remote_code=False
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load fast tokenizer, falling back to slow: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=False,
                    trust_remote_code=False
                )

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with error handling
            try:
                # Use mixed precision for CUDA if available
                if torch.cuda.is_available():
                    # Check if bf16 is available (better than fp16 for training)
                    if torch.cuda.is_bf16_supported():
                        torch_dtype = torch.bfloat16
                        logger.info(
                            "Using bfloat16 precision for CUDA training")
                    else:
                        torch_dtype = torch.float16
                        logger.info(
                            "Using float16 precision for CUDA training")
                else:
                    torch_dtype = torch.float32
                    logger.info("Using float32 precision for CPU/MPS training")

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=self.num_labels,
                    problem_type="single_label_classification",
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load with optimizations, falling back: {e}")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=self.num_labels,
                    problem_type="single_label_classification"
                )

            # Move to device with error handling
            try:
                self.model.to(DEVICE)
                logger.info(f"✅ Model loaded successfully on {DEVICE}")
            except Exception as e:
                logger.error(f"Failed to move model to {DEVICE}: {e}")
                # Fallback to CPU
                fallback_device = torch.device('cpu')
                self.model.to(fallback_device)
                logger.warning(f"Fallback: Model loaded on {fallback_device}")

        except Exception as e:
            logger.error(f"❌ Failed to load model {self.model_name}: {e}")
            raise

    def prepare_data(self, texts: List[str], labels: List[int],
                     test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
        """
        Prepare datasets for training and evaluation.

        Args:
            texts: List of contract texts
            labels: List of integer labels
            test_size: Proportion for test set

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.tokenizer is None:
            self._load_model()

        # Split data
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=self.random_state, stratify=labels
        )

        # Create datasets
        train_dataset = ContractDataset(
            train_texts, train_labels, self.tokenizer, self.max_length)
        eval_dataset = ContractDataset(
            eval_texts, eval_labels, self.tokenizer, self.max_length)

        logger.info(
            f"Prepared datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        return train_dataset, eval_dataset

    def train(self, texts: List[str], labels: List[int],
              class_names: Optional[List[str]] = None,
              eval_texts: Optional[List[str]] = None,
              eval_labels: Optional[List[int]] = None,
              **training_args) -> Dict[str, Any]:
        """
        Fine-tune the transformer model.

        Args:
            texts: Training texts
            labels: Training labels
            class_names: Names of classes
            eval_texts: Evaluation texts (optional)
            eval_labels: Evaluation labels (optional)
            **training_args: Training arguments

        Returns:
            Dictionary containing training results
        """
        logger.info(f"Starting fine-tuning of {self.model_name}")

        self.class_names = class_names

        if self.model is None:
            self._load_model()

        # Prepare datasets
        if eval_texts is not None and eval_labels is not None:
            train_dataset = ContractDataset(
                texts, labels, self.tokenizer, self.max_length)
            eval_dataset = ContractDataset(
                eval_texts, eval_labels, self.tokenizer, self.max_length)
        else:
            train_dataset, eval_dataset = self.prepare_data(
                texts, labels, test_size=0.2)

        # Set up training arguments
        output_dir = training_args.get('output_dir', './transformer_results')

        # Enable mixed precision based on device
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                fp16 = False
                bf16 = True
                logger.info("Using bfloat16 mixed precision for CUDA training")
            else:
                fp16 = True
                bf16 = False
                logger.info("Using float16 mixed precision for CUDA training")
        else:
            fp16 = False
            bf16 = False
            logger.info("Mixed precision disabled for CPU/MPS training")

        # Dynamically adjust batch size based on GPU memory if available
        if self.gpu_info and torch.cuda.is_available():
            gpu_memory = self.gpu_info['memory_total']
            requested_batch_size = training_args.get('batch_size', 16)

            # Adjust batch size based on GPU memory
            if gpu_memory >= 24:  # High-end GPU
                max_batch_size = 32
            elif gpu_memory >= 16:  # Mid-range GPU
                max_batch_size = 24
            elif gpu_memory >= 8:  # Entry-level GPU
                max_batch_size = 16
            else:  # Low VRAM GPU
                max_batch_size = 8

            # Use the smaller of requested or max supported
            adjusted_batch_size = min(requested_batch_size, max_batch_size)
            if adjusted_batch_size != requested_batch_size:
                logger.info(
                    f"Adjusted batch size from {requested_batch_size} to {adjusted_batch_size} based on GPU memory ({gpu_memory:.1f}GB)")
                training_args['batch_size'] = adjusted_batch_size

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_args.get('num_epochs', 3),
            per_device_train_batch_size=training_args.get('batch_size', 16),
            per_device_eval_batch_size=training_args.get(
                'eval_batch_size', 16),
            warmup_steps=training_args.get('warmup_steps', 500),
            weight_decay=training_args.get('weight_decay', 0.01),
            learning_rate=training_args.get('learning_rate', 2e-5),
            gradient_accumulation_steps=training_args.get(
                'gradient_accumulation_steps', 1),
            max_grad_norm=training_args.get('max_grad_norm', 1.0),

            # Logging and evaluation
            logging_dir=f'{output_dir}/logs',
            logging_steps=training_args.get('logging_steps', 50),
            eval_steps=training_args.get('eval_steps', 100),
            save_steps=training_args.get('save_steps', 500),
            eval_strategy="steps",
            save_strategy="steps",

            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",  # Use F1 instead of accuracy
            greater_is_better=True,

            # Memory and performance optimizations
            dataloader_num_workers=training_args.get(
                'dataloader_num_workers', 2 if torch.cuda.is_available() else 0),
            dataloader_pin_memory=training_args.get(
                'dataloader_pin_memory', True if torch.cuda.is_available() else False),
            remove_unused_columns=False,
            fp16=fp16,
            bf16=bf16,

            # GPU-specific optimizations
            gradient_checkpointing=training_args.get(
                'gradient_checkpointing', True if torch.cuda.is_available() else False),
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",

            # Disable external logging
            push_to_hub=False,
            report_to=None,

            # Save disk space
            save_total_limit=2,  # Only keep 2 checkpoints
            resume_from_checkpoint=None,  # Don't auto-resume for cleaner runs
        )

        # Define compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted', zero_division=0
            )

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Start training
        logger.info("Starting training...")

        # Monitor GPU memory if available
        if torch.cuda.is_available():
            logger.info(
                f"GPU memory before training: {torch.cuda.memory_allocated(0) / (1024**3):.2f}GB")

        train_result = self.trainer.train()

        # Evaluate
        logger.info("Evaluating model...")
        eval_result = self.trainer.evaluate()

        # Log final GPU memory usage
        if torch.cuda.is_available():
            logger.info(
                f"GPU memory after training: {torch.cuda.memory_allocated(0) / (1024**3):.2f}GB")

        self.is_trained = True

        # Prepare results
        results = {
            'train_loss': train_result.training_loss,
            'eval_accuracy': eval_result['eval_accuracy'],
            'eval_precision': eval_result['eval_precision'],
            'eval_recall': eval_result['eval_recall'],
            'eval_f1': eval_result['eval_f1'],
            'model_name': self.model_name,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

        logger.info(f"Training completed:")
        logger.info(f"  Final training loss: {results['train_loss']:.4f}")
        logger.info(f"  Eval accuracy: {results['eval_accuracy']:.4f}")
        logger.info(f"  Eval F1: {results['eval_f1']:.4f}")

        return results

    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions on new texts."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        # Prepare dataset
        dummy_labels = [0] * len(texts)  # Dummy labels for inference
        dataset = ContractDataset(
            texts, dummy_labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        predictions = []
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)

                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
                logits = outputs.logits

                batch_predictions = torch.argmax(logits, dim=-1)
                predictions.extend(batch_predictions.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        # Prepare dataset
        dummy_labels = [0] * len(texts)  # Dummy labels for inference
        dataset = ContractDataset(
            texts, dummy_labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        probabilities = []
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)

                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
                logits = outputs.logits

                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def evaluate(self, texts: List[str], y_true: List[int]) -> Dict[str, Any]:
        """Evaluate model performance on test data."""
        y_pred = self.predict(texts)
        y_proba = self.predict_proba(texts)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Calculate per-class metrics
        per_class_metrics = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # ROC AUC (if binary or multiclass)
        roc_auc = None
        try:
            if len(np.unique(y_true)) == 2:
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
            elif len(np.unique(y_true)) > 2:
                roc_auc = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='weighted')
        except Exception:
            pass

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_precision': per_class_metrics[0],
            'per_class_recall': per_class_metrics[1],
            'per_class_f1': per_class_metrics[2],
            'per_class_support': per_class_metrics[3],
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

        if roc_auc is not None:
            results['roc_auc'] = roc_auc

        # Classification report
        if self.class_names:
            results['classification_report'] = classification_report(
                y_true, y_pred, target_names=self.class_names, output_dict=True
            )
        else:
            results['classification_report'] = classification_report(
                y_true, y_pred, output_dict=True
            )

        return results

    def save_model(self, filepath: str):
        """Save the fine-tuned model and tokenizer."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")

        save_path = Path(filepath)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'class_names': self.class_names,
            'random_state': self.random_state
        }

        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    def load_model(self, filepath: str):
        """Load a fine-tuned model and tokenizer."""
        load_path = Path(filepath)

        if not load_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {load_path}")

        # Load metadata
        with open(load_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.model_name = metadata['model_name']
        self.num_labels = metadata['num_labels']
        self.max_length = metadata['max_length']
        self.class_names = metadata.get('class_names')

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_path)
        self.model.to(DEVICE)

        self.is_trained = True

        logger.info(f"Model loaded from {load_path}")


class BERTClassifier(TransformerClassifier):
    """BERT-base-uncased classifier for contract classification."""

    def __init__(self, num_labels: int, max_length: int = 512, random_state: int = 42):
        """Initialize BERT classifier."""
        super().__init__(
            model_name='bert-base-uncased',
            num_labels=num_labels,
            max_length=max_length,
            random_state=random_state
        )


class LegalBERTClassifier(TransformerClassifier):
    """LegalBERT classifier for contract classification."""

    def __init__(self, num_labels: int, max_length: int = 512, random_state: int = 42):
        """Initialize LegalBERT classifier."""
        # Try multiple LegalBERT models in order of preference
        legal_bert_models = [
            'nlpaueb/legal-bert-small-uncased',  # Smaller, more stable
            'nlpaueb/legal-bert-base-uncased',   # Original choice
            'bert-base-uncased'  # Fallback to regular BERT
        ]

        model_name = None
        for candidate in legal_bert_models:
            try:
                # Test if model is accessible
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(candidate, use_fast=False)
                model_name = candidate
                logger.info(f"Using LegalBERT model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to access {candidate}: {e}")
                continue

        if model_name is None:
            model_name = 'bert-base-uncased'
            logger.warning(
                "Falling back to BERT-base-uncased for legal classification")

        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            max_length=max_length,
            random_state=random_state
        )


class RoBERTaClassifier(TransformerClassifier):
    """RoBERTa classifier for contract classification."""

    def __init__(self, num_labels: int, max_length: int = 512, random_state: int = 42):
        """Initialize RoBERTa classifier."""
        super().__init__(
            model_name='roberta-base',
            num_labels=num_labels,
            max_length=max_length,
            random_state=random_state
        )


# Utility functions for transformer training
def get_available_models():
    """Get list of available transformer models."""
    if not TRANSFORMERS_AVAILABLE:
        return []

    return [
        'bert-base-uncased',
        'nlpaueb/legal-bert-base-uncased',  # LegalBERT
        'roberta-base',
        'distilbert-base-uncased'
    ]


def create_transformer_classifier(model_name: str, num_labels: int, **kwargs):
    """Factory function to create transformer classifiers."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library not available")

    if model_name == 'bert-base-uncased':
        return BERTClassifier(num_labels=num_labels, **kwargs)
    elif model_name == 'nlpaueb/legal-bert-base-uncased':
        return LegalBERTClassifier(num_labels=num_labels, **kwargs)
    elif model_name == 'roberta-base':
        return RoBERTaClassifier(num_labels=num_labels, **kwargs)
    else:
        return TransformerClassifier(model_name=model_name, num_labels=num_labels, **kwargs)


# Export main classes and functions
__all__ = [
    'TransformerClassifier',
    'BERTClassifier',
    'LegalBERTClassifier',
    'RoBERTaClassifier',
    'get_available_models',
    'create_transformer_classifier',
    'TRANSFORMERS_AVAILABLE'
]
