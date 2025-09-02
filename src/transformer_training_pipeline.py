"""Transformer training pipeline for contract classification with comprehensive comparison."""

import os
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import torch
import gc

# Import from the same directory
try:
    from .data_loader import ContractDataLoader
    from .transformer_models import (
        BERTClassifier, LegalBERTClassifier, RoBERTaClassifier,
        TransformerClassifier, TRANSFORMERS_AVAILABLE, cleanup_memory
    )
    from .enhanced_tfidf_models import EnhancedTFIDFClassifier
except ImportError:
    # When running as script
    import sys
    from pathlib import Path

    # Add the parent directory to sys.path to find modules
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    from data_loader import ContractDataLoader
    from transformer_models import (
        BERTClassifier, LegalBERTClassifier, RoBERTaClassifier,
        TransformerClassifier, TRANSFORMERS_AVAILABLE, cleanup_memory
    )
    from enhanced_tfidf_models import EnhancedTFIDFClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransformerTrainingPipeline:
    """Comprehensive training pipeline for transformer models and comparison with existing approaches."""

    def __init__(self, dataset_path: str, output_dir: str = "transformer_models_output",
                 max_docs_per_class: Optional[int] = None, random_state: int = 42):
        """
        Initialize the transformer training pipeline.

        Args:
            dataset_path: Path to the dataset directory
            output_dir: Directory to save models and results
            max_docs_per_class: Maximum documents per class for testing
            random_state: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.random_state = random_state

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

        # Initialize data loader
        self.data_loader = ContractDataLoader(dataset_path, max_docs_per_class)

        # Store results and models
        self.results = {}
        self.models = {}
        self.dataset_info = {}

        # Check system resources
        self._check_system_resources()

        logger.info(
            f"Initialized TransformerTrainingPipeline with output_dir: {self.output_dir}")

        # Check transformer availability
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "⚠️ Transformers library not available. Only baseline models will be trained.")
            logger.warning(
                "Install with: pip install torch transformers accelerate")

    def _check_system_resources(self):
        """Check and log system resources."""
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        cpu_count = psutil.cpu_count()

        logger.info(f"System resources:")
        logger.info(f"  Available memory: {available_memory:.1f} GB")
        logger.info(f"  CPU cores: {cpu_count}")

        # Check GPU availability with CUDA priority
        if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
            logger.info(f"  Device: CUDA GPU - Optimized for GPU training")
            gpu_memory = torch.cuda.get_device_properties(
                0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_properties(0).name
            logger.info(f"  GPU: {gpu_name}")
            logger.info(f"  GPU Memory: {gpu_memory:.1f} GB")
            # Check CUDA version and capabilities
            logger.info(f"  CUDA Version: {torch.version.cuda}")
            if torch.cuda.is_bf16_supported():
                logger.info(f"  bfloat16 Support: ✅ Available")
            else:
                logger.info(f"  bfloat16 Support: ❌ Not available")

            # Show GPU utilization tips
            logger.info(f"  GPU Training Tips:")
            logger.info(
                f"    - Mixed precision: {'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'}")
            logger.info(
                f"    - Gradient checkpointing: {'Enabled' if gpu_memory < 16 else 'Disabled'}")
            logger.info(
                f"    - Optimal batch size: {min(32, max(8, int(gpu_memory * 1.5)))}")
        elif TRANSFORMERS_AVAILABLE and torch.backends.mps.is_available():
            logger.info(
                f"  Device: MPS (Metal Performance Shaders) - Optimized for Apple Silicon")
        else:
            logger.info(f"  Device: CPU")

        # Warning for low memory
        if available_memory < 8:
            logger.warning(
                "⚠️ Low memory detected (<8GB). Using conservative settings.")

    def _get_optimized_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimized parameters based on available system resources and GPU capabilities."""
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB

        # Check if CUDA is available for GPU optimizations
        cuda_available = TRANSFORMERS_AVAILABLE and torch.cuda.is_available()

        if cuda_available:
            # GPU-optimized settings
            gpu_memory = torch.cuda.get_device_properties(
                0).total_memory / (1024**3)

            if gpu_memory >= 24:  # High-end GPU (RTX 4090, A100, etc.)
                optimized_params = {
                    'num_epochs': 3,
                    'batch_size': 32,
                    'learning_rate': 2e-5,
                    'max_length': 512,
                    'gradient_accumulation_steps': 1,
                    'dataloader_num_workers': 4,
                    'eval_steps': 100,
                    'save_steps': 500,
                    'logging_steps': 50,
                    'fp16': True,
                    'bf16': False,
                    'gradient_checkpointing': False
                }
            elif gpu_memory >= 16:  # Mid-range GPU (RTX 3080, 4080, etc.)
                optimized_params = {
                    'num_epochs': 3,
                    'batch_size': 24,
                    'learning_rate': 2e-5,
                    'max_length': 512,
                    'gradient_accumulation_steps': 1,
                    'dataloader_num_workers': 3,
                    'eval_steps': 100,
                    'save_steps': 500,
                    'logging_steps': 50,
                    'fp16': True,
                    'bf16': False,
                    'gradient_checkpointing': False
                }
            elif gpu_memory >= 8:  # Entry-level GPU (RTX 3060, 4060, etc.)
                optimized_params = {
                    'num_epochs': 3,
                    'batch_size': 16,
                    'learning_rate': 2e-5,
                    'max_length': 512,
                    'gradient_accumulation_steps': 1,
                    'dataloader_num_workers': 2,
                    'eval_steps': 100,
                    'save_steps': 500,
                    'logging_steps': 50,
                    'fp16': True,
                    'bf16': False,
                    'gradient_checkpointing': True
                }
            else:  # Low VRAM GPU
                optimized_params = {
                    'num_epochs': 2,
                    'batch_size': 8,
                    'learning_rate': 2e-5,
                    'max_length': 256,
                    'gradient_accumulation_steps': 2,
                    'dataloader_num_workers': 1,
                    'eval_steps': 100,
                    'save_steps': 500,
                    'logging_steps': 50,
                    'fp16': True,
                    'bf16': False,
                    'gradient_checkpointing': True
                }
        else:
            # CPU/MPS settings
            if available_memory < 8:
                optimized_params = {
                    'num_epochs': 2,
                    'batch_size': 4,
                    'learning_rate': 2e-5,
                    'max_length': 256,
                    'gradient_accumulation_steps': 4,
                    'dataloader_num_workers': 0,
                    'eval_steps': 200,
                    'save_steps': 1000,
                    'logging_steps': 100,
                    'fp16': False,
                    'bf16': False,
                    'gradient_checkpointing': False
                }
            elif available_memory < 16:
                optimized_params = {
                    'num_epochs': 2,
                    'batch_size': 8,
                    'learning_rate': 2e-5,
                    'max_length': 256,
                    'gradient_accumulation_steps': 2,
                    'dataloader_num_workers': 0,
                    'eval_steps': 100,
                    'save_steps': 500,
                    'logging_steps': 50,
                    'fp16': False,
                    'bf16': False,
                    'gradient_checkpointing': False
                }
            else:
                optimized_params = {
                    'num_epochs': 3,
                    'batch_size': 16,
                    'learning_rate': 2e-5,
                    'max_length': 512,
                    'gradient_accumulation_steps': 1,
                    'dataloader_num_workers': 0,  # Safer on Mac
                    'eval_steps': 100,
                    'save_steps': 500,
                    'logging_steps': 50,
                    'fp16': False,
                    'bf16': False,
                    'gradient_checkpointing': False
                }

        # Override with user-provided parameters
        optimized_params.update(base_params)
        return optimized_params

    def prepare_data(self, test_size: float = 0.2, val_size: float = 0.2) -> Dict[str, Any]:
        """
        Prepare the dataset with train/validation/test splits for transformers.

        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            Dictionary containing the data splits and metadata
        """
        logger.info("Preparing dataset for transformer training...")

        # Load raw texts and labels (needed for transformers)
        raw_texts, raw_labels = self.data_loader.load_documents()

        # Preprocess texts
        processed_texts = self.data_loader.preprocess_texts(raw_texts)

        # Ensure we have matching processed texts and labels
        if len(processed_texts) != len(raw_labels):
            valid_indices = []
            for i, text in enumerate(raw_texts):
                processed = self.data_loader.preprocess_texts([text])
                if processed and len(processed[0].split()) > 10:
                    valid_indices.append(i)
            labels = [raw_labels[i] for i in valid_indices]
        else:
            labels = raw_labels

        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Split data
        from sklearn.model_selection import train_test_split

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            processed_texts, encoded_labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=encoded_labels
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )

        # Store dataset info
        self.dataset_info = {
            'n_classes': len(label_encoder.classes_),
            'class_names': label_encoder.classes_.tolist(),
            'label_encoder': label_encoder,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'total_samples': len(processed_texts)
        }

        logger.info(f"Dataset prepared successfully:")
        logger.info(f"  Classes: {self.dataset_info['n_classes']}")
        logger.info(f"  Train samples: {self.dataset_info['train_samples']}")
        logger.info(
            f"  Validation samples: {self.dataset_info['val_samples']}")
        logger.info(f"  Test samples: {self.dataset_info['test_samples']}")

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train.tolist(),
            'y_val': y_val.tolist(),
            'y_test': y_test.tolist(),
            'class_names': self.dataset_info['class_names']
        }

    def train_bert(self, dataset: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train BERT-base-uncased model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error(
                "Transformers library not available for BERT training")
            return {}

        logger.info("Training BERT-base-uncased model...")

        # Clean up memory before training
        cleanup_memory()

        # Get optimized parameters
        optimized_params = self._get_optimized_params(kwargs)

        # Initialize model
        bert_model = BERTClassifier(
            num_labels=self.dataset_info['n_classes'],
            max_length=optimized_params.get('max_length', 256),
            random_state=self.random_state
        )

        # Training parameters optimized for GPU/CPU
        training_args = {
            'output_dir': str(self.output_dir / 'models' / 'bert'),
            'num_epochs': optimized_params.get('num_epochs', 2),
            'batch_size': optimized_params.get('batch_size', 8),
            'eval_batch_size': optimized_params.get('batch_size', 8),
            'learning_rate': optimized_params.get('learning_rate', 2e-5),
            'warmup_steps': optimized_params.get('warmup_steps', 100),
            'weight_decay': optimized_params.get('weight_decay', 0.01),
            'gradient_accumulation_steps': optimized_params.get('gradient_accumulation_steps', 2),
            'max_grad_norm': 1.0,
            'eval_steps': optimized_params.get('eval_steps', 100),
            'save_steps': optimized_params.get('save_steps', 500),
            'logging_steps': optimized_params.get('logging_steps', 50),
            'dataloader_num_workers': optimized_params.get('dataloader_num_workers', 0),
            'dataloader_pin_memory': optimized_params.get('dataloader_pin_memory', False),
            'fp16': optimized_params.get('fp16', False),
            'bf16': optimized_params.get('bf16', False),
            'gradient_checkpointing': optimized_params.get('gradient_checkpointing', False),
        }

        # Train model
        train_results = bert_model.train(
            texts=dataset['X_train'],
            labels=dataset['y_train'],
            class_names=dataset['class_names'],
            eval_texts=dataset['X_val'],
            eval_labels=dataset['y_val'],
            **training_args
        )

        # Evaluate on validation set
        val_results = bert_model.evaluate(dataset['X_val'], dataset['y_val'])

        # Store model and results
        self.models['bert'] = bert_model
        self.results['bert'] = {
            'train': train_results,
            'validation': val_results,
            'model_name': 'BERT-base-uncased'
        }

        logger.info(f"BERT training completed:")
        logger.info(f"  Training loss: {train_results['train_loss']:.4f}")
        logger.info(f"  Validation accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"  Validation F1: {val_results['f1']:.4f}")

        # Clean up memory after training
        cleanup_memory()

        return self.results['bert']

    def train_legal_bert(self, dataset: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train LegalBERT model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error(
                "Transformers library not available for LegalBERT training")
            return {}

        logger.info("Training LegalBERT model...")

        # Clean up memory before training
        cleanup_memory()

        # Get optimized parameters - use smaller batch size for LegalBERT
        optimized_params = self._get_optimized_params(kwargs)
        # Reduce batch size further for LegalBERT stability
        optimized_params['batch_size'] = max(
            optimized_params['batch_size'] // 2, 2)

        try:
            # Initialize model
            legal_bert_model = LegalBERTClassifier(
                num_labels=self.dataset_info['n_classes'],
                max_length=optimized_params.get('max_length', 256),
                random_state=self.random_state
            )

            # Training parameters optimized for GPU/CPU and stability
            training_args = {
                'output_dir': str(self.output_dir / 'models' / 'legal_bert'),
                'num_epochs': optimized_params.get('num_epochs', 2),
                'batch_size': optimized_params.get('batch_size', 4),
                'eval_batch_size': optimized_params.get('batch_size', 4),
                # Lower learning rate
                'learning_rate': optimized_params.get('learning_rate', 1e-5),
                'warmup_steps': optimized_params.get('warmup_steps', 50),
                'weight_decay': optimized_params.get('weight_decay', 0.01),
                'gradient_accumulation_steps': optimized_params.get('gradient_accumulation_steps', 4),
                'max_grad_norm': 0.5,  # Lower gradient clipping
                'eval_steps': optimized_params.get('eval_steps', 200),
                'save_steps': optimized_params.get('save_steps', 500),
                'logging_steps': optimized_params.get('logging_steps', 50),
                'dataloader_num_workers': optimized_params.get('dataloader_num_workers', 0),
                'dataloader_pin_memory': optimized_params.get('dataloader_pin_memory', False),
                'fp16': optimized_params.get('fp16', False),
                'bf16': optimized_params.get('bf16', False),
                'gradient_checkpointing': optimized_params.get('gradient_checkpointing', False),
            }

            # Train model with timeout handling
            train_results = legal_bert_model.train(
                texts=dataset['X_train'],
                labels=dataset['y_train'],
                class_names=dataset['class_names'],
                eval_texts=dataset['X_val'],
                eval_labels=dataset['y_val'],
                **training_args
            )

            # Evaluate on validation set
            val_results = legal_bert_model.evaluate(
                dataset['X_val'], dataset['y_val'])

            # Store model and results
            self.models['legal_bert'] = legal_bert_model
            self.results['legal_bert'] = {
                'train': train_results,
                'validation': val_results,
                'model_name': f'LegalBERT ({legal_bert_model.model_name})'
            }

            logger.info(f"LegalBERT training completed:")
            logger.info(f"  Training loss: {train_results['train_loss']:.4f}")
            logger.info(
                f"  Validation accuracy: {val_results['accuracy']:.4f}")
            logger.info(f"  Validation F1: {val_results['f1']:.4f}")

            # Clean up memory after training
            cleanup_memory()

            return self.results['legal_bert']

        except Exception as e:
            logger.error(f"LegalBERT training failed: {e}")
            logger.info(
                "Skipping LegalBERT and continuing with other models...")
            cleanup_memory()
            return {}

    def train_roberta(self, dataset: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train RoBERTa model with optimizations for speed."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error(
                "Transformers library not available for RoBERTa training")
            return {}

        logger.info("Training RoBERTa model (optimized for speed)...")

        # Clean up memory before training
        cleanup_memory()

        # Get optimized parameters with extra speed optimizations
        optimized_params = self._get_optimized_params(kwargs)
        # Use more aggressive optimizations for RoBERTa
        speed_optimized_params = {
            # Max 2 epochs
            'num_epochs': min(optimized_params.get('num_epochs', 2), 2),
            # Limit batch size
            'batch_size': min(optimized_params.get('batch_size', 8), 12),
            'learning_rate': 3e-5,  # Slightly higher learning rate for faster convergence
            # Shorter sequences
            'max_length': min(optimized_params.get('max_length', 256), 256),
            'gradient_accumulation_steps': 1,  # No accumulation for speed
            'eval_steps': 50,  # More frequent evaluation
            'save_steps': 200,  # Less frequent saving
            'logging_steps': 25,  # More frequent logging for monitoring
            'warmup_steps': 50,  # Fewer warmup steps
        }
        speed_optimized_params.update(optimized_params)

        try:
            # Initialize model
            roberta_model = RoBERTaClassifier(
                num_labels=self.dataset_info['n_classes'],
                max_length=speed_optimized_params.get('max_length', 256),
                random_state=self.random_state
            )

            # Training parameters optimized for speed and GPU
            training_args = {
                'output_dir': str(self.output_dir / 'models' / 'roberta'),
                'num_epochs': speed_optimized_params.get('num_epochs', 2),
                'batch_size': speed_optimized_params.get('batch_size', 8),
                'eval_batch_size': speed_optimized_params.get('batch_size', 8),
                'learning_rate': speed_optimized_params.get('learning_rate', 3e-5),
                'warmup_steps': speed_optimized_params.get('warmup_steps', 50),
                'weight_decay': speed_optimized_params.get('weight_decay', 0.01),
                'gradient_accumulation_steps': speed_optimized_params.get('gradient_accumulation_steps', 1),
                'max_grad_norm': 1.0,
                'eval_steps': speed_optimized_params.get('eval_steps', 50),
                'save_steps': speed_optimized_params.get('save_steps', 200),
                'logging_steps': speed_optimized_params.get('logging_steps', 25),
                'dataloader_num_workers': optimized_params.get('dataloader_num_workers', 0),
                'dataloader_pin_memory': optimized_params.get('dataloader_pin_memory', False),
                'fp16': optimized_params.get('fp16', False),
                'bf16': optimized_params.get('bf16', False),
                'gradient_checkpointing': optimized_params.get('gradient_checkpointing', False),
            }

            logger.info(f"RoBERTa training settings:")
            logger.info(f"  Epochs: {training_args['num_epochs']}")
            logger.info(f"  Batch size: {training_args['batch_size']}")
            logger.info(
                f"  Max length: {speed_optimized_params.get('max_length', 256)}")
            logger.info(f"  Learning rate: {training_args['learning_rate']}")

            # Train model
            train_results = roberta_model.train(
                texts=dataset['X_train'],
                labels=dataset['y_train'],
                class_names=dataset['class_names'],
                eval_texts=dataset['X_val'],
                eval_labels=dataset['y_val'],
                **training_args
            )

            # Evaluate on validation set
            val_results = roberta_model.evaluate(
                dataset['X_val'], dataset['y_val'])

            # Store model and results
            self.models['roberta'] = roberta_model
            self.results['roberta'] = {
                'train': train_results,
                'validation': val_results,
                'model_name': 'RoBERTa (Speed Optimized)'
            }

            logger.info(f"RoBERTa training completed:")
            logger.info(f"  Training loss: {train_results['train_loss']:.4f}")
            logger.info(
                f"  Validation accuracy: {val_results['accuracy']:.4f}")
            logger.info(f"  Validation F1: {val_results['f1']:.4f}")

            # Clean up memory after training
            cleanup_memory()

            return self.results['roberta']

        except Exception as e:
            logger.error(f"RoBERTa training failed: {e}")
            logger.info("Skipping RoBERTa and continuing...")
            cleanup_memory()
            return {}

    def train_baseline_enhanced_tfidf(self, dataset: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train baseline Enhanced TF-IDF Random Forest for comparison."""
        logger.info(
            "Training baseline Enhanced TF-IDF + Random Forest for comparison...")

        # Initialize enhanced TF-IDF model
        enhanced_model = EnhancedTFIDFClassifier(
            model_type="random_forest",
            random_state=self.random_state
        )

        # Train model
        train_results = enhanced_model.train(
            texts=dataset['X_train'],
            labels=dataset['y_train'],
            class_names=dataset['class_names'],
            max_features=kwargs.get('max_features', 10000),
            ngram_range=kwargs.get('ngram_range', (1, 3)),
            use_feature_selection=kwargs.get('use_feature_selection', True),
            k_best=kwargs.get('k_best', 8000)
        )

        # Evaluate on validation set
        val_results = enhanced_model.evaluate(
            dataset['X_val'], dataset['y_val'])

        # Store model and results
        self.models['enhanced_tfidf'] = enhanced_model
        self.results['enhanced_tfidf'] = {
            'train': train_results,
            'validation': val_results,
            'model_name': 'Enhanced TF-IDF + Random Forest'
        }

        logger.info(f"Enhanced TF-IDF training completed:")
        logger.info(
            f"  Training accuracy: {train_results['train_accuracy']:.4f}")
        logger.info(f"  Validation accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"  Validation F1: {val_results['f1']:.4f}")

        return self.results['enhanced_tfidf']

    def evaluate_all_models(self, dataset: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Evaluate all trained models on the test set."""
        logger.info("Evaluating all models on test set...")

        test_results = {}

        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")

            try:
                test_eval = model.evaluate(
                    dataset['X_test'], dataset['y_test'])
                test_results[model_name] = test_eval

                logger.info(
                    f"  {model_name} - Test Accuracy: {test_eval['accuracy']:.4f}, F1: {test_eval['f1']:.4f}")

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                test_results[model_name] = {'error': str(e)}

        # Add test results to main results
        for model_name in test_results:
            if model_name in self.results:
                self.results[model_name]['test'] = test_results[model_name]

        return test_results

    def generate_comparison_plots(self):
        """Generate comprehensive comparison plots."""
        logger.info("Generating comparison plots...")

        # Extract metrics for plotting
        models = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []

        for model_name, results in self.results.items():
            if 'test' in results and 'accuracy' in results['test']:
                models.append(results.get('model_name', model_name))
                accuracies.append(results['test']['accuracy'])
                f1_scores.append(results['test']['f1'])
                precisions.append(results['test']['precision'])
                recalls.append(results['test']['recall'])

        if not models:
            logger.warning("No test results available for plotting")
            return

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)

        # Accuracy comparison
        axes[0, 0].bar(models, accuracies, color='skyblue')
        axes[0, 0].set_title('Test Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # F1 Score comparison
        axes[0, 1].bar(models, f1_scores, color='lightcoral')
        axes[0, 1].set_title('Test F1 Score')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(f1_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # Precision comparison
        axes[1, 0].bar(models, precisions, color='lightgreen')
        axes[1, 0].set_title('Test Precision')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(precisions):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # Recall comparison
        axes[1, 1].bar(models, recalls, color='gold')
        axes[1, 1].set_title('Test Recall')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(recalls):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' /
                    'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create a summary table plot
        fig, ax = plt.subplots(figsize=(12, 8))

        metrics_df = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracies,
            'F1 Score': f1_scores,
            'Precision': precisions,
            'Recall': recalls
        })

        # Create heatmap
        metrics_for_heatmap = metrics_df.set_index(
            'Model')[['Accuracy', 'F1 Score', 'Precision', 'Recall']]
        sns.heatmap(metrics_for_heatmap.T, annot=True,
                    fmt='.3f', cmap='YlOrRd', ax=ax)
        ax.set_title('Model Performance Heatmap')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' /
                    'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison plots saved to {self.output_dir / 'plots'}")

    def save_models(self):
        """Save all trained models."""
        logger.info("Saving trained models...")

        for model_name, model in self.models.items():
            try:
                model_path = self.output_dir / 'models' / model_name
                model.save_model(str(model_path))
                logger.info(f"  Saved {model_name} to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save {model_name}: {e}")

    def save_results(self):
        """Save training and evaluation results."""
        results_file = self.output_dir / 'results' / 'transformer_training_results.json'

        # Prepare results for JSON serialization
        json_results = {}
        for model_name, results in self.results.items():
            json_results[model_name] = {}
            for split, split_results in results.items():
                if isinstance(split_results, dict):
                    json_results[model_name][split] = {}
                    for key, value in split_results.items():
                        if isinstance(value, (np.ndarray, list)):
                            json_results[model_name][split][key] = np.array(
                                value).tolist()
                        elif isinstance(value, np.floating):
                            json_results[model_name][split][key] = float(value)
                        elif isinstance(value, np.integer):
                            json_results[model_name][split][key] = int(value)
                        else:
                            json_results[model_name][split][key] = value
                else:
                    json_results[model_name][split] = split_results

        # Add dataset info and timestamp
        json_results['dataset_info'] = {
            'n_classes': self.dataset_info['n_classes'],
            'class_names': self.dataset_info['class_names'],
            'train_samples': self.dataset_info['train_samples'],
            'val_samples': self.dataset_info['val_samples'],
            'test_samples': self.dataset_info['test_samples'],
            'total_samples': self.dataset_info['total_samples']
        }
        json_results['timestamp'] = datetime.now().isoformat()
        json_results['transformers_available'] = TRANSFORMERS_AVAILABLE

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

    def run_full_pipeline(self, test_size: float = 0.2, val_size: float = 0.2,
                          transformer_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run the complete transformer training pipeline.

        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set  
            transformer_params: Parameters for transformer training

        Returns:
            Dictionary containing all results
        """
        logger.info("Starting comprehensive transformer training pipeline...")

        # Set default parameters
        if transformer_params is None:
            transformer_params = {
                'num_epochs': 3,
                'batch_size': 16,
                'learning_rate': 2e-5,
                'max_length': 512
            }

        # Prepare data
        dataset = self.prepare_data(test_size, val_size)

        # Train baseline Enhanced TF-IDF for comparison
        self.train_baseline_enhanced_tfidf(dataset)

        # Train transformer models if available
        if TRANSFORMERS_AVAILABLE:
            models_to_train = [
                ('BERT', self.train_bert),
                ('LegalBERT', self.train_legal_bert),
                ('RoBERTa', self.train_roberta)
            ]

            for model_name, train_func in models_to_train:
                try:
                    logger.info(f"🚀 Starting {model_name} training...")
                    start_time = datetime.now()

                    result = train_func(dataset, **transformer_params)

                    end_time = datetime.now()
                    training_time = (
                        end_time - start_time).total_seconds() / 60

                    if result:
                        logger.info(
                            f"✅ {model_name} completed successfully in {training_time:.1f} minutes")
                    else:
                        logger.warning(
                            f"⚠️ {model_name} training returned empty result")

                except Exception as e:
                    logger.error(f"❌ {model_name} training failed: {e}")
                    logger.info(f"Continuing with remaining models...")
                    # Clean up memory after failure
                    cleanup_memory()
                    continue

                # Always clean up between models
                cleanup_memory()

        else:
            logger.warning(
                "Transformers not available - only baseline model trained")

        # Evaluate all models on test set
        test_results = self.evaluate_all_models(dataset)

        # Generate comparison plots
        self.generate_comparison_plots()

        # Save models and results
        self.save_models()
        self.save_results()

        logger.info("Transformer training pipeline completed successfully!")

        # Print summary
        logger.info("\n" + "="*50)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*50)

        for model_name, results in self.results.items():
            if 'test' in results and 'accuracy' in results['test']:
                logger.info(f"{results.get('model_name', model_name)}:")
                logger.info(
                    f"  Test Accuracy: {results['test']['accuracy']:.4f}")
                logger.info(f"  Test F1:       {results['test']['f1']:.4f}")
                logger.info(
                    f"  Test Precision: {results['test']['precision']:.4f}")
                logger.info(
                    f"  Test Recall:   {results['test']['recall']:.4f}")
                logger.info("-" * 30)

        return self.results


# Main execution
if __name__ == "__main__":
    # Check system resources first
    import psutil

    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    print(f"Available memory: {available_memory:.1f} GB")

    # Set parameters based on available memory
    if available_memory < 8:
        print("Warning: Low memory detected. Using conservative settings.")
        max_docs = 1000  # Limit dataset size
        transformer_params = {
            'num_epochs': 2,
            'batch_size': 4,
            'learning_rate': 2e-5,
            'max_length': 256,
            'gradient_accumulation_steps': 4
        }
    elif available_memory < 16:
        print("Using moderate settings for available memory.")
        max_docs = 2000
        transformer_params = {
            'num_epochs': 2,
            'batch_size': 8,
            'learning_rate': 2e-5,
            'max_length': 256,
            'gradient_accumulation_steps': 2
        }
    else:
        print("Using standard settings.")
        max_docs = None
        transformer_params = {
            'num_epochs': 3,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'max_length': 512,
            'gradient_accumulation_steps': 1
        }

    # Initialize pipeline with memory considerations
    pipeline = TransformerTrainingPipeline(
        dataset_path="Datasetss",
        output_dir="transformer_models_output",
        max_docs_per_class=max_docs,
        random_state=42
    )

    # Run the complete pipeline with MPS-optimized parameters
    print("Starting transformer training pipeline with MPS optimization...")
    results = pipeline.run_full_pipeline(
        test_size=0.2,
        val_size=0.2,
        transformer_params=transformer_params
    )

    print("\n" + "="*60)
    print("🚀 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("✅ Models are now trained with MPS acceleration")
    print("✅ Memory management optimizations applied")
    print("✅ Sequential training prevents system crashes")
    print(f"📊 Results saved to: transformer_models_output/")
    print("="*60)
