"""Transformer training pipeline for contract classification with comprehensive comparison."""

import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import from the same directory
try:
    from .data_loader import ContractDataLoader
    from .transformer_models import (
        BERTClassifier, LegalBERTClassifier, RoBERTaClassifier,
        TransformerClassifier, TRANSFORMERS_AVAILABLE
    )
    from .enhanced_tfidf_models import EnhancedTFIDFClassifier
except ImportError:
    # When running as script
    from data_loader import ContractDataLoader
    from transformer_models import (
        BERTClassifier, LegalBERTClassifier, RoBERTaClassifier,
        TransformerClassifier, TRANSFORMERS_AVAILABLE
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

        logger.info(
            f"Initialized TransformerTrainingPipeline with output_dir: {self.output_dir}")

        # Check transformer availability
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "⚠️ Transformers library not available. Only baseline models will be trained.")
            logger.warning(
                "Install with: pip install torch transformers accelerate")

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

        # Initialize model
        bert_model = BERTClassifier(
            num_labels=self.dataset_info['n_classes'],
            max_length=kwargs.get('max_length', 512),
            random_state=self.random_state
        )

        # Training parameters
        training_args = {
            'output_dir': str(self.output_dir / 'models' / 'bert'),
            'num_epochs': kwargs.get('num_epochs', 3),
            'batch_size': kwargs.get('batch_size', 16),
            'learning_rate': kwargs.get('learning_rate', 2e-5),
            'warmup_steps': kwargs.get('warmup_steps', 500),
            'weight_decay': kwargs.get('weight_decay', 0.01),
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

        return self.results['bert']

    def train_legal_bert(self, dataset: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train LegalBERT model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error(
                "Transformers library not available for LegalBERT training")
            return {}

        logger.info("Training LegalBERT model...")

        # Initialize model
        legal_bert_model = LegalBERTClassifier(
            num_labels=self.dataset_info['n_classes'],
            max_length=kwargs.get('max_length', 512),
            random_state=self.random_state
        )

        # Training parameters
        training_args = {
            'output_dir': str(self.output_dir / 'models' / 'legal_bert'),
            'num_epochs': kwargs.get('num_epochs', 3),
            'batch_size': kwargs.get('batch_size', 16),
            'learning_rate': kwargs.get('learning_rate', 2e-5),
            'warmup_steps': kwargs.get('warmup_steps', 500),
            'weight_decay': kwargs.get('weight_decay', 0.01),
        }

        # Train model
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
            'model_name': 'LegalBERT'
        }

        logger.info(f"LegalBERT training completed:")
        logger.info(f"  Training loss: {train_results['train_loss']:.4f}")
        logger.info(f"  Validation accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"  Validation F1: {val_results['f1']:.4f}")

        return self.results['legal_bert']

    def train_roberta(self, dataset: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train RoBERTa model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error(
                "Transformers library not available for RoBERTa training")
            return {}

        logger.info("Training RoBERTa model...")

        # Initialize model
        roberta_model = RoBERTaClassifier(
            num_labels=self.dataset_info['n_classes'],
            max_length=kwargs.get('max_length', 512),
            random_state=self.random_state
        )

        # Training parameters
        training_args = {
            'output_dir': str(self.output_dir / 'models' / 'roberta'),
            'num_epochs': kwargs.get('num_epochs', 3),
            'batch_size': kwargs.get('batch_size', 16),
            'learning_rate': kwargs.get('learning_rate', 2e-5),
            'warmup_steps': kwargs.get('warmup_steps', 500),
            'weight_decay': kwargs.get('weight_decay', 0.01),
        }

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
            'model_name': 'RoBERTa'
        }

        logger.info(f"RoBERTa training completed:")
        logger.info(f"  Training loss: {train_results['train_loss']:.4f}")
        logger.info(f"  Validation accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"  Validation F1: {val_results['f1']:.4f}")

        return self.results['roberta']

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
            # Train BERT
            try:
                self.train_bert(dataset, **transformer_params)
            except Exception as e:
                logger.error(f"BERT training failed: {e}")

            # Train LegalBERT
            try:
                self.train_legal_bert(dataset, **transformer_params)
            except Exception as e:
                logger.error(f"LegalBERT training failed: {e}")

            # Train RoBERTa
            try:
                self.train_roberta(dataset, **transformer_params)
            except Exception as e:
                logger.error(f"RoBERTa training failed: {e}")
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
    # Initialize and run the pipeline
    pipeline = TransformerTrainingPipeline(
        dataset_path="Datasetss",
        output_dir="transformer_models_output",
        max_docs_per_class=None,  # Use all documents
        random_state=42
    )

    # Run the complete pipeline
    results = pipeline.run_full_pipeline(
        test_size=0.2,
        val_size=0.2,
        transformer_params={
            'num_epochs': 3,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'max_length': 512
        }
    )

    print("Training completed! Check the transformer_models_output directory for results.")
