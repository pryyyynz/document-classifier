"""Enhanced training pipeline for contract classification using advanced TF-IDF techniques."""

import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

# Import from the same directory - handle both module and script execution
try:
    from .data_loader import ContractDataLoader
    from .enhanced_tfidf_models import (
        EnhancedTFIDFClassifier,
        TopicModelingClassifier,
        plot_model_comparison,
        plot_feature_importance
    )
except ImportError:
    # When running as script
    from data_loader import ContractDataLoader
    from enhanced_tfidf_models import (
        EnhancedTFIDFClassifier,
        TopicModelingClassifier,
        plot_model_comparison,
        plot_feature_importance
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedTrainingPipeline:
    """Enhanced training pipeline for contract classification using advanced NLP techniques."""

    def __init__(self, dataset_path: str, output_dir: str = "enhanced_models_output",
                 max_docs_per_class: Optional[int] = None, random_state: int = 42):
        """
        Initialize the enhanced training pipeline.

        Args:
            dataset_path: Path to the dataset directory
            output_dir: Directory to save models and results
            max_docs_per_class: Maximum documents per class (for testing)
            random_state: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.max_docs_per_class = max_docs_per_class
        self.random_state = random_state

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "features").mkdir(exist_ok=True)

        # Initialize data loader
        self.data_loader = ContractDataLoader(dataset_path, max_docs_per_class)

        # Store results
        self.results = {}
        self.dataset_info = {}
        self.models = {}

    def prepare_data(self, test_size: float = 0.2, val_size: float = 0.2) -> Dict[str, Any]:
        """
        Prepare the dataset with train/validation/test splits.

        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            Dictionary containing the data splits and metadata
        """
        logger.info("Preparing dataset for enhanced NLP approaches...")

        # Load raw texts and labels first (before TF-IDF processing)
        raw_texts, raw_labels = self.data_loader.load_documents()

        # Preprocess the raw texts
        processed_texts = self.data_loader.preprocess_texts(raw_texts)

        # Ensure we have matching processed texts and labels
        if len(processed_texts) != len(raw_labels):
            # Filter labels to match processed texts length if some were filtered out
            valid_indices = []
            processed_idx = 0
            for i, text in enumerate(raw_texts):
                processed = self.data_loader.preprocess_texts([text])
                # Same filtering logic as in preprocess_texts
                if processed and len(processed[0].split()) > 10:
                    valid_indices.append(i)
            labels = [raw_labels[i] for i in valid_indices]
        else:
            labels = raw_labels

        # Now split the texts and labels
        from sklearn.model_selection import train_test_split

        # Split into train and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            processed_texts, labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels
        )

        # Split remaining data into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )

        # Store dataset info
        self.dataset_info = {
            'n_classes': len(self.data_loader.get_class_names()),
            'class_names': self.data_loader.get_class_names(),
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
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'class_names': self.dataset_info['class_names']
        }

    def train_enhanced_tfidf(self, dataset: Dict[str, Any], model_type: str = "random_forest", **kwargs) -> Dict[str, Any]:
        """
        Train an enhanced TF-IDF classifier.

        Args:
            dataset: Dataset dictionary with train/val/test splits
            model_type: Type of classifier to use
            **kwargs: Additional training parameters

        Returns:
            Training results
        """
        logger.info(
            f"Training enhanced TF-IDF classifier with {model_type}...")

        # Initialize model
        enhanced_model = EnhancedTFIDFClassifier(
            model_type=model_type,
            random_state=self.random_state
        )

        # Train model
        train_results = enhanced_model.train(
            texts=dataset['X_train'],
            labels=dataset['y_train'],
            class_names=dataset['class_names'],
            **kwargs
        )

        # Evaluate on validation set
        val_results = enhanced_model.evaluate(
            dataset['X_val'], dataset['y_val'])

        # Store model
        model_key = f"enhanced_tfidf_{model_type}"
        self.models[model_key] = enhanced_model

        # Store results
        self.results[model_key] = {
            'train': train_results,
            'validation': val_results,
            'model_name': train_results['model_name']
        }

        logger.info(f"Enhanced TF-IDF {model_type} training completed:")
        logger.info(
            f"  Training accuracy: {train_results['train_accuracy']:.4f}")
        logger.info(f"  Validation accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"  Validation F1: {val_results['f1']:.4f}")

        return self.results[model_key]

    def train_topic_modeling(self, dataset: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Train a topic modeling classifier.

        Args:
            dataset: Dataset dictionary with train/val/test splits
            **kwargs: Additional training parameters

        Returns:
            Training results
        """
        logger.info("Training topic modeling classifier...")

        # Initialize model
        topic_model = TopicModelingClassifier(
            n_topics=kwargs.get('n_topics', 50),
            random_state=self.random_state
        )

        # Train model
        train_results = topic_model.train(
            texts=dataset['X_train'],
            labels=dataset['y_train'],
            class_names=dataset['class_names'],
            **kwargs
        )

        # Evaluate on validation set
        val_results = topic_model.evaluate(dataset['X_val'], dataset['y_val'])

        # Store model
        self.models['topic_modeling'] = topic_model

        # Store results
        self.results['topic_modeling'] = {
            'train': train_results,
            'validation': val_results,
            'model_name': train_results['model_name']
        }

        logger.info(f"Topic modeling training completed:")
        logger.info(
            f"  Training accuracy: {train_results['train_accuracy']:.4f}")
        logger.info(f"  Validation accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"  Validation F1: {val_results['f1']:.4f}")
        logger.info(f"  Topics extracted: {len(train_results['topic_names'])}")

        return self.results['topic_modeling']

    def evaluate_all_models(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all trained models on the test set.

        Args:
            dataset: Dataset dictionary with test split

        Returns:
            Evaluation results for all models
        """
        logger.info("Evaluating all models on test set...")

        test_results = {}

        for model_name, model in self.models.items():
            if model.is_trained:
                logger.info(f"Evaluating {model_name}...")
                results = model.evaluate(dataset['X_test'], dataset['y_test'])
                test_results[model_name] = results

                logger.info(f"{model_name} test results:")
                logger.info(f"  Accuracy: {results['accuracy']:.4f}")
                logger.info(f"  F1: {results['f1']:.4f}")

        # Store test results
        self.results['test_results'] = test_results

        return test_results

    def save_models(self):
        """Save all trained models."""
        logger.info("Saving trained models...")

        for model_name, model in self.models.items():
            if model.is_trained:
                model_path = self.output_dir / \
                    "models" / f"{model_name}_model.pkl"
                model.save_model(str(model_path))
                logger.info(f"{model_name} model saved to {model_path}")

    def save_results(self):
        """Save training results and metadata."""
        logger.info("Saving training results...")

        # Save results summary
        results_summary = {
            'dataset_info': self.dataset_info,
            'training_results': self.results,
            'timestamp': datetime.now().isoformat(),
            'random_state': self.random_state
        }

        results_path = self.output_dir / "results" / "enhanced_training_summary.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)

        # Save detailed results
        for model_name, results in self.results.items():
            if model_name != 'test_results':
                continue

            for model, metrics in results.items():
                results_path = self.output_dir / \
                    "results" / f"{model}_test_results.json"
                with open(results_path, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Results saved to {self.output_dir}/results/")

    def generate_comparison_plots(self):
        """Generate comparison plots for all models."""
        logger.info("Generating comparison plots...")

        if 'test_results' not in self.results:
            logger.warning("No test results available for plotting")
            return

        # Create comparison plot
        plot_path = self.output_dir / "plots" / "enhanced_model_comparison.png"
        plot_model_comparison(self.results['test_results'], str(plot_path))

        logger.info(f"Comparison plot saved to {plot_path}")

    def generate_feature_importance_plots(self):
        """Generate feature importance plots for applicable models."""
        logger.info("Generating feature importance plots...")

        for model_name, model in self.models.items():
            if hasattr(model, 'get_feature_importance') and model.is_trained:
                try:
                    plot_path = self.output_dir / "plots" / \
                        f"{model_name}_feature_importance.png"
                    plot_feature_importance(
                        model, top_n=30, save_path=str(plot_path))
                    logger.info(
                        f"Feature importance plot saved for {model_name}")
                except Exception as e:
                    logger.warning(
                        f"Could not generate feature importance plot for {model_name}: {e}")

    def run_full_pipeline(self, test_size: float = 0.2, val_size: float = 0.2) -> Dict[str, Any]:
        """
        Run the complete enhanced training pipeline.

        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            Dictionary containing all results
        """
        logger.info("Starting enhanced training pipeline...")

        # Prepare data
        dataset = self.prepare_data(test_size, val_size)

        # Train different enhanced TF-IDF models
        logger.info("Training multiple enhanced TF-IDF models...")

        # Random Forest with enhanced features
        rf_results = self.train_enhanced_tfidf(
            dataset,
            model_type="random_forest",
            max_features=15000,
            ngram_range=(1, 4),
            use_feature_selection=True,
            k_best=12000,
            n_estimators=300,
            max_depth=25
        )

        # SVM with enhanced features
        svm_results = self.train_enhanced_tfidf(
            dataset,
            model_type="svm",
            max_features=12000,
            ngram_range=(1, 3),
            use_feature_selection=True,
            k_best=10000,
            C=15.0,
            gamma='scale'
        )

        # Logistic Regression with enhanced features
        lr_results = self.train_enhanced_tfidf(
            dataset,
            model_type="logistic",
            max_features=10000,
            ngram_range=(1, 3),
            use_feature_selection=True,
            k_best=8000,
            C=1.0,
            max_iter=1000
        )

        # Gradient Boosting with enhanced features
        gb_results = self.train_enhanced_tfidf(
            dataset,
            model_type="gradient_boosting",
            max_features=12000,
            ngram_range=(1, 3),
            use_feature_selection=True,
            k_best=10000,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10
        )

        # Topic Modeling
        topic_results = self.train_topic_modeling(
            dataset,
            n_topics=60,
            max_features=10000,
            ngram_range=(1, 2),
            max_iter=25
        )

        # Evaluate all models
        test_results = self.evaluate_all_models(dataset)

        # Generate plots
        self.generate_comparison_plots()
        self.generate_feature_importance_plots()

        # Save models and results
        self.save_models()
        self.save_results()

        logger.info("Enhanced training pipeline completed successfully!")

        return self.results


def main():
    """Main function to run the enhanced training pipeline."""
    # Configuration
    dataset_path = "Datasetss"  # Relative to document-classifier directory
    output_dir = "enhanced_models_output"
    max_docs_per_class = None  # Set to a number for testing with smaller dataset
    random_state = 42

    # Initialize pipeline
    pipeline = EnhancedTrainingPipeline(
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_docs_per_class=max_docs_per_class,
        random_state=random_state
    )

    # Run pipeline
    results = pipeline.run_full_pipeline(
        test_size=0.2,
        val_size=0.2
    )

    # Print final summary
    print("\n" + "="*70)
    print("ENHANCED TRAINING PIPELINE COMPLETED")
    print("="*70)
    print(f"Dataset: {pipeline.dataset_info['n_classes']} classes")
    print(f"Total samples: {pipeline.dataset_info['total_samples']}")
    print(f"Train samples: {pipeline.dataset_info['train_samples']}")
    print(f"Validation samples: {pipeline.dataset_info['val_samples']}")
    print(f"Test samples: {pipeline.dataset_info['test_samples']}")

    print("\nFinal Test Results:")
    if 'test_results' in results:
        for model_name, metrics in results['test_results'].items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")

    print(f"\nResults saved to: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
