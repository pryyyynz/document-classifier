"""Main training pipeline for contract classification models."""

import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from data_loader import ContractDataLoader
from models import (
    RandomForestContractClassifier,
    SVMContractClassifier,
    plot_confusion_matrix,
    plot_feature_importance
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContractTrainingPipeline:
    """Main training pipeline for contract classification."""

    def __init__(self, dataset_path: str, output_dir: str = "models",
                 max_docs_per_class: Optional[int] = None, random_state: int = 42):
        """
        Initialize the training pipeline.

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

        # Initialize data loader
        self.data_loader = ContractDataLoader(dataset_path, max_docs_per_class)

        # Store results
        self.results = {}
        self.dataset_info = {}

    def prepare_data(self, test_size: float = 0.2, val_size: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Prepare the dataset with train/validation/test splits.

        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            Dictionary containing the data splits
        """
        logger.info("Preparing dataset...")

        # Prepare dataset
        dataset = self.data_loader.prepare_dataset(
            test_size=test_size,
            val_size=val_size,
            random_state=self.random_state
        )

        # Store dataset info
        self.dataset_info = {
            'n_classes': len(self.data_loader.get_class_names()),
            'class_names': self.data_loader.get_class_names(),
            'feature_names': self.data_loader.get_feature_names(),
            'train_samples': dataset['X_train'].shape[0],
            'val_samples': dataset['X_val'].shape[0],
            'test_samples': dataset['X_test'].shape[0],
            'n_features': dataset['X_train'].shape[1]
        }

        logger.info(f"Dataset prepared successfully:")
        logger.info(f"  Classes: {self.dataset_info['n_classes']}")
        logger.info(f"  Features: {self.dataset_info['n_features']}")
        logger.info(f"  Train samples: {self.dataset_info['train_samples']}")
        logger.info(
            f"  Validation samples: {self.dataset_info['val_samples']}")
        logger.info(f"  Test samples: {self.dataset_info['test_samples']}")

        return dataset

    def train_random_forest(self, dataset: Dict[str, np.ndarray],
                            **rf_params) -> RandomForestContractClassifier:
        """
        Train Random Forest model.

        Args:
            dataset: Dataset dictionary with train/val/test splits
            **rf_params: Additional Random Forest parameters

        Returns:
            Trained Random Forest classifier
        """
        logger.info("Training Random Forest model...")

        # Initialize model
        rf_model = RandomForestContractClassifier(
            random_state=self.random_state,
            **rf_params
        )

        # Train model
        train_results = rf_model.train(
            dataset['X_train'], dataset['y_train'],
            dataset['X_val'], dataset['y_val']
        )

        # Evaluate on validation set
        val_results = rf_model.evaluate(
            dataset['X_val'], dataset['y_val'],
            self.dataset_info['class_names']
        )

        # Store results
        self.results['random_forest'] = {
            'training': train_results,
            'validation': val_results
        }

        logger.info(f"Random Forest training completed:")
        logger.info(
            f"  Training accuracy: {train_results['train_accuracy']:.4f}")
        logger.info(
            f"  Validation accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"  Validation F1: {val_results['f1_score']:.4f}")

        return rf_model

    def train_svm(self, dataset: Dict[str, np.ndarray],
                  **svm_params) -> SVMContractClassifier:
        """
        Train SVM model.

        Args:
            dataset: Dataset dictionary with train/val/test splits
            **svm_params: Additional SVM parameters

        Returns:
            Trained SVM classifier
        """
        logger.info("Training SVM model...")

        # Initialize model
        svm_model = SVMContractClassifier(
            random_state=self.random_state,
            **svm_params
        )

        # Train model
        train_results = svm_model.train(
            dataset['X_train'], dataset['y_train'],
            dataset['X_val'], dataset['y_val']
        )

        # Evaluate on validation set
        val_results = svm_model.evaluate(
            dataset['X_val'], dataset['y_val'],
            self.dataset_info['class_names']
        )

        # Store results
        self.results['svm'] = {
            'training': train_results,
            'validation': val_results
        }

        logger.info(f"SVM training completed:")
        logger.info(
            f"  Training accuracy: {train_results['train_accuracy']:.4f}")
        logger.info(
            f"  Validation accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"  Validation F1: {val_results['f1_score']:.4f}")

        return svm_model

    def evaluate_models(self, rf_model: RandomForestContractClassifier,
                        svm_model: SVMContractClassifier,
                        dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate both models on the test set.

        Args:
            rf_model: Trained Random Forest model
            svm_model: Trained SVM model
            dataset: Dataset dictionary with train/val/test splits

        Returns:
            Dictionary containing test results for both models
        """
        logger.info("Evaluating models on test set...")

        test_results = {}

        # Evaluate Random Forest
        rf_test_results = rf_model.evaluate(
            dataset['X_test'], dataset['y_test'],
            self.dataset_info['class_names']
        )
        test_results['random_forest'] = rf_test_results

        # Evaluate SVM
        svm_test_results = svm_model.evaluate(
            dataset['X_test'], dataset['y_test'],
            self.dataset_info['class_names']
        )
        test_results['svm'] = svm_test_results

        # Store test results
        self.results['test'] = test_results

        # Log results
        logger.info("Test set results:")
        logger.info(
            f"  Random Forest - Accuracy: {rf_test_results['accuracy']:.4f}, F1: {rf_test_results['f1_score']:.4f}")
        logger.info(
            f"  SVM - Accuracy: {svm_test_results['accuracy']:.4f}, F1: {svm_test_results['f1_score']:.4f}")

        return test_results

    def generate_plots(self, rf_model: RandomForestContractClassifier,
                       svm_model: SVMContractClassifier,
                       dataset: Dict[str, np.ndarray]) -> None:
        """
        Generate evaluation plots.

        Args:
            rf_model: Trained Random Forest model
            svm_model: Trained SVM model
            dataset: Dataset dictionary with train/val/test splits
        """
        logger.info("Generating evaluation plots...")

        # Plot confusion matrices
        rf_test_results = self.results['test']['random_forest']
        svm_test_results = self.results['test']['svm']

        # Random Forest confusion matrix
        plot_confusion_matrix(
            rf_test_results['confusion_matrix'],
            self.dataset_info['class_names'],
            title="Random Forest - Test Set Confusion Matrix",
            save_path=str(self.output_dir / "plots" /
                          "rf_confusion_matrix.png")
        )

        # SVM confusion matrix
        plot_confusion_matrix(
            svm_test_results['confusion_matrix'],
            self.dataset_info['class_names'],
            title="SVM - Test Set Confusion Matrix",
            save_path=str(self.output_dir / "plots" /
                          "svm_confusion_matrix.png")
        )

        # Feature importance for Random Forest
        if rf_model.feature_importance is not None:
            feature_names = self.data_loader.get_feature_names()
            importance_df = rf_model.get_feature_importance(
                feature_names, top_n=30)

            plot_feature_importance(
                importance_df,
                title="Random Forest - Top 30 Feature Importance",
                save_path=str(self.output_dir / "plots" /
                              "rf_feature_importance.png")
            )

    def save_models(self, rf_model: RandomForestContractClassifier,
                    svm_model: SVMContractClassifier) -> None:
        """
        Save trained models to disk.

        Args:
            rf_model: Trained Random Forest model
            svm_model: Trained SVM model
        """
        logger.info("Saving trained models...")

        # Save Random Forest
        rf_path = self.output_dir / "models" / "random_forest_model.pkl"
        rf_model.save_model(str(rf_path))

        # Save SVM
        svm_path = self.output_dir / "models" / "svm_model.pkl"
        svm_model.save_model(str(svm_path))

        # Save TF-IDF vectorizer
        vectorizer_path = self.output_dir / "models" / "tfidf_vectorizer.pkl"
        import pickle
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.data_loader.tfidf_vectorizer, f)

        # Save label encoder
        encoder_path = self.output_dir / "models" / "label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.data_loader.label_encoder, f)

        logger.info("Models saved successfully")

    def save_results(self) -> None:
        """Save training results and metadata to disk."""
        logger.info("Saving training results...")

        # Prepare results for saving
        save_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': self.dataset_info,
            'results': self.results,
            'random_state': self.random_state
        }

        # Save as JSON
        results_path = self.output_dir / "results" / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)

        # Save summary as CSV
        summary_data = []
        for model_name, results in self.results.items():
            if model_name != 'test':
                summary_data.append({
                    'model': model_name,
                    'train_accuracy': results['training'].get('train_accuracy', None),
                    'val_accuracy': results['training'].get('val_accuracy', None),
                    'val_f1': results['validation']['f1_score'],
                    'val_precision': results['validation']['precision'],
                    'val_recall': results['validation']['recall']
                })

        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "results" / "model_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        logger.info("Results saved successfully")

    def run_full_pipeline(self, test_size: float = 0.2, val_size: float = 0.2,
                          rf_params: Optional[Dict[str, Any]] = None,
                          svm_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set
            rf_params: Additional Random Forest parameters
            svm_params: Additional SVM parameters

        Returns:
            Dictionary containing all results
        """
        logger.info("Starting full training pipeline...")

        # Prepare data
        dataset = self.prepare_data(test_size, val_size)

        # Train models
        rf_model = self.train_random_forest(dataset, **(rf_params or {}))
        svm_model = self.train_svm(dataset, **(svm_params or {}))

        # Evaluate on test set
        test_results = self.evaluate_models(rf_model, svm_model, dataset)

        # Generate plots
        self.generate_plots(rf_model, svm_model, dataset)

        # Save models and results
        self.save_models(rf_model, svm_model)
        self.save_results()

        logger.info("Training pipeline completed successfully!")

        return self.results


def main():
    """Main function to run the training pipeline."""
    # Configuration
    dataset_path = "Datasetss"  # Relative to document-classifier directory
    output_dir = "models_output"
    max_docs_per_class = None  # Set to a number for testing with smaller dataset
    random_state = 42

    # Initialize pipeline
    pipeline = ContractTrainingPipeline(
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_docs_per_class=max_docs_per_class,
        random_state=random_state
    )

    # Optional: Customize model parameters
    rf_params = {
        'n_estimators': 300,
        'max_depth': 25,
        'min_samples_split': 3
    }

    svm_params = {
        'C': 10.0,
        'gamma': 'auto'
    }

    # Run pipeline
    results = pipeline.run_full_pipeline(
        test_size=0.2,
        val_size=0.2,
        rf_params=rf_params,
        svm_params=svm_params
    )

    # Print final summary
    print("\n" + "="*50)
    print("TRAINING PIPELINE COMPLETED")
    print("="*50)
    print(
        f"Dataset: {pipeline.dataset_info['n_classes']} classes, {pipeline.dataset_info['n_features']} features")
    print(f"Train samples: {pipeline.dataset_info['train_samples']}")
    print(f"Validation samples: {pipeline.dataset_info['val_samples']}")
    print(f"Test samples: {pipeline.dataset_info['test_samples']}")
    print("\nFinal Test Results:")
    print(
        f"Random Forest - Accuracy: {results['test']['random_forest']['accuracy']:.4f}")
    print(f"SVM - Accuracy: {results['test']['svm']['accuracy']:.4f}")
    print(f"\nResults saved to: {output_dir}/")
    print("="*50)


if __name__ == "__main__":
    main()
