"""Machine learning models for contract classification."""

import os
import pickle
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContractClassifier:
    """Base class for contract classification models."""

    def __init__(self, model_name: str, random_state: int = 42):
        """
        Initialize the classifier.

        Args:
            model_name: Name of the model
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.class_names = None
        self.feature_importance = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training results
        """
        raise NotImplementedError("Subclasses must implement train method")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Input features

        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(
                f"{self.model_name} does not support probability predictions")

    def evaluate(self, X: np.ndarray, y_true: np.ndarray,
                 class_names: Optional[list] = None) -> Dict[str, Any]:
        """
        Evaluate the model performance.

        Args:
            X: Test features
            y_true: True labels
            class_names: Names of the classes

        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X) if hasattr(
            self.model, 'predict_proba') else None

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        # Calculate per-class metrics
        per_class_metrics = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        # Calculate ROC AUC if possible
        roc_auc = None
        if y_proba is not None and len(np.unique(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        elif y_proba is not None and len(np.unique(y_true)) > 2:
            roc_auc = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='weighted')

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Prepare results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
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

        # Generate classification report
        if class_names:
            results['classification_report'] = classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True
            )
        else:
            results['classification_report'] = classification_report(
                y_true, y_pred, output_dict=True
            )

        return results

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'class_names': self.class_names,
            'feature_importance': self.feature_importance,
            'random_state': self.random_state
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self.class_names = model_data['class_names']
        self.feature_importance = model_data['feature_importance']
        self.random_state = model_data['random_state']

        logger.info(f"Model loaded from {filepath}")


class RandomForestContractClassifier(ContractClassifier):
    """Random Forest classifier for contract classification."""

    def __init__(self, random_state: int = 42, **kwargs):
        """
        Initialize Random Forest classifier.

        Args:
            random_state: Random seed for reproducibility
            **kwargs: Additional Random Forest parameters
        """
        super().__init__("Random Forest", random_state)

        # Default parameters optimized for text classification
        default_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'n_jobs': -1,
            'random_state': random_state,
            'class_weight': 'balanced'
        }

        # Update with any provided parameters
        default_params.update(kwargs)

        self.model = RandomForestClassifier(**default_params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training results
        """
        logger.info(f"Training {self.model_name}...")

        # Store class names
        self.class_names = list(np.unique(y_train))

        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

        # Calculate training accuracy
        train_accuracy = self.model.score(X_train, y_train)

        results = {
            'train_accuracy': train_accuracy,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth
        }

        # Calculate validation accuracy if validation data is provided
        if X_val is not None and y_val is not None:
            val_accuracy = self.model.score(X_val, y_val)
            results['val_accuracy'] = val_accuracy
            logger.info(f"Training accuracy: {train_accuracy:.4f}")
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        else:
            logger.info(f"Training accuracy: {train_accuracy:.4f}")

        # OOB score if available
        if hasattr(self.model, 'oob_score_'):
            results['oob_score'] = self.model.oob_score_
            logger.info(f"OOB score: {self.model.oob_score_:.4f}")

        return results

    def get_feature_importance(self, feature_names: Optional[list] = None,
                               top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            feature_names: Names of the features
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_trained or self.feature_importance is None:
            raise ValueError("Model must be trained to get feature importance")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(
                len(self.feature_importance))]

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        })

        # Sort by importance and get top N
        importance_df = importance_df.sort_values(
            'importance', ascending=False).head(top_n)

        return importance_df


class SVMContractClassifier(ContractClassifier):
    """Support Vector Machine classifier for contract classification."""

    def __init__(self, random_state: int = 42, **kwargs):
        """
        Initialize SVM classifier.

        Args:
            random_state: Random seed for reproducibility
            **kwargs: Additional SVM parameters
        """
        super().__init__("SVM", random_state)

        # Default parameters optimized for text classification
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': random_state,
            'class_weight': 'balanced'
        }

        # Update with any provided parameters
        default_params.update(kwargs)

        self.model = SVC(**default_params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the SVM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training results
        """
        logger.info(f"Training {self.model_name}...")

        # Store class names
        self.class_names = list(np.unique(y_train))

        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Calculate training accuracy
        train_accuracy = self.model.score(X_train, y_train)

        results = {
            'train_accuracy': train_accuracy,
            'C': self.model.C,
            'kernel': self.model.kernel,
            'gamma': self.model.gamma
        }

        # Calculate validation accuracy if validation data is provided
        if X_val is not None and y_val is not None:
            val_accuracy = self.model.score(X_val, y_val)
            results['val_accuracy'] = val_accuracy
            logger.info(f"Training accuracy: {train_accuracy:.4f}")
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        else:
            logger.info(f"Training accuracy: {train_accuracy:.4f}")

        return results


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                          title: str = "Confusion Matrix",
                          save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Names of the classes
        title: Title of the plot
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")

    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                            title: str = "Feature Importance",
                            save_path: Optional[str] = None) -> None:
    """
    Plot feature importance.

    Args:
        importance_df: DataFrame with feature importance
        title: Title of the plot
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.show()
