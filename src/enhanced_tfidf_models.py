"""Enhanced TF-IDF models for contract classification with advanced NLP techniques."""

import os
import logging
import pickle
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from pathlib import Path

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_selection import SelectKBest, chi2

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTFIDFClassifier:
    """Enhanced TF-IDF classifier with advanced NLP techniques."""

    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        """
        Initialize the enhanced TF-IDF classifier.

        Args:
            model_type: Type of classifier ('random_forest', 'svm', 'logistic', 'gradient_boosting')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.vectorizer = None
        self.feature_selector = None  # Add feature selector storage
        self.classifier = None
        self.is_trained = False
        self.class_names = None
        self.feature_names = None
        self.use_feature_selection = False  # Track if feature selection was used

    def _create_classifier(self, **kwargs):
        """Create the specified classifier with optimized parameters."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 300),
                max_depth=kwargs.get('max_depth', 25),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                class_weight=kwargs.get('class_weight', 'balanced'),
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "svm":
            return SVC(
                C=kwargs.get('C', 15.0),
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'scale'),
                class_weight=kwargs.get('class_weight', 'balanced'),
                probability=True,
                random_state=self.random_state
            )
        elif self.model_type == "logistic":
            return LogisticRegression(
                C=kwargs.get('C', 1.0),
                max_iter=kwargs.get('max_iter', 1000),
                class_weight=kwargs.get('class_weight', 'balanced'),
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 5),
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, texts: List[str], labels: List[int],
              class_names: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the enhanced TF-IDF classifier.

        Args:
            texts: List of contract texts
            labels: List of integer labels
            class_names: Names of the classes
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training results
        """
        self.class_names = class_names
        self.use_feature_selection = kwargs.get('use_feature_selection', True)

        # Create enhanced TF-IDF features
        logger.info("Creating enhanced TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=kwargs.get('max_features', 15000),
            ngram_range=kwargs.get('ngram_range', (1, 4)
                                   ),  # Extended to 4-grams
            min_df=kwargs.get('min_df', 2),
            max_df=kwargs.get('max_df', 0.9),
            stop_words=kwargs.get('stop_words', 'english'),
            sublinear_tf=True  # Apply sublinear tf scaling
        )

        # Create TF-IDF features
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        logger.info(f"TF-IDF features shape: {X.shape}")

        # Feature selection (optional)
        if self.use_feature_selection:
            logger.info("Applying feature selection...")
            self.feature_selector = SelectKBest(chi2, k=min(
                kwargs.get('k_best', 10000), X.shape[1]))
            X = self.feature_selector.fit_transform(X, labels)
            selected_features = self.feature_selector.get_support()
            self.feature_names = self.feature_names[selected_features]
            logger.info(f"Selected features shape: {X.shape}")

        # Create and train classifier
        logger.info(f"Training {self.model_type} classifier...")
        self.classifier = self._create_classifier(**kwargs)

        self.classifier.fit(X, labels)
        self.is_trained = True

        # Evaluate on training data
        train_preds = self.classifier.predict(X)
        train_accuracy = accuracy_score(labels, train_preds)

        logger.info(
            f"Training completed. Training accuracy: {train_accuracy:.4f}")

        return {
            'train_accuracy': train_accuracy,
            'features_shape': X.shape,
            'model_name': f'Enhanced TF-IDF + {self.model_type.replace("_", " ").title()}',
            'feature_names': self.feature_names.tolist()
        }

    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions on new texts."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X = self.vectorizer.transform(texts)
        if self.use_feature_selection and self.feature_selector:
            X = self.feature_selector.transform(X)
        return self.classifier.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X = self.vectorizer.transform(texts)
        if self.use_feature_selection and self.feature_selector:
            X = self.feature_selector.transform(X)
        return self.classifier.predict_proba(X)

    def evaluate(self, texts: List[str], y_true: List[int]) -> Dict[str, Any]:
        """Evaluate model performance."""
        y_pred = self.predict(texts)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError(
                "Model must be trained before getting feature importance")

        if hasattr(self.classifier, 'feature_importances_'):
            importance_dict = dict(
                zip(self.feature_names, self.classifier.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(self.classifier, 'coef_'):
            # For linear models, use absolute coefficients
            importance_dict = dict(
                zip(self.feature_names, np.abs(self.classifier.coef_[0])))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            logger.warning(
                "Feature importance not available for this classifier type")
            return {}

    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'vectorizer': self.vectorizer,
            'feature_selector': self.feature_selector,
            'classifier': self.classifier,
            'class_names': self.class_names,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'use_feature_selection': self.use_feature_selection
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data['vectorizer']
        self.feature_selector = model_data['feature_selector']
        self.classifier = model_data['classifier']
        self.class_names = model_data['class_names']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.use_feature_selection = model_data['use_feature_selection']
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class TopicModelingClassifier:
    """Classifier using topic modeling features for contract classification."""

    def __init__(self, n_topics: int = 50, random_state: int = 42):
        """
        Initialize the topic modeling classifier.

        Args:
            n_topics: Number of topics to extract
            random_state: Random seed for reproducibility
        """
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = None
        self.topic_model = None
        self.classifier = None
        self.is_trained = False
        self.class_names = None
        self.topic_names = None

    def train(self, texts: List[str], labels: List[int],
              class_names: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the topic modeling classifier.

        Args:
            texts: List of contract texts
            labels: List of integer labels
            class_names: Names of the classes
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training results
        """
        self.class_names = class_names

        # Create TF-IDF features for topic modeling
        logger.info("Creating TF-IDF features for topic modeling...")
        self.vectorizer = TfidfVectorizer(
            max_features=kwargs.get('max_features', 10000),
            ngram_range=kwargs.get('ngram_range', (1, 2)),
            min_df=kwargs.get('min_df', 3),
            max_df=kwargs.get('max_df', 0.8),
            stop_words='english'
        )

        X = self.vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF features shape: {X.shape}")

        # Apply topic modeling
        logger.info(f"Extracting {self.n_topics} topics...")
        self.topic_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=kwargs.get('max_iter', 20),
            learning_method='batch'
        )

        topic_features = self.topic_model.fit_transform(X)
        logger.info(f"Topic features shape: {topic_features.shape}")

        # Create topic names
        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_names = []
        for topic_idx, topic in enumerate(self.topic_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            self.topic_names.append(
                f"Topic_{topic_idx}: {' '.join(top_words[:5])}")

        # Train classifier on topic features
        logger.info("Training classifier on topic features...")
        self.classifier = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 20),
            random_state=self.random_state,
            n_jobs=-1
        )

        self.classifier.fit(topic_features, labels)
        self.is_trained = True

        # Evaluate on training data
        train_preds = self.classifier.predict(topic_features)
        train_accuracy = accuracy_score(labels, train_preds)

        logger.info(
            f"Training completed. Training accuracy: {train_accuracy:.4f}")

        return {
            'train_accuracy': train_accuracy,
            'features_shape': topic_features.shape,
            'model_name': f'Topic Modeling + Random Forest ({self.n_topics} topics)',
            'topic_names': self.topic_names
        }

    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions on new texts."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X = self.vectorizer.transform(texts)
        topic_features = self.topic_model.transform(X)
        return self.classifier.predict(topic_features)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X = self.vectorizer.transform(texts)
        topic_features = self.topic_model.transform(X)
        return self.classifier.predict_proba(topic_features)

    def evaluate(self, texts: List[str], y_true: List[int]) -> Dict[str, Any]:
        """Evaluate model performance."""
        y_pred = self.predict(texts)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

    def get_topics(self) -> List[str]:
        """Get the extracted topics."""
        return self.topic_names if self.topic_names else []

    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'vectorizer': self.vectorizer,
            'topic_model': self.topic_model,
            'classifier': self.classifier,
            'class_names': self.class_names,
            'topic_names': self.topic_names,
            'n_topics': self.n_topics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data['vectorizer']
        self.topic_model = model_data['topic_model']
        self.classifier = model_data['classifier']
        self.class_names = model_data['class_names']
        self.topic_names = model_data['topic_names']
        self.n_topics = model_data['n_topics']
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


def plot_model_comparison(results: Dict[str, Dict[str, float]], save_path: str = None):
    """Plot comparison of different model approaches."""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Enhanced Model Performance Comparison', fontsize=16)

    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        values = [results[model][metric] for model in models]

        bars = axes[row, col].bar(models, values, color=[
                                  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[row, col].set_title(f'{metric.capitalize()}')
        axes[row, col].set_ylabel(metric.capitalize())
        axes[row, col].set_ylim(0, 1)

        # Rotate x-axis labels for better readability
        axes[row, col].tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")

    plt.show()


def plot_feature_importance(classifier, top_n: int = 20, save_path: str = None):
    """Plot feature importance for a trained classifier."""
    if not classifier.is_trained:
        logger.warning(
            "Classifier must be trained before plotting feature importance")
        return

    importance = classifier.get_feature_importance()
    if not importance:
        logger.warning("Feature importance not available")
        return

    # Get top features
    top_features = list(importance.items())[:top_n]
    feature_names, importance_scores = zip(*top_features)

    # Create plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(feature_names))

    plt.barh(y_pos, importance_scores, color='skyblue', edgecolor='black')
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.show()
