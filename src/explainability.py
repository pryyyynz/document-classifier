"""LIME explainability for contract classification models."""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# LIME for explainability
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContractExplainer:
    """LIME-based explainability for contract classification."""

    def __init__(self, model, vectorizer, class_names: List[str], feature_selector=None, random_state: int = 42):
        """
        Initialize the explainer.

        Args:
            model: Trained classifier (already has predict_proba method)
            vectorizer: TF-IDF vectorizer
            class_names: List of contract class names
            feature_selector: Optional feature selector (SelectKBest, etc.)
            random_state: Random seed for reproducibility
        """
        if not LIME_AVAILABLE:
            raise ImportError(
                "LIME is not available. Install with: pip install lime")

        self.model = model
        self.vectorizer = vectorizer
        self.feature_selector = feature_selector
        self.class_names = class_names
        self.random_state = random_state
        self.explainer = LimeTextExplainer(
            class_names=class_names,
            random_state=random_state
        )

        logger.info(
            f"ContractExplainer initialized with {len(class_names)} classes: {class_names}")
        if feature_selector:
            logger.info(f"Feature selector enabled: {type(feature_selector).__name__}")

    def explain_prediction(self, text: str, num_features: int = 15,
                           num_samples: int = 500) -> Dict[str, Any]:
        """
        Explain why the model made a specific prediction.

        Args:
            text: Contract text to explain
            num_features: Number of top features to show
            num_samples: Number of LIME samples for explanation

        Returns:
            Dictionary with explanation details
        """
        try:
            # Wrapper function for LIME compatibility
            def predict_proba_wrapper(texts):
                features = self.vectorizer.transform(texts)
                if self.feature_selector:
                    features = self.feature_selector.transform(features)
                return self.model.predict_proba(features)

            # Generate LIME explanation
            exp = self.explainer.explain_instance(
                text,
                predict_proba_wrapper,
                num_features=num_features,
                num_samples=num_samples,
                top_labels=1  # Focus on predicted class
            )

            # Get prediction details
            all_probs = predict_proba_wrapper([text])[0]
            predicted_class_idx = np.argmax(all_probs)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = all_probs[predicted_class_idx]

            # Extract important features for the predicted class
            important_features = exp.as_list(label=predicted_class_idx)

            # Process important features to get single best phrase
            processed_features = self._get_best_phrase_feature(important_features, text)
            
            # Create explanation summary
            explanation = {
                'text': text[:200] + "..." if len(text) > 200 else text,
                'full_text': text,
                'prediction': predicted_class,
                'confidence': confidence,
                'important_features': processed_features,
                'explanation_html': exp.as_html(),
                'num_features': num_features,
                'success': True,
                'explanation_object': exp  # Keep for advanced usage
            }

            logger.info(
                f"Successfully explained prediction: {predicted_class} (confidence: {confidence:.3f})")
            return explanation

        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': text[:200] + "..." if len(text) > 200 else text,
                'full_text': text
            }

    def explain_multiple(self, texts: List[str], num_features: int = 10,
                         num_samples: int = 5000) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions at once.

        Args:
            texts: List of contract texts
            num_features: Number of top features to show
            num_samples: Number of LIME samples for explanation

        Returns:
            List of explanation dictionaries
        """
        explanations = []
        total_texts = len(texts)

        logger.info(f"Explaining {total_texts} texts...")

        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{total_texts}")
            explanation = self.explain_prediction(
                text, num_features, num_samples)
            explanations.append(explanation)

        logger.info(f"Completed explanations for {total_texts} texts")
        return explanations

    def get_feature_importance_summary(self, explanations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate feature importance across multiple explanations.

        Args:
            explanations: List of explanation dictionaries

        Returns:
            Dictionary of aggregated feature importance scores
        """
        feature_scores = {}
        successful_explanations = [
            exp for exp in explanations if exp['success']]

        if not successful_explanations:
            logger.warning("No successful explanations to aggregate")
            return {}

        for exp in successful_explanations:
            for feature, score in exp['important_features']:
                if feature in feature_scores:
                    feature_scores[feature] += abs(score)
                else:
                    feature_scores[feature] = abs(score)

        # Sort by total importance
        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = dict(sorted_features[:20])  # Top 20 features

        logger.info(
            f"Aggregated feature importance from {len(successful_explanations)} explanations")
        return top_features

    def visualize_explanation(self, explanation: Dict[str, Any],
                              save_path: Optional[str] = None) -> None:
        """
        Visualize the explanation results.

        Args:
            explanation: Explanation dictionary from explain_prediction
            save_path: Optional path to save the visualization
        """
        if not explanation['success']:
            logger.warning("Cannot visualize failed explanation")
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Class probabilities
        classes = list(explanation['class_probabilities'].keys())
        probs = list(explanation['class_probabilities'].values())
        colors = ['red' if c == explanation['prediction']
                  else 'lightblue' for c in classes]

        bars = ax1.bar(classes, probs, color=colors, alpha=0.7)
        ax1.set_title(
            f'Class Probabilities\nPrediction: {explanation["prediction"]} ({explanation["confidence"]:.3f})')
        ax1.set_ylabel('Probability')
        ax1.set_ylim(0, 1)

        # Rotate x-axis labels for better readability
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{prob:.3f}', ha='center', va='bottom')

        # Plot 2: Feature importance
        features = [f[0] for f in explanation['important_features'][:10]]
        scores = [f[1] for f in explanation['important_features'][:10]]
        colors = ['red' if s > 0 else 'blue' for s in scores]

        bars = ax2.barh(features, scores, color=colors, alpha=0.7)
        ax2.set_title('Top Influential Features')
        ax2.set_xlabel('Feature Importance Score')

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax2.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2.,
                     f'{score:.3f}', ha='left' if width > 0 else 'right', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")

        plt.show()

    def generate_explanation_report(self, explanations: List[Dict[str, Any]],
                                    output_dir: str = "explanation_reports") -> str:
        """
        Generate a comprehensive explanation report.

        Args:
            explanations: List of explanation dictionaries
            output_dir: Directory to save the report

        Returns:
            Path to the generated report
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Filter successful explanations
        successful_explanations = [
            exp for exp in explanations if exp['success']]

        if not successful_explanations:
            logger.warning("No successful explanations to report")
            return ""

        # Create summary statistics
        predictions = [exp['prediction'] for exp in successful_explanations]
        confidences = [exp['confidence'] for exp in successful_explanations]

        # Count predictions by class
        class_counts = pd.Series(predictions).value_counts()

        # Create summary DataFrame
        summary_data = []
        for exp in successful_explanations:
            summary_data.append({
                'prediction': exp['prediction'],
                'confidence': exp['confidence'],
                'text_length': len(exp['full_text']),
                'top_feature': exp['important_features'][0][0] if exp['important_features'] else '',
                'top_feature_score': exp['important_features'][0][1] if exp['important_features'] else 0
            })

        summary_df = pd.DataFrame(summary_data)

        # Generate report
        report_path = os.path.join(
            output_dir, f"explanation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html")

        html_content = f"""
        <html>
        <head>
            <title>Contract Classification Explanation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .high-confidence {{ color: green; }}
                .medium-confidence {{ color: orange; }}
                .low-confidence {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Contract Classification Explanation Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total explanations: {len(successful_explanations)}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <div class="metric">
                    <strong>Average Confidence:</strong> {np.mean(confidences):.3f}
                </div>
                <div class="metric">
                    <strong>Min Confidence:</strong> {np.min(confidences):.3f}
                </div>
                <div class="metric">
                    <strong>Max Confidence:</strong> {np.max(confidences):.3f}
                </div>
            </div>
            
            <div class="section">
                <h2>Predictions by Class</h2>
                {class_counts.to_frame('Count').to_html()}
            </div>
            
            <div class="section">
                <h2>Detailed Explanations</h2>
                <table>
                    <tr>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Text Length</th>
                        <th>Top Feature</th>
                        <th>Top Feature Score</th>
                    </tr>
        """

        for exp in successful_explanations:
            confidence_class = "high-confidence" if exp['confidence'] > 0.8 else "medium-confidence" if exp['confidence'] > 0.6 else "low-confidence"
            html_content += f"""
                    <tr>
                        <td>{exp['prediction']}</td>
                        <td class="{confidence_class}">{exp['confidence']:.3f}</td>
                        <td>{exp['text_length']}</td>
                        <td>{exp['important_features'][0][0] if exp['important_features'] else ''}</td>
                        <td>{exp['important_features'][0][1]:.3f if exp['important_features'] else 0}</td>
                    </tr>
            """

        html_content += """
                </table>
            </div>
        </body>
        </html>
        """

        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Explanation report saved to {report_path}")
        return report_path

    def save_explanation(self, explanation: Dict[str, Any], filepath: str) -> None:
        """
        Save explanation to a file.

        Args:
            explanation: Explanation dictionary
            filepath: Path to save the explanation
        """
        # Remove the explanation object before saving (not serializable)
        save_data = explanation.copy()
        if 'explanation_object' in save_data:
            del save_data['explanation_object']

        # Save as pickle
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Explanation saved to {filepath}")

    def load_explanation(self, filepath: str) -> Dict[str, Any]:
        """
        Load explanation from a file.

        Args:
            filepath: Path to load the explanation from

        Returns:
            Loaded explanation dictionary
        """
        import pickle
        with open(filepath, 'rb') as f:
            explanation = pickle.load(f)

        logger.info(f"Explanation loaded from {filepath}")
        return explanation

    def _get_best_phrase_feature(self, important_features: List[Tuple[str, float]], text: str) -> List[Tuple[str, float]]:
        """
        Get the single best phrase feature with 3+ words and highest LIME score.
        
        Args:
            important_features: List of (feature, score) tuples from LIME
            text: Original text being explained
            
        Returns:
            List containing single (phrase, score) tuple with best phrase
        """
        text_lower = text.lower()
        candidate_phrases = []
        
        for feature, score in important_features:
            # If feature is already a phrase with 3+ words, add it as candidate
            if ' ' in feature and len(feature.split()) >= 3:
                candidate_phrases.append((feature, abs(score)))
            else:
                # Find the feature word in context and extract surrounding words
                feature_lower = feature.lower()
                words = text_lower.split()
                
                # Find occurrences of the feature word
                for i, word in enumerate(words):
                    if feature_lower in word.lower():
                        # Extract larger context: 2-3 words before and after
                        start_idx = max(0, i - 2)
                        end_idx = min(len(words), i + 4)
                        context_phrase = ' '.join(words[start_idx:end_idx])
                        
                        # Clean up the phrase
                        context_phrase = context_phrase.strip('.,!?;:"()[]{}')
                        
                        # Only keep phrases with 3+ words
                        if len(context_phrase.split()) >= 3:
                            candidate_phrases.append((context_phrase, abs(score)))
                            break
        
        # Return the phrase with highest absolute LIME score, or empty if none found
        if candidate_phrases:
            best_phrase = max(candidate_phrases, key=lambda x: x[1])
            return [best_phrase]
        else:
            # Fallback: return first feature if no good phrases found
            return [important_features[0]] if important_features else []


def create_explainer_from_model(model_path: str) -> ContractExplainer:
    """
    Create a ContractExplainer from a saved model.

    Args:
        model_path: Path to the saved model file

    Returns:
        Initialized ContractExplainer
    """
    import pickle

    # Load the model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Extract components
    classifier = model_data['classifier']
    vectorizer = model_data['vectorizer']
    class_names = model_data['class_names']
    feature_selector = model_data.get('feature_selector', None)

    # Create explainer
    explainer = ContractExplainer(classifier, vectorizer, class_names, feature_selector)

    logger.info(f"Explainer created from model: {model_path}")
    return explainer


# Example usage function
def demonstrate_explainability():
    """Demonstrate the explainability functionality."""
    try:
        # Try to load the best model
        model_path = "enhanced_models_output/models/enhanced_tfidf_gradient_boosting_model.pkl"

        if os.path.exists(model_path):
            explainer = create_explainer_from_model(model_path)

            # Example text
            test_text = """
            This consulting services agreement is made between ABC Corporation and XYZ Consulting LLC.
            ABC Corporation hereby engages XYZ Consulting to provide strategic consulting services
            including market analysis, business strategy development, and operational optimization.
            The term of this agreement shall be twelve months from the effective date.
            """

            # Get explanation
            explanation = explainer.explain_prediction(test_text)

            if explanation['success']:
                print("🎯 Explanation Demo:")
                print(f"Prediction: {explanation['prediction']}")
                print(f"Confidence: {explanation['confidence']:.3f}")
                print("\n🔍 Top influential features:")
                for feature, score in explanation['important_features'][:5]:
                    print(f"  • '{feature}': {score:.3f}")

                # Visualize
                explainer.visualize_explanation(explanation)

            else:
                print(
                    f"❌ Explanation failed: {explanation.get('error', 'Unknown error')}")

        else:
            print(f"❌ Model not found at {model_path}")
            print("Please train the model first or check the path.")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure LIME is installed: pip install lime")


if __name__ == "__main__":
    demonstrate_explainability()
