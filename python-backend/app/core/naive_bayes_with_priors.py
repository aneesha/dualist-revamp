"""
Naive Bayes classifier with feature priors for active learning.

This implementation supports:
- Multinomial Naive Bayes classification
- Feature prior incorporation via pseudocounts
- Incremental learning (add instances/features on-the-fly)
- Label and feature probability estimation

Based on the original Java implementation in NaiveBayesWithPriorsTrainer.java
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NaiveBayesWithPriors:
    """
    Multinomial Naive Bayes classifier with support for feature priors.

    The model combines:
    1. Standard NB trained on labeled instances
    2. Feature priors from expert-labeled features (via pseudocounts)
    """

    def __init__(self, alpha: float = 100.0, smoothing: float = 1.0):
        """
        Initialize the classifier.

        Args:
            alpha: Weight for feature priors (pseudocount strength)
            smoothing: Laplace smoothing parameter (default 1.0)
        """
        self.alpha = alpha
        self.smoothing = smoothing

        # Vocabularies
        self.feature_vocab: Dict[str, int] = {}  # feature -> index
        self.label_vocab: Dict[str, int] = {}    # label -> index
        self.idx_to_feature: Dict[int, str] = {}
        self.idx_to_label: Dict[int, str] = {}

        # Model parameters
        self.feature_counts: np.ndarray = None  # [n_labels, n_features]
        self.label_counts: np.ndarray = None    # [n_labels]
        self.feature_priors: Dict[Tuple[int, int], float] = {}  # (label_idx, feature_idx) -> pseudocount

        # Training data
        self.n_features = 0
        self.n_labels = 0

    def _get_or_create_feature_idx(self, feature: str) -> int:
        """Get or create index for a feature."""
        if feature not in self.feature_vocab:
            idx = len(self.feature_vocab)
            self.feature_vocab[feature] = idx
            self.idx_to_feature[idx] = feature
            self.n_features = len(self.feature_vocab)
        return self.feature_vocab[feature]

    def _get_or_create_label_idx(self, label: str) -> int:
        """Get or create index for a label."""
        if label not in self.label_vocab:
            idx = len(self.label_vocab)
            self.label_vocab[label] = idx
            self.idx_to_label[idx] = label
            self.n_labels = len(self.label_vocab)
        return self.label_vocab[label]

    def add_label_feature(self, label: str, feature: str, weight: Optional[float] = None):
        """
        Add a feature prior (expert labeling).

        Args:
            label: The label this feature is associated with
            feature: The feature term
            weight: Optional custom weight (defaults to self.alpha)
        """
        label_idx = self._get_or_create_label_idx(label)
        feature_idx = self._get_or_create_feature_idx(feature)

        if weight is None:
            weight = self.alpha

        self.feature_priors[(label_idx, feature_idx)] = weight
        logger.info(f"Added feature prior: '{feature}' -> '{label}' (weight={weight})")

    def train(self, X: List[Dict[str, int]], y: List[str]):
        """
        Train the classifier on labeled instances.

        Args:
            X: List of feature dictionaries (feature -> count)
            y: List of labels
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        # Build vocabularies
        for features in X:
            for feature in features:
                self._get_or_create_feature_idx(feature)

        for label in y:
            self._get_or_create_label_idx(label)

        # Initialize count matrices
        self.feature_counts = np.zeros((self.n_labels, self.n_features))
        self.label_counts = np.zeros(self.n_labels)

        # Count features and labels
        for features, label in zip(X, y):
            label_idx = self.label_vocab[label]
            self.label_counts[label_idx] += 1

            for feature, count in features.items():
                feature_idx = self.feature_vocab[feature]
                self.feature_counts[label_idx, feature_idx] += count

        logger.info(f"Trained on {len(X)} instances, {self.n_labels} labels, {self.n_features} features")

    def train_incremental(self, X: List[Dict[str, int]], y: List[str]):
        """
        Incrementally add labeled instances to the model.

        Args:
            X: List of feature dictionaries
            y: List of labels
        """
        # Expand matrices if needed
        old_n_features = self.n_features
        old_n_labels = self.n_labels

        for features in X:
            for feature in features:
                self._get_or_create_feature_idx(feature)

        for label in y:
            self._get_or_create_label_idx(label)

        # Resize matrices if vocabulary expanded
        if self.feature_counts is None:
            self.feature_counts = np.zeros((self.n_labels, self.n_features))
            self.label_counts = np.zeros(self.n_labels)
        else:
            if self.n_features > old_n_features or self.n_labels > old_n_labels:
                new_feature_counts = np.zeros((self.n_labels, self.n_features))
                new_feature_counts[:old_n_labels, :old_n_features] = self.feature_counts
                self.feature_counts = new_feature_counts

                if self.n_labels > old_n_labels:
                    new_label_counts = np.zeros(self.n_labels)
                    new_label_counts[:old_n_labels] = self.label_counts
                    self.label_counts = new_label_counts

        # Add new counts
        for features, label in zip(X, y):
            label_idx = self.label_vocab[label]
            self.label_counts[label_idx] += 1

            for feature, count in features.items():
                feature_idx = self.feature_vocab[feature]
                self.feature_counts[label_idx, feature_idx] += count

        logger.info(f"Incrementally added {len(X)} instances")

    def _estimate_log_probs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate log probabilities with smoothing and feature priors.

        Returns:
            log_label_probs: Log P(label)
            log_feature_probs: Log P(feature|label) [n_labels, n_features]
        """
        # Label probabilities with smoothing
        label_total = self.label_counts.sum() + self.n_labels * self.smoothing
        log_label_probs = np.log((self.label_counts + self.smoothing) / label_total)

        # Feature probabilities with smoothing and priors
        feature_counts_with_priors = self.feature_counts.copy()

        # Add feature priors as pseudocounts
        for (label_idx, feature_idx), weight in self.feature_priors.items():
            if label_idx < self.n_labels and feature_idx < self.n_features:
                feature_counts_with_priors[label_idx, feature_idx] += weight

        # Compute log probabilities
        feature_totals = feature_counts_with_priors.sum(axis=1) + self.n_features * self.smoothing
        log_feature_probs = np.log(
            (feature_counts_with_priors + self.smoothing) / feature_totals[:, np.newaxis]
        )

        return log_label_probs, log_feature_probs

    def predict_proba(self, X: List[Dict[str, int]]) -> np.ndarray:
        """
        Predict class probabilities for instances.

        Args:
            X: List of feature dictionaries

        Returns:
            Array of shape [n_instances, n_labels] with probabilities
        """
        if self.feature_counts is None:
            raise ValueError("Model not trained yet")

        log_label_probs, log_feature_probs = self._estimate_log_probs()

        n_instances = len(X)
        log_probs = np.zeros((n_instances, self.n_labels))

        for i, features in enumerate(X):
            # Start with label prior
            log_probs[i] = log_label_probs.copy()

            # Add log P(feature|label) for each feature
            for feature, count in features.items():
                if feature in self.feature_vocab:
                    feature_idx = self.feature_vocab[feature]
                    log_probs[i] += count * log_feature_probs[:, feature_idx]

        # Convert to probabilities (normalize)
        # Subtract max for numerical stability
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X: List[Dict[str, int]]) -> List[str]:
        """
        Predict class labels for instances.

        Args:
            X: List of feature dictionaries

        Returns:
            List of predicted labels
        """
        probs = self.predict_proba(X)
        label_indices = probs.argmax(axis=1)
        return [self.idx_to_label[idx] for idx in label_indices]

    def get_feature_importance(self, label: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get the most important features for a given label.

        Args:
            label: The label to analyze
            top_k: Number of top features to return

        Returns:
            List of (feature, log_prob) tuples
        """
        if label not in self.label_vocab:
            return []

        label_idx = self.label_vocab[label]
        _, log_feature_probs = self._estimate_log_probs()

        # Get features for this label
        feature_log_probs = log_feature_probs[label_idx]
        top_indices = np.argsort(feature_log_probs)[-top_k:][::-1]

        return [
            (self.idx_to_feature[idx], feature_log_probs[idx])
            for idx in top_indices
        ]

    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'NaiveBayesWithPriors':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model

    def get_stats(self) -> Dict:
        """Get model statistics."""
        return {
            "n_labels": self.n_labels,
            "n_features": self.n_features,
            "n_training_instances": int(self.label_counts.sum()) if self.label_counts is not None else 0,
            "n_feature_priors": len(self.feature_priors),
            "labels": list(self.label_vocab.keys()),
            "alpha": self.alpha,
            "smoothing": self.smoothing
        }
