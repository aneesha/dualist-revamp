"""
Active learning query strategies for instance and feature selection.

Implements:
- Instance querying: entropy, margin, least confident
- Feature querying: per-label mutual information (MI)

Based on the original Java implementation in Queries.java
"""

import numpy as np
from typing import List, Dict, Tuple, Set
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class ActiveLearningQueries:
    """
    Query selection strategies for active learning.
    """

    @staticmethod
    def entropy(probs: np.ndarray) -> float:
        """
        Calculate entropy of a probability distribution.

        Args:
            probs: Array of probabilities

        Returns:
            Entropy value
        """
        # Avoid log(0)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    @staticmethod
    def margin(probs: np.ndarray) -> float:
        """
        Calculate margin between top two predictions.

        Args:
            probs: Array of probabilities

        Returns:
            Margin (difference between top 2 probabilities)
        """
        if len(probs) < 2:
            return 0.0

        sorted_probs = np.sort(probs)
        return sorted_probs[-1] - sorted_probs[-2]

    @staticmethod
    def least_confident(probs: np.ndarray) -> float:
        """
        Calculate least confidence (1 - max probability).

        Args:
            probs: Array of probabilities

        Returns:
            Least confidence score
        """
        return 1.0 - np.max(probs)

    @classmethod
    def query_instances(
        cls,
        classifier,
        X_pool: List[Dict[str, int]],
        pool_ids: List[str],
        n_queries: int,
        strategy: str = "margin"
    ) -> List[Tuple[str, Dict[str, int], np.ndarray]]:
        """
        Select instances to query based on uncertainty.

        Args:
            classifier: Trained NaiveBayesWithPriors classifier
            X_pool: Pool of unlabeled instances (feature dicts)
            pool_ids: IDs for each instance
            n_queries: Number of instances to query
            strategy: One of "entropy", "margin", "least_confident", "random"

        Returns:
            List of (instance_id, features, probabilities) tuples
        """
        if len(X_pool) == 0:
            return []

        n_queries = min(n_queries, len(X_pool))

        if strategy == "random":
            indices = np.random.choice(len(X_pool), n_queries, replace=False)
            probs = classifier.predict_proba([X_pool[i] for i in indices])
            return [
                (pool_ids[idx], X_pool[idx], probs[i])
                for i, idx in enumerate(indices)
            ]

        # Predict probabilities for all instances
        probs = classifier.predict_proba(X_pool)

        # Calculate uncertainty scores
        if strategy == "entropy":
            scores = np.array([cls.entropy(p) for p in probs])
            # Higher entropy = more uncertain, so we want highest scores
            top_indices = np.argsort(scores)[-n_queries:][::-1]

        elif strategy == "margin":
            scores = np.array([cls.margin(p) for p in probs])
            # Lower margin = more uncertain, so we want lowest scores
            top_indices = np.argsort(scores)[:n_queries]

        elif strategy == "least_confident":
            scores = np.array([cls.least_confident(p) for p in probs])
            # Higher score = more uncertain, so we want highest scores
            top_indices = np.argsort(scores)[-n_queries:][::-1]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(f"Selected {len(top_indices)} instances using {strategy} strategy")

        return [
            (pool_ids[idx], X_pool[idx], probs[idx])
            for idx in top_indices
        ]

    @staticmethod
    def mutual_information(
        feature_counts: Dict[str, Dict[str, int]],
        label_counts: Dict[str, int],
        total_instances: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-label mutual information for features.

        MI(feature, label) = sum_f sum_l P(f,l) * log(P(f,l) / (P(f) * P(l)))

        Args:
            feature_counts: {feature: {label: count}}
            label_counts: {label: count}
            total_instances: Total number of instances

        Returns:
            {label: {feature: MI_score}}
        """
        if total_instances == 0:
            return {}

        # Calculate P(label)
        label_probs = {
            label: count / total_instances
            for label, count in label_counts.items()
        }

        # Calculate P(feature)
        feature_totals = defaultdict(int)
        for feature, label_dict in feature_counts.items():
            feature_totals[feature] = sum(label_dict.values())

        feature_probs = {
            feature: count / total_instances
            for feature, count in feature_totals.items()
        }

        # Calculate MI per label
        mi_scores = defaultdict(dict)

        for label in label_counts.keys():
            for feature in feature_counts.keys():
                # P(feature, label)
                joint_count = feature_counts[feature].get(label, 0)
                if joint_count == 0:
                    mi_scores[label][feature] = 0.0
                    continue

                p_joint = joint_count / total_instances
                p_feature = feature_probs[feature]
                p_label = label_probs[label]

                # MI = P(f,l) * log(P(f,l) / (P(f) * P(l)))
                if p_feature > 0 and p_label > 0:
                    mi = p_joint * np.log2(p_joint / (p_feature * p_label))
                    mi_scores[label][feature] = mi
                else:
                    mi_scores[label][feature] = 0.0

        return mi_scores

    @classmethod
    def query_features_per_label(
        cls,
        classifier,
        X_labeled: List[Dict[str, int]],
        y_labeled: List[str],
        labeled_features: Set[str],
        n_queries_per_label: int,
        min_correlation: float = 0.75
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Select features to query based on per-label mutual information.

        Args:
            classifier: Trained NaiveBayesWithPriors classifier
            X_labeled: Labeled instances (feature dicts)
            y_labeled: Labels for instances
            labeled_features: Set of already-labeled features (to skip)
            n_queries_per_label: Number of features to query per label
            min_correlation: Minimum correlation threshold (default 0.75)

        Returns:
            {label: [(feature, MI_score), ...]}
        """
        if len(X_labeled) == 0:
            return {}

        # Count feature occurrences per label
        feature_counts = defaultdict(lambda: defaultdict(int))
        label_counts = Counter(y_labeled)

        for features, label in zip(X_labeled, y_labeled):
            for feature in features.keys():
                feature_counts[feature][label] += features[feature]

        # Calculate MI
        mi_scores = cls.mutual_information(
            feature_counts,
            label_counts,
            len(X_labeled)
        )

        # Select top features per label with correlation filtering
        selected_features = {}

        for label in label_counts.keys():
            # Filter out already-labeled features
            candidates = {
                feature: score
                for feature, score in mi_scores[label].items()
                if feature not in labeled_features
            }

            # Apply correlation filter
            # Feature must have > min_correlation of its max label frequency in this label
            filtered_candidates = {}
            for feature, score in candidates.items():
                if feature not in feature_counts:
                    continue

                label_dist = feature_counts[feature]
                max_count = max(label_dist.values()) if label_dist else 0
                this_label_count = label_dist.get(label, 0)

                if max_count > 0 and this_label_count / max_count >= min_correlation:
                    filtered_candidates[feature] = score

            # Sort by MI and take top N
            sorted_features = sorted(
                filtered_candidates.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_queries_per_label]

            selected_features[label] = sorted_features

            logger.info(f"Selected {len(sorted_features)} features for label '{label}'")

        return selected_features


class BootstrappingSampler:
    """
    Utility for bootstrapping initial labeled data for cold-start.
    """

    @staticmethod
    def sample_random_instances(
        X_pool: List[Dict[str, int]],
        pool_ids: List[str],
        n_samples: int
    ) -> List[Tuple[str, Dict[str, int]]]:
        """
        Randomly sample instances from pool.

        Args:
            X_pool: Pool of unlabeled instances
            pool_ids: IDs for each instance
            n_samples: Number of samples

        Returns:
            List of (instance_id, features) tuples
        """
        n_samples = min(n_samples, len(X_pool))
        indices = np.random.choice(len(X_pool), n_samples, replace=False)

        return [
            (pool_ids[idx], X_pool[idx])
            for idx in indices
        ]

    @staticmethod
    def sample_diverse_instances(
        X_pool: List[Dict[str, int]],
        pool_ids: List[str],
        n_samples: int
    ) -> List[Tuple[str, Dict[str, int]]]:
        """
        Sample diverse instances using simple clustering/diversity.

        This is a simple implementation that samples instances with
        different feature sets.

        Args:
            X_pool: Pool of unlabeled instances
            pool_ids: IDs for each instance
            n_samples: Number of samples

        Returns:
            List of (instance_id, features) tuples
        """
        if len(X_pool) <= n_samples:
            return [(pool_ids[i], X_pool[i]) for i in range(len(X_pool))]

        selected_indices = []
        selected_features = set()

        # First, select instance with most features
        feature_counts = [len(x) for x in X_pool]
        first_idx = np.argmax(feature_counts)
        selected_indices.append(first_idx)
        selected_features.update(X_pool[first_idx].keys())

        # Greedily select instances with most new features
        while len(selected_indices) < n_samples:
            best_idx = -1
            best_new_features = 0

            for i, features in enumerate(X_pool):
                if i in selected_indices:
                    continue

                new_features = len(set(features.keys()) - selected_features)
                if new_features > best_new_features:
                    best_new_features = new_features
                    best_idx = i

            if best_idx == -1:
                # No more diverse instances, sample randomly
                remaining = list(set(range(len(X_pool))) - set(selected_indices))
                best_idx = np.random.choice(remaining)

            selected_indices.append(best_idx)
            selected_features.update(X_pool[best_idx].keys())

        return [
            (pool_ids[idx], X_pool[idx])
            for idx in selected_indices
        ]
