"""
Evaluation utilities for model assessment.

Implements:
- Cross-validation
- Accuracy, F1, Precision, Recall
- Statistical utilities

Based on the original Java Test.java and Util.java implementations.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate classifier performance."""

    @staticmethod
    def evaluate(
        classifier,
        X_test: List[Dict[str, int]],
        y_test: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate classifier on test data.

        Args:
            classifier: Trained classifier
            X_test: Test instances (feature dicts)
            y_test: True labels

        Returns:
            Dictionary of metrics
        """
        if len(X_test) == 0:
            return {}

        # Predict
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)

        # Get unique labels
        labels = sorted(set(y_test))

        # Calculate metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro', labels=labels, zero_division=0),
            'macro_precision': precision_score(y_test, y_pred, average='macro', labels=labels, zero_division=0),
            'macro_recall': recall_score(y_test, y_pred, average='macro', labels=labels, zero_division=0),
            'weighted_f1': f1_score(y_test, y_pred, average='weighted', labels=labels, zero_division=0),
        }

        # Per-label F1
        per_label_f1 = f1_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
        for label, f1 in zip(labels, per_label_f1):
            results[f'f1_{label}'] = f1

        logger.info(f"Evaluation: Accuracy={results['accuracy']:.4f}, Macro F1={results['macro_f1']:.4f}")

        return results

    @staticmethod
    def cross_validate(
        X: List[Dict[str, int]],
        y: List[str],
        classifier_class,
        classifier_params: Dict,
        n_folds: int = 10,
        random_state: int = 42
    ) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation.

        Args:
            X: Instances (feature dicts)
            y: Labels
            classifier_class: Classifier class to instantiate
            classifier_params: Parameters for classifier
            n_folds: Number of folds
            random_state: Random seed

        Returns:
            Dictionary of metric lists (one per fold)
        """
        if len(X) < n_folds:
            logger.warning(f"Not enough data for {n_folds}-fold CV. Using {len(X)} folds.")
            n_folds = len(X)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        fold_results = {
            'accuracy': [],
            'macro_f1': [],
            'macro_precision': [],
            'macro_recall': []
        }

        # Get unique labels for per-label metrics
        labels = sorted(set(y))
        for label in labels:
            fold_results[f'f1_{label}'] = []

        X_array = np.array(X)
        y_array = np.array(y)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_array)):
            # Split data
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]

            # Train classifier
            clf = classifier_class(**classifier_params)
            clf.train(X_train, y_train)

            # Evaluate
            metrics = ModelEvaluator.evaluate(clf, X_test, y_test)

            # Store results
            for key, value in metrics.items():
                if key in fold_results:
                    fold_results[key].append(value)

            logger.info(f"Fold {fold_idx + 1}/{n_folds}: Accuracy={metrics['accuracy']:.4f}")

        return fold_results

    @staticmethod
    def summarize_cv_results(cv_results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Summarize cross-validation results.

        Args:
            cv_results: Results from cross_validate()

        Returns:
            Dictionary with mean and std for each metric
        """
        summary = {}

        for metric, values in cv_results.items():
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return summary

    @staticmethod
    def print_evaluation_report(
        test_results: Dict[str, float],
        cv_results: Dict[str, List[float]]
    ):
        """
        Print formatted evaluation report.

        Args:
            test_results: Results from evaluate()
            cv_results: Results from cross_validate()
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)

        print("\nTest Set Performance:")
        print(f"  Accuracy:          {test_results['accuracy']:.4f}")
        print(f"  Macro F1:          {test_results['macro_f1']:.4f}")
        print(f"  Macro Precision:   {test_results['macro_precision']:.4f}")
        print(f"  Macro Recall:      {test_results['macro_recall']:.4f}")
        print(f"  Weighted F1:       {test_results['weighted_f1']:.4f}")

        # Per-label F1
        print("\n  Per-Label F1:")
        for key, value in sorted(test_results.items()):
            if key.startswith('f1_') and key != 'f1':
                label = key[3:]
                print(f"    {label:20s} {value:.4f}")

        # Cross-validation results
        cv_summary = ModelEvaluator.summarize_cv_results(cv_results)

        print(f"\nCross-Validation ({len(cv_results['accuracy'])} folds):")
        print(f"  Accuracy:          {cv_summary['accuracy']['mean']:.4f} ± {cv_summary['accuracy']['std']:.4f}")
        print(f"  Macro F1:          {cv_summary['macro_f1']['mean']:.4f} ± {cv_summary['macro_f1']['std']:.4f}")
        print(f"  Macro Precision:   {cv_summary['macro_precision']['mean']:.4f} ± {cv_summary['macro_precision']['std']:.4f}")
        print(f"  Macro Recall:      {cv_summary['macro_recall']['mean']:.4f} ± {cv_summary['macro_recall']['std']:.4f}")

        print("\n" + "=" * 60 + "\n")


class ActiveLearningEvaluator:
    """Evaluate active learning performance."""

    @staticmethod
    def evaluate_learning_curve(
        classifier,
        X_pool: List[Dict[str, int]],
        y_pool: List[str],
        X_test: List[Dict[str, int]],
        y_test: List[str],
        query_strategy: str = "margin",
        n_iterations: int = 10,
        n_queries_per_iteration: int = 10,
        initial_samples: int = 5
    ) -> Dict[str, List]:
        """
        Evaluate learning curve for active learning.

        Args:
            classifier: Classifier instance
            X_pool: Pool of unlabeled instances
            y_pool: True labels for pool (for simulation)
            X_test: Test instances
            y_test: Test labels
            query_strategy: Query strategy to use
            n_iterations: Number of AL iterations
            n_queries_per_iteration: Number of queries per iteration
            initial_samples: Number of initial random samples

        Returns:
            Dictionary with iteration results
        """
        from ..core.queries import ActiveLearningQueries

        results = {
            'iteration': [],
            'n_labeled': [],
            'test_accuracy': [],
            'test_f1': []
        }

        # Start with random initial samples
        pool_indices = list(range(len(X_pool)))
        np.random.shuffle(pool_indices)

        labeled_indices = pool_indices[:initial_samples]
        unlabeled_indices = pool_indices[initial_samples:]

        for iteration in range(n_iterations):
            # Get labeled data
            X_train = [X_pool[i] for i in labeled_indices]
            y_train = [y_pool[i] for i in labeled_indices]

            # Train classifier
            if iteration == 0:
                classifier.train(X_train, y_train)
            else:
                classifier.train_incremental(X_train[-n_queries_per_iteration:],
                                            y_train[-n_queries_per_iteration:])

            # Evaluate on test set
            metrics = ModelEvaluator.evaluate(classifier, X_test, y_test)

            # Record results
            results['iteration'].append(iteration)
            results['n_labeled'].append(len(labeled_indices))
            results['test_accuracy'].append(metrics['accuracy'])
            results['test_f1'].append(metrics['macro_f1'])

            logger.info(f"Iteration {iteration}: {len(labeled_indices)} labeled, "
                       f"Accuracy={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")

            # Break if no more unlabeled instances
            if len(unlabeled_indices) == 0:
                break

            # Query next instances
            X_unlabeled = [X_pool[i] for i in unlabeled_indices]
            pool_ids = [str(i) for i in unlabeled_indices]

            queries = ActiveLearningQueries.query_instances(
                classifier,
                X_unlabeled,
                pool_ids,
                min(n_queries_per_iteration, len(unlabeled_indices)),
                strategy=query_strategy
            )

            # Add queried instances to labeled set
            for query_id, _, _ in queries:
                idx = int(query_id)
                labeled_indices.append(idx)
                unlabeled_indices.remove(idx)

        return results

    @staticmethod
    def plot_learning_curve(results: Dict[str, List], output_path: str = None):
        """
        Plot learning curve.

        Args:
            results: Results from evaluate_learning_curve()
            output_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Accuracy curve
            ax1.plot(results['n_labeled'], results['test_accuracy'], 'b-o', linewidth=2)
            ax1.set_xlabel('Number of Labeled Instances')
            ax1.set_ylabel('Test Accuracy')
            ax1.set_title('Active Learning Curve: Accuracy')
            ax1.grid(True, alpha=0.3)

            # F1 curve
            ax2.plot(results['n_labeled'], results['test_f1'], 'r-o', linewidth=2)
            ax2.set_xlabel('Number of Labeled Instances')
            ax2.set_ylabel('Test Macro F1')
            ax2.set_title('Active Learning Curve: F1')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Learning curve saved to {output_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not installed. Cannot plot learning curve.")
