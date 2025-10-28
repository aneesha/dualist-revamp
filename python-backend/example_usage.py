"""
Example: Using the Active Learning system programmatically.

This script demonstrates how to use the core components
without the FastAPI/Django web interface.
"""

import os
import sys

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.naive_bayes_with_priors import NaiveBayesWithPriors
from app.core.queries import ActiveLearningQueries
from app.core.llm_labeler import LLMLabeler
from app.pipelines.text_pipelines import get_pipe
from app.utils.data_loader import DataLoader
from app.evaluation.metrics import ModelEvaluator


def main():
    print("=" * 60)
    print("Dualist Active Learning - Programmatic Example")
    print("=" * 60)
    print()

    # 1. Initialize pipeline and classifier
    print("1. Initializing pipeline and classifier...")
    pipe = get_pipe("document")
    classifier = NaiveBayesWithPriors(alpha=100.0, smoothing=1.0)
    print("   ✓ Pipeline: document")
    print("   ✓ Classifier: NaiveBayesWithPriors")
    print()

    # 2. Load data
    print("2. Loading data...")
    data_path = "../data/baseball-hockey.zip"

    if not os.path.exists(data_path):
        print(f"   ✗ Data file not found: {data_path}")
        print("   Please ensure the baseball-hockey.zip file exists.")
        return

    loader = DataLoader()
    X, y, ids = loader.load_from_zip(data_path, pipe)
    print(f"   ✓ Loaded {len(X)} instances")
    print(f"   ✓ Labels: {set(y)}")
    print()

    # Split into pool and test set
    split_idx = int(len(X) * 0.8)
    X_pool, y_pool, ids_pool = X[:split_idx], y[:split_idx], ids[:split_idx]
    X_test, y_test, ids_test = X[split_idx:], y[split_idx:], ids[split_idx:]

    print(f"   ✓ Pool size: {len(X_pool)}")
    print(f"   ✓ Test size: {len(X_test)}")
    print()

    # 3. Initialize LLM (optional - comment out if no API key)
    print("3. Initializing LLM labeler...")
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print("   ✗ OPENAI_API_KEY not set. Skipping LLM features.")
        print("   Set with: export OPENAI_API_KEY='your-key'")
        use_llm = False
    else:
        try:
            llm = LLMLabeler(model_name="gpt-3.5-turbo", temperature=0.1, api_key=api_key)
            print("   ✓ LLM initialized: gpt-3.5-turbo")
            use_llm = True
        except Exception as e:
            print(f"   ✗ LLM initialization failed: {e}")
            use_llm = False
    print()

    # 4. Active Learning Loop (simplified)
    print("4. Starting Active Learning loop...")
    print()

    X_labeled = []
    y_labeled = []
    labeled_indices = []
    task_description = "Classify sports articles into baseball or hockey"
    labels = list(set(y))

    n_iterations = 5
    n_queries_per_iteration = 5

    for iteration in range(n_iterations):
        print(f"   Iteration {iteration + 1}/{n_iterations}")
        print(f"   {'=' * 50}")

        # Get unlabeled pool
        X_unlabeled = [X_pool[i] for i in range(len(X_pool)) if i not in labeled_indices]
        pool_ids = [ids_pool[i] for i in range(len(X_pool)) if i not in labeled_indices]

        if len(X_unlabeled) == 0:
            print("   No more unlabeled instances!")
            break

        # First iteration: bootstrap with random samples
        if iteration == 0:
            # Randomly select initial instances
            import random
            random.seed(42)
            selected_indices = random.sample(range(len(X_unlabeled)), n_queries_per_iteration)
            queries = [(pool_ids[i], X_unlabeled[i], None) for i in selected_indices]
            print(f"   Bootstrapping with {len(queries)} random samples")
        else:
            # Active learning: query most uncertain instances
            queries = ActiveLearningQueries.query_instances(
                classifier,
                X_unlabeled,
                pool_ids,
                n_queries_per_iteration,
                strategy="margin"
            )
            print(f"   Selected {len(queries)} uncertain instances")

        # Label instances (with LLM or simulated)
        for query_id, features, probs in queries:
            # Find original index
            orig_idx = ids_pool.index(query_id)

            if use_llm and iteration > 0:  # Use LLM for active queries
                # Reconstruct text from features (simplified)
                text = ' '.join(list(features.keys())[:20])

                try:
                    label, confidence, reasoning = llm.label_instance(
                        text, labels, task_description
                    )
                    print(f"   LLM labeled '{query_id[:30]}...' as '{label}' "
                          f"(confidence: {confidence:.2f})")
                except Exception as e:
                    print(f"   LLM error: {e}. Using true label.")
                    label = y_pool[orig_idx]
            else:
                # Use true label (simulating oracle)
                label = y_pool[orig_idx]
                print(f"   Oracle labeled '{query_id[:30]}...' as '{label}'")

            # Add to labeled set
            X_labeled.append(features)
            y_labeled.append(label)
            labeled_indices.append(orig_idx)

        # Train classifier
        if iteration == 0:
            classifier.train(X_labeled, y_labeled)
        else:
            classifier.train_incremental(
                X_labeled[-n_queries_per_iteration:],
                y_labeled[-n_queries_per_iteration:]
            )

        print(f"   Trained on {len(X_labeled)} total labeled instances")

        # Evaluate on test set
        metrics = ModelEvaluator.evaluate(classifier, X_test, y_test)
        print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Test F1: {metrics['macro_f1']:.4f}")
        print()

    # 5. Final evaluation
    print("5. Final Evaluation")
    print("=" * 60)

    final_metrics = ModelEvaluator.evaluate(classifier, X_test, y_test)

    print(f"Accuracy:        {final_metrics['accuracy']:.4f}")
    print(f"Macro F1:        {final_metrics['macro_f1']:.4f}")
    print(f"Macro Precision: {final_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {final_metrics['macro_recall']:.4f}")
    print()

    print("Per-Label F1:")
    for key, value in final_metrics.items():
        if key.startswith('f1_') and key not in ['macro_f1', 'weighted_f1']:
            label = key[3:]
            print(f"  {label:15s} {value:.4f}")
    print()

    # 6. Save model
    print("6. Saving model...")
    os.makedirs("saved_models", exist_ok=True)
    model_path = "saved_models/example_model.pkl"
    classifier.save(model_path)
    print(f"   ✓ Model saved to {model_path}")
    print()

    # 7. Test predictions
    print("7. Testing predictions on new texts...")
    new_texts = [
        "The pitcher threw a fastball for a strike",
        "The goalie made an incredible save",
        "He hit a home run in the ninth inning",
        "The team scored in overtime"
    ]

    X_new = [pipe.process(text) for text in new_texts]
    predictions = classifier.predict(X_new)
    probabilities = classifier.predict_proba(X_new)

    for text, pred, probs in zip(new_texts, predictions, probabilities):
        conf = probs[classifier.label_vocab[pred]]
        print(f"   '{text}'")
        print(f"   → {pred} (confidence: {conf:.3f})")
        print()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
