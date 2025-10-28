"""
FastAPI backend for Active Learning with LLM integration.

Provides endpoints for:
- Data loading and preprocessing
- Model training and incremental updates
- Active query generation
- Predictions
- Model evaluation
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import tempfile
import logging
from pathlib import Path

from ..core.naive_bayes_with_priors import NaiveBayesWithPriors
from ..core.queries import ActiveLearningQueries, BootstrappingSampler
from ..pipelines.text_pipelines import get_pipe
from ..utils.data_loader import DataLoader
from ..evaluation.metrics import ModelEvaluator
from .llm_endpoints import router as llm_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dualist Active Learning API",
    description="Active Learning API with LLM integration for text classification",
    version="2.0.0"
)

# CORS middleware for Django frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include LLM router
app.include_router(llm_router)

# Global state (in production, use proper state management)
class AppState:
    def __init__(self):
        self.classifier: Optional[NaiveBayesWithPriors] = None
        self.pipe = None
        self.X_pool: List[Dict[str, int]] = []
        self.y_pool: List[str] = []  # True labels (hidden in real AL)
        self.pool_ids: List[str] = []
        self.X_labeled: List[Dict[str, int]] = []
        self.y_labeled: List[str] = []
        self.labeled_ids: List[str] = []
        self.labeled_features: set = set()
        self.model_path: str = "/home/user/dualist-revamp/python-backend/saved_models"

state = AppState()


# Pydantic models for API
class InitializeRequest(BaseModel):
    pipe_type: str = "document"  # document, lines, twitter, entity
    alpha: float = 100.0
    smoothing: float = 1.0


class LoadDataRequest(BaseModel):
    data_path: str
    data_format: str = "zip"  # zip, directory, json, csv
    text_field: Optional[str] = "text"
    label_field: Optional[str] = "label"


class LabelInstanceRequest(BaseModel):
    instance_id: str
    label: str


class LabelFeatureRequest(BaseModel):
    feature: str
    label: str
    weight: Optional[float] = None


class QueryRequest(BaseModel):
    n_instances: int = 10
    strategy: str = "margin"  # entropy, margin, least_confident, random
    n_features_per_label: int = 5


class PredictRequest(BaseModel):
    texts: List[str]


class TrainRequest(BaseModel):
    retrain: bool = False


# API Endpoints

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Dualist Active Learning API",
        "version": "2.0.0"
    }


@app.post("/initialize")
def initialize(request: InitializeRequest):
    """Initialize the classifier and pipeline."""
    try:
        # Create pipeline
        state.pipe = get_pipe(request.pipe_type)

        # Create classifier
        state.classifier = NaiveBayesWithPriors(
            alpha=request.alpha,
            smoothing=request.smoothing
        )

        # Reset state
        state.X_pool = []
        state.y_pool = []
        state.pool_ids = []
        state.X_labeled = []
        state.y_labeled = []
        state.labeled_ids = []
        state.labeled_features = set()

        logger.info(f"Initialized with pipe_type={request.pipe_type}, alpha={request.alpha}")

        return {
            "status": "success",
            "pipe_type": request.pipe_type,
            "alpha": request.alpha,
            "smoothing": request.smoothing
        }

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_data")
def load_data(request: LoadDataRequest):
    """Load data from file."""
    try:
        if state.pipe is None:
            raise ValueError("Pipeline not initialized. Call /initialize first.")

        loader = DataLoader()

        # Load based on format
        if request.data_format == "zip":
            X, y, ids = loader.load_from_zip(request.data_path, state.pipe)
        elif request.data_format == "directory":
            X, y, ids = loader.load_from_directory(request.data_path, state.pipe)
        elif request.data_format == "json":
            X, y, ids = loader.load_from_json(
                request.data_path,
                state.pipe,
                text_field=request.text_field,
                label_field=request.label_field
            )
        elif request.data_format == "csv":
            X, y, ids = loader.load_from_csv(
                request.data_path,
                state.pipe,
                text_column=request.text_field,
                label_column=request.label_field
            )
        else:
            raise ValueError(f"Unknown data format: {request.data_format}")

        # Store in pool
        state.X_pool = X
        state.y_pool = y  # Hidden in real AL, used for evaluation
        state.pool_ids = ids

        logger.info(f"Loaded {len(X)} instances from {request.data_path}")

        return {
            "status": "success",
            "n_instances": len(X),
            "labels": sorted(set(y)),
            "n_features": sum(len(x) for x in X)
        }

    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(request: QueryRequest):
    """Generate active learning queries."""
    try:
        if state.classifier is None:
            raise ValueError("Classifier not initialized.")

        if len(state.X_pool) == 0:
            return {
                "status": "no_data",
                "instance_queries": [],
                "feature_queries": {}
            }

        # Query instances
        instance_queries = []

        if len(state.X_labeled) > 0:
            # Use active learning strategy
            queries = ActiveLearningQueries.query_instances(
                state.classifier,
                state.X_pool,
                state.pool_ids,
                min(request.n_instances, len(state.X_pool)),
                strategy=request.strategy
            )

            for inst_id, features, probs in queries:
                # Get top predictions
                label_indices = probs.argsort()[::-1]
                predictions = [
                    {
                        "label": state.classifier.idx_to_label[idx],
                        "probability": float(probs[idx])
                    }
                    for idx in label_indices
                ]

                # Get text preview (first 5 features)
                feature_preview = list(features.keys())[:5]

                instance_queries.append({
                    "instance_id": inst_id,
                    "features": feature_preview,
                    "predictions": predictions
                })

        else:
            # Bootstrap with random/diverse samples
            samples = BootstrappingSampler.sample_diverse_instances(
                state.X_pool,
                state.pool_ids,
                request.n_instances
            )

            for inst_id, features in samples:
                feature_preview = list(features.keys())[:5]
                instance_queries.append({
                    "instance_id": inst_id,
                    "features": feature_preview,
                    "predictions": []
                })

        # Query features (if we have labeled data)
        feature_queries = {}

        if len(state.X_labeled) > 0:
            feature_selections = ActiveLearningQueries.query_features_per_label(
                state.classifier,
                state.X_labeled,
                state.y_labeled,
                state.labeled_features,
                request.n_features_per_label
            )

            for label, features in feature_selections.items():
                feature_queries[label] = [
                    {"feature": feat, "score": float(score)}
                    for feat, score in features
                ]

        logger.info(f"Generated {len(instance_queries)} instance queries, "
                   f"{len(feature_queries)} feature query sets")

        return {
            "status": "success",
            "instance_queries": instance_queries,
            "feature_queries": feature_queries,
            "n_labeled": len(state.X_labeled),
            "n_pool": len(state.X_pool)
        }

    except Exception as e:
        logger.error(f"Query generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/label_instance")
def label_instance(request: LabelInstanceRequest):
    """Label an instance and update the model."""
    try:
        # Find instance in pool
        if request.instance_id not in state.pool_ids:
            raise ValueError(f"Instance not found: {request.instance_id}")

        idx = state.pool_ids.index(request.instance_id)

        # Move from pool to labeled
        state.X_labeled.append(state.X_pool[idx])
        state.y_labeled.append(request.label)
        state.labeled_ids.append(state.pool_ids[idx])

        # Remove from pool
        del state.X_pool[idx]
        del state.y_pool[idx]
        del state.pool_ids[idx]

        logger.info(f"Labeled instance {request.instance_id} as '{request.label}'")

        return {
            "status": "success",
            "instance_id": request.instance_id,
            "label": request.label,
            "n_labeled": len(state.X_labeled),
            "n_pool": len(state.X_pool)
        }

    except Exception as e:
        logger.error(f"Instance labeling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/label_feature")
def label_feature(request: LabelFeatureRequest):
    """Label a feature and add to classifier priors."""
    try:
        if state.classifier is None:
            raise ValueError("Classifier not initialized.")

        # Add feature prior
        state.classifier.add_label_feature(
            request.label,
            request.feature,
            weight=request.weight
        )

        # Track labeled features
        state.labeled_features.add(request.feature)

        logger.info(f"Labeled feature '{request.feature}' as '{request.label}'")

        return {
            "status": "success",
            "feature": request.feature,
            "label": request.label,
            "n_labeled_features": len(state.labeled_features)
        }

    except Exception as e:
        logger.error(f"Feature labeling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train(request: TrainRequest):
    """Train or retrain the classifier."""
    try:
        if state.classifier is None:
            raise ValueError("Classifier not initialized.")

        if len(state.X_labeled) == 0:
            raise ValueError("No labeled data available.")

        if request.retrain or state.classifier.n_features == 0:
            # Full retrain
            state.classifier.train(state.X_labeled, state.y_labeled)
        else:
            # Incremental update
            state.classifier.train_incremental(
                [state.X_labeled[-1]],
                [state.y_labeled[-1]]
            )

        stats = state.classifier.get_stats()

        logger.info(f"Trained classifier: {stats}")

        return {
            "status": "success",
            "model_stats": stats
        }

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(request: PredictRequest):
    """Predict labels for new texts."""
    try:
        if state.classifier is None or state.pipe is None:
            raise ValueError("Model not initialized.")

        # Process texts
        X = [state.pipe.process(text) for text in request.texts]

        # Predict
        y_pred = state.classifier.predict(X)
        y_proba = state.classifier.predict_proba(X)

        results = []
        for i, (text, label, probs) in enumerate(zip(request.texts, y_pred, y_proba)):
            # Get all label probabilities
            label_probs = {
                state.classifier.idx_to_label[j]: float(probs[j])
                for j in range(len(probs))
            }

            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "predicted_label": label,
                "confidence": float(probs[state.classifier.label_vocab[label]]),
                "all_probabilities": label_probs
            })

        return {
            "status": "success",
            "predictions": results
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
def evaluate():
    """Evaluate current model on remaining pool data (for testing)."""
    try:
        if state.classifier is None:
            raise ValueError("Model not initialized.")

        if len(state.X_pool) == 0:
            return {
                "status": "no_data",
                "message": "No pool data available for evaluation."
            }

        # Evaluate on pool (using true labels)
        metrics = ModelEvaluator.evaluate(
            state.classifier,
            state.X_pool,
            state.y_pool
        )

        return {
            "status": "success",
            "metrics": metrics,
            "n_test_instances": len(state.X_pool)
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def get_status():
    """Get current system status."""
    try:
        stats = {}

        if state.classifier:
            stats = state.classifier.get_stats()

        return {
            "status": "ok",
            "classifier_initialized": state.classifier is not None,
            "pipe_initialized": state.pipe is not None,
            "n_pool": len(state.X_pool),
            "n_labeled": len(state.X_labeled),
            "n_labeled_features": len(state.labeled_features),
            "model_stats": stats
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_model")
def save_model(filename: str = "model.pkl"):
    """Save current model to disk."""
    try:
        if state.classifier is None:
            raise ValueError("No model to save.")

        # Ensure model directory exists
        os.makedirs(state.model_path, exist_ok=True)

        filepath = os.path.join(state.model_path, filename)
        state.classifier.save(filepath)

        return {
            "status": "success",
            "filepath": filepath
        }

    except Exception as e:
        logger.error(f"Model save failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_model")
def load_model(filename: str = "model.pkl"):
    """Load model from disk."""
    try:
        filepath = os.path.join(state.model_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        state.classifier = NaiveBayesWithPriors.load(filepath)

        return {
            "status": "success",
            "filepath": filepath,
            "model_stats": state.classifier.get_stats()
        }

    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
