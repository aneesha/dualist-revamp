"""
LLM-specific endpoints for automated labeling in active learning.

These endpoints integrate LLM labeling with the active learning loop.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

from ..core.llm_labeler import LLMLabeler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])

# Global LLM labeler instance
llm_labeler: Optional[LLMLabeler] = None


class InitializeLLMRequest(BaseModel):
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    api_key: Optional[str] = None


class LLMLabelInstanceRequest(BaseModel):
    text: str
    labels: List[str]
    task_description: str
    few_shot_examples: Optional[List[Dict[str, str]]] = None


class LLMLabelFeatureRequest(BaseModel):
    feature: str
    label: str
    task_description: str
    context_examples: Optional[List[str]] = None


class LLMLabelBatchRequest(BaseModel):
    texts: List[str]
    labels: List[str]
    task_description: str
    few_shot_examples: Optional[List[Dict[str, str]]] = None


class GenerateFeaturesRequest(BaseModel):
    labels: List[str]
    task_description: str
    n_features_per_label: int = 10


@router.post("/initialize")
def initialize_llm(request: InitializeLLMRequest):
    """
    Initialize the LLM labeler.

    This must be called before using LLM labeling endpoints.
    """
    global llm_labeler

    try:
        llm_labeler = LLMLabeler(
            model_name=request.model_name,
            temperature=request.temperature,
            api_key=request.api_key
        )

        return {
            "status": "success",
            "model_name": request.model_name,
            "temperature": request.temperature
        }

    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/label_instance")
def llm_label_instance(request: LLMLabelInstanceRequest):
    """
    Use LLM to label a single text instance.

    Returns the predicted label, confidence, and reasoning.
    """
    global llm_labeler

    if llm_labeler is None:
        raise HTTPException(
            status_code=400,
            detail="LLM not initialized. Call /llm/initialize first."
        )

    try:
        label, confidence, reasoning = llm_labeler.label_instance(
            text=request.text,
            labels=request.labels,
            task_description=request.task_description,
            few_shot_examples=request.few_shot_examples
        )

        return {
            "status": "success",
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning
        }

    except Exception as e:
        logger.error(f"LLM instance labeling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/label_feature")
def llm_label_feature(request: LLMLabelFeatureRequest):
    """
    Use LLM to determine if a feature is relevant for a label.

    Returns relevance decision, confidence, and reasoning.
    """
    global llm_labeler

    if llm_labeler is None:
        raise HTTPException(
            status_code=400,
            detail="LLM not initialized. Call /llm/initialize first."
        )

    try:
        relevant, confidence, reasoning = llm_labeler.label_feature(
            feature=request.feature,
            label=request.label,
            task_description=request.task_description,
            context_examples=request.context_examples
        )

        return {
            "status": "success",
            "relevant": relevant,
            "confidence": confidence,
            "reasoning": reasoning
        }

    except Exception as e:
        logger.error(f"LLM feature labeling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/label_batch")
def llm_label_batch(request: LLMLabelBatchRequest):
    """
    Use LLM to label a batch of text instances.

    Returns list of predictions with confidence and reasoning.
    """
    global llm_labeler

    if llm_labeler is None:
        raise HTTPException(
            status_code=400,
            detail="LLM not initialized. Call /llm/initialize first."
        )

    try:
        results = llm_labeler.label_batch_instances(
            texts=request.texts,
            labels=request.labels,
            task_description=request.task_description,
            few_shot_examples=request.few_shot_examples
        )

        predictions = [
            {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "label": label,
                "confidence": confidence,
                "reasoning": reasoning
            }
            for text, (label, confidence, reasoning) in zip(request.texts, results)
        ]

        return {
            "status": "success",
            "predictions": predictions
        }

    except Exception as e:
        logger.error(f"LLM batch labeling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_features")
def llm_generate_features(request: GenerateFeaturesRequest):
    """
    Use LLM to generate initial discriminative features for each label.

    This is useful for cold-start initialization.
    """
    global llm_labeler

    if llm_labeler is None:
        raise HTTPException(
            status_code=400,
            detail="LLM not initialized. Call /llm/initialize first."
        )

    try:
        features = llm_labeler.generate_initial_labeled_features(
            labels=request.labels,
            task_description=request.task_description,
            n_features_per_label=request.n_features_per_label
        )

        return {
            "status": "success",
            "features": features
        }

    except Exception as e:
        logger.error(f"LLM feature generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def llm_status():
    """Get LLM labeler status."""
    global llm_labeler

    return {
        "initialized": llm_labeler is not None,
        "model": "gpt-3.5-turbo" if llm_labeler else None
    }
