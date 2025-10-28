"""
Views for Active Learning UI with LLM integration.
"""

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import requests
import json
import logging

logger = logging.getLogger(__name__)

# Helper function to call FastAPI backend
def call_backend(endpoint, method='GET', data=None):
    """Call the FastAPI backend."""
    url = f"{settings.FASTAPI_BACKEND_URL}{endpoint}"

    try:
        if method == 'GET':
            response = requests.get(url)
        elif method == 'POST':
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Backend call failed: {e}")
        return {"status": "error", "detail": str(e)}


# Main views

def index(request):
    """Main dashboard view."""
    # Get system status
    status = call_backend('/status')

    context = {
        'status': status,
        'fastapi_url': settings.FASTAPI_BACKEND_URL
    }

    return render(request, 'active_learning/index.html', context)


def initialize(request):
    """Initialize the active learning system."""
    if request.method == 'POST':
        pipe_type = request.POST.get('pipe_type', 'document')
        alpha = float(request.POST.get('alpha', 100.0))
        smoothing = float(request.POST.get('smoothing', 1.0))

        # Initialize classifier
        result = call_backend('/initialize', 'POST', {
            'pipe_type': pipe_type,
            'alpha': alpha,
            'smoothing': smoothing
        })

        # Initialize LLM
        llm_result = call_backend('/llm/initialize', 'POST', {
            'model_name': settings.LLM_MODEL,
            'temperature': 0.1,
            'api_key': settings.OPENAI_API_KEY
        })

        if result.get('status') == 'success' and llm_result.get('status') == 'success':
            request.session['pipe_type'] = pipe_type
            request.session['initialized'] = True
            return redirect('load_data')

    context = {
        'pipe_types': ['document', 'lines', 'twitter', 'entity']
    }

    return render(request, 'active_learning/initialize.html', context)


def load_data(request):
    """Load data for active learning."""
    if request.method == 'POST':
        data_path = request.POST.get('data_path')
        data_format = request.POST.get('data_format', 'zip')

        result = call_backend('/load_data', 'POST', {
            'data_path': data_path,
            'data_format': data_format
        })

        if result.get('status') == 'success':
            request.session['data_loaded'] = True
            request.session['labels'] = result.get('labels', [])
            request.session['task_description'] = request.POST.get('task_description', '')
            return redirect('active_learning_loop')

        context = {
            'error': result.get('detail', 'Failed to load data')
        }

    else:
        context = {}

    return render(request, 'active_learning/load_data.html', context)


def active_learning_loop(request):
    """Main active learning loop with LLM labeling."""
    # Get current status
    status = call_backend('/status')

    # Get query parameters
    n_instances = int(request.GET.get('n_instances', 5))
    n_features = int(request.GET.get('n_features', 5))
    strategy = request.GET.get('strategy', 'margin')

    # Generate queries
    queries = call_backend('/query', 'POST', {
        'n_instances': n_instances,
        'strategy': strategy,
        'n_features_per_label': n_features
    })

    context = {
        'status': status,
        'queries': queries,
        'labels': request.session.get('labels', []),
        'task_description': request.session.get('task_description', ''),
        'strategies': ['margin', 'entropy', 'least_confident', 'random']
    }

    return render(request, 'active_learning/learning_loop.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def llm_label_instance(request):
    """Use LLM to label an instance."""
    data = json.loads(request.body)

    instance_id = data.get('instance_id')
    text = data.get('text', '')  # Get text if provided
    labels = data.get('labels', request.session.get('labels', []))
    task_description = data.get('task_description', request.session.get('task_description', ''))

    # If text not provided, we need to fetch it from the query results
    # For now, we'll use a simplified approach

    # Call LLM to label
    llm_result = call_backend('/llm/label_instance', 'POST', {
        'text': text,
        'labels': labels,
        'task_description': task_description,
        'few_shot_examples': data.get('few_shot_examples')
    })

    if llm_result.get('status') == 'success':
        # Add label to classifier
        label_result = call_backend('/label_instance', 'POST', {
            'instance_id': instance_id,
            'label': llm_result['label']
        })

        # Train classifier
        train_result = call_backend('/train', 'POST', {'retrain': False})

        return JsonResponse({
            'status': 'success',
            'label': llm_result['label'],
            'confidence': llm_result['confidence'],
            'reasoning': llm_result['reasoning']
        })

    return JsonResponse(llm_result)


@csrf_exempt
@require_http_methods(["POST"])
def llm_label_feature(request):
    """Use LLM to label a feature."""
    data = json.loads(request.body)

    feature = data.get('feature')
    label = data.get('label')
    task_description = data.get('task_description', request.session.get('task_description', ''))

    # Call LLM to evaluate feature
    llm_result = call_backend('/llm/label_feature', 'POST', {
        'feature': feature,
        'label': label,
        'task_description': task_description,
        'context_examples': data.get('context_examples')
    })

    if llm_result.get('status') == 'success' and llm_result['relevant']:
        # Add feature label to classifier
        feature_result = call_backend('/label_feature', 'POST', {
            'feature': feature,
            'label': label
        })

        # Train classifier
        train_result = call_backend('/train', 'POST', {'retrain': True})

        return JsonResponse({
            'status': 'success',
            'relevant': llm_result['relevant'],
            'confidence': llm_result['confidence'],
            'reasoning': llm_result['reasoning']
        })

    return JsonResponse(llm_result)


@csrf_exempt
@require_http_methods(["POST"])
def llm_auto_label_batch(request):
    """Automatically label a batch of instances using LLM."""
    data = json.loads(request.body)

    queries = data.get('queries', [])
    labels = data.get('labels', request.session.get('labels', []))
    task_description = data.get('task_description', request.session.get('task_description', ''))

    results = []

    for query in queries:
        instance_id = query.get('instance_id')

        # Reconstruct text from features (simplified)
        features = query.get('features', [])
        text = ' '.join(features)  # Simple reconstruction

        # Call LLM
        llm_result = call_backend('/llm/label_instance', 'POST', {
            'text': text,
            'labels': labels,
            'task_description': task_description
        })

        if llm_result.get('status') == 'success':
            # Label instance
            label_result = call_backend('/label_instance', 'POST', {
                'instance_id': instance_id,
                'label': llm_result['label']
            })

            results.append({
                'instance_id': instance_id,
                'label': llm_result['label'],
                'confidence': llm_result['confidence']
            })

    # Retrain after batch
    train_result = call_backend('/train', 'POST', {'retrain': True})

    return JsonResponse({
        'status': 'success',
        'results': results,
        'n_labeled': len(results)
    })


def evaluate(request):
    """Evaluate the current model."""
    result = call_backend('/evaluate', 'POST')

    context = {
        'evaluation': result
    }

    return render(request, 'active_learning/evaluate.html', context)


def predict(request):
    """Make predictions on new texts."""
    if request.method == 'POST':
        texts = request.POST.get('texts', '').split('\n')
        texts = [t.strip() for t in texts if t.strip()]

        result = call_backend('/predict', 'POST', {
            'texts': texts
        })

        context = {
            'predictions': result.get('predictions', []),
            'input_texts': texts
        }

        return render(request, 'active_learning/predict_results.html', context)

    return render(request, 'active_learning/predict.html')
