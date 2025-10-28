# Dualist Active Learning with LLM Integration

A modern Python port of the Dualist active learning system, now featuring LLM-powered automated labeling through Langchain. This system combines traditional machine learning with large language models to create an efficient, cost-effective text classification solution.

## 🌟 Overview

This project modernizes the original Java-based Dualist system by:

1. **Porting the backend algorithm to Python** with a FastAPI REST API
2. **Building a Django web interface** for easy interaction
3. **Integrating LLMs via Langchain** to replace human annotation
4. **Maintaining evaluation capabilities** to measure ML performance

### Key Innovation: LLM + ML Synergy

Instead of requiring expensive LLM inference on every document, this system:
- Uses **Active Learning** to select only the most informative instances
- Leverages **LLMs** to label these critical examples
- Trains a **lightweight Naive Bayes classifier** that can run efficiently at scale
- Achieves high accuracy while minimizing LLM API costs

## 📁 Project Structure

```
dualist-revamp/
├── python-backend/              # FastAPI backend
│   ├── app/
│   │   ├── core/               # Core ML algorithms
│   │   │   ├── naive_bayes_with_priors.py    # NB classifier with feature priors
│   │   │   ├── queries.py                     # Active learning strategies
│   │   │   └── llm_labeler.py                 # LLM integration via Langchain
│   │   ├── pipelines/          # Text processing pipelines
│   │   │   └── text_pipelines.py              # Document/Twitter/Entity pipes
│   │   ├── api/                # FastAPI endpoints
│   │   │   ├── main.py                        # Main API routes
│   │   │   └── llm_endpoints.py               # LLM-specific routes
│   │   ├── utils/              # Utilities
│   │   │   └── data_loader.py                 # Data loading utilities
│   │   └── evaluation/         # Evaluation tools
│   │       └── metrics.py                     # Cross-validation, metrics
│   ├── saved_models/           # Trained models
│   └── requirements.txt
│
├── django-frontend/            # Django web interface
│   ├── dualist_ui/            # Django project
│   │   ├── settings.py
│   │   └── urls.py
│   ├── active_learning/       # Main app
│   │   ├── views.py           # View controllers with LLM integration
│   │   ├── urls.py
│   │   └── templates/         # HTML templates
│   ├── manage.py
│   └── requirements.txt
│
├── core/                      # Original Java implementation (for reference)
├── gui/                       # Original web GUI (for reference)
└── data/                      # Sample datasets
    └── baseball-hockey.zip    # Example classification task
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- OpenAI API key (for LLM integration)

### Installation

1. **Clone the repository** (if you haven't already)

```bash
cd /home/user/dualist-revamp
```

2. **Set up the FastAPI backend**

```bash
cd python-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up the Django frontend**

```bash
cd ../django-frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **Set environment variables**

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export FASTAPI_BACKEND_URL="http://localhost:8000"
```

### Running the System

#### Terminal 1: Start the FastAPI Backend

```bash
cd python-backend
source venv/bin/activate
python -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

#### Terminal 2: Start the Django Frontend

```bash
cd django-frontend
source venv/bin/activate
python manage.py migrate  # First time only
python manage.py runserver 0.0.0.0:8080
```

The web interface will be available at `http://localhost:8080`

## 📖 Usage Guide

### 1. Initialize the System

Visit `http://localhost:8080/initialize/`

- **Choose a pipeline type:**
  - `document`: Full documents (news, reviews, articles)
  - `lines`: Short texts, sentences
  - `twitter`: Social media posts with emoticons, mentions, hashtags
  - `entity`: Named entity classification

- **Set parameters:**
  - `alpha`: Feature prior weight (default: 100)
  - `smoothing`: Laplace smoothing (default: 1.0)

### 2. Load Data

Visit `http://localhost:8080/load-data/`

**Example with the included dataset:**

```
Data Path: /home/user/dualist-revamp/data/baseball-hockey.zip
Data Format: ZIP Archive
Task Description: Classify sports articles into baseball or hockey
```

**Supported formats:**
- ZIP archives with folder-based labels
- Directories with folder-based labels
- JSON files: `[{"text": "...", "label": "..."}, ...]`
- CSV files with text and label columns

### 3. Active Learning Loop

Visit `http://localhost:8080/learn/`

This is the main interface where the magic happens:

**Instance Queries:**
- System selects most informative unlabeled instances
- Click "🤖 Ask LLM to Label" for individual instances
- Or "🤖 Auto-Label All with LLM" for batch processing

**Feature Queries:**
- System suggests discriminative terms for each class
- LLM evaluates which features are truly indicative
- These feature labels improve the classifier

**Query Strategies:**
- `margin`: Select instances closest to decision boundary (default)
- `entropy`: Select instances with highest uncertainty
- `least_confident`: Select least confident predictions
- `random`: Random sampling (baseline)

### 4. Evaluate the Model

Visit `http://localhost:8080/evaluate/`

See performance metrics:
- Accuracy
- Precision, Recall, F1 (macro and per-label)
- Tested on remaining unlabeled pool

### 5. Make Predictions

Visit `http://localhost:8080/predict/`

Classify new texts using the trained model. Input one text per line.

## 🔧 API Endpoints

### FastAPI Backend (`http://localhost:8000`)

#### Core Endpoints

- `POST /initialize` - Initialize classifier and pipeline
- `POST /load_data` - Load training data
- `POST /query` - Generate active learning queries
- `POST /label_instance` - Label an instance
- `POST /label_feature` - Add feature prior
- `POST /train` - Train/retrain classifier
- `POST /predict` - Classify new texts
- `POST /evaluate` - Evaluate model
- `GET /status` - System status

#### LLM Endpoints

- `POST /llm/initialize` - Initialize LLM labeler
- `POST /llm/label_instance` - LLM labels an instance
- `POST /llm/label_feature` - LLM evaluates a feature
- `POST /llm/label_batch` - LLM labels multiple instances
- `POST /llm/generate_features` - LLM suggests initial features

Full API documentation: `http://localhost:8000/docs`

## 🧠 How It Works

### The Active Learning Algorithm

1. **Initialization:**
   - Load unlabeled data pool
   - Optionally: LLM generates initial feature labels (cold start)

2. **Active Learning Loop:**
   ```
   WHILE pool not empty:
       1. Train Naive Bayes classifier on labeled data
       2. Query most uncertain instances (uncertainty sampling)
       3. Query discriminative features (mutual information)
       4. LLM labels instances and features
       5. Add to training set
       6. Repeat
   ```

3. **Query Strategies:**
   - **Margin:** `margin = |P(top_label) - P(second_label)|` (smaller = more uncertain)
   - **Entropy:** `entropy = -Σ P(label) * log(P(label))` (higher = more uncertain)
   - **Feature MI:** Mutual information between feature and label

### The Classifier: Naive Bayes with Priors

The classifier is a custom Multinomial Naive Bayes with feature priors:

```
P(label|document) ∝ P(label) × ∏ P(feature|label)^count

Where:
- P(feature|label) includes feature priors as pseudocounts
- Laplace smoothing handles rare/unseen features
- Incremental learning updates model without full retraining
```

**Advantages:**
- Fast training and inference
- Works well with limited labeled data
- Incorporates expert knowledge via feature priors
- Probabilistic output (confidence scores)

### LLM Integration

The system uses Langchain to interface with LLMs (default: GPT-3.5-turbo):

**For Instance Labeling:**
```python
PROMPT: You are a classifier for {task_description}
        Labels: {label1, label2, ...}
        Classify this text: {text}
        Respond with JSON: {"label": "...", "confidence": 0.x, "reasoning": "..."}
```

**For Feature Labeling:**
```python
PROMPT: Is the word "{feature}" relevant and indicative of class "{label}"
        for task: {task_description}?
        Respond with JSON: {"relevant": true/false, "confidence": 0.x, "reasoning": "..."}
```

**Cost Optimization:**
- Only label ~10-50 instances per iteration (not entire dataset)
- Use lightweight NB classifier for bulk predictions
- Combine instance + feature labeling for better priors

## 📊 Evaluation

### Running ML Evaluation Scripts

The original evaluation capabilities are preserved:

**Cross-Validation:**
```python
from app.evaluation.metrics import ModelEvaluator
from app.core.naive_bayes_with_priors import NaiveBayesWithPriors

# 10-fold cross-validation
cv_results = ModelEvaluator.cross_validate(
    X, y,
    classifier_class=NaiveBayesWithPriors,
    classifier_params={'alpha': 100.0},
    n_folds=10
)

# Print summary
summary = ModelEvaluator.summarize_cv_results(cv_results)
print(f"Accuracy: {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")
```

**Active Learning Curves:**
```python
from app.evaluation.metrics import ActiveLearningEvaluator

results = ActiveLearningEvaluator.evaluate_learning_curve(
    classifier, X_pool, y_pool, X_test, y_test,
    query_strategy='margin',
    n_iterations=20,
    n_queries_per_iteration=10
)

# Plot results
ActiveLearningEvaluator.plot_learning_curve(results, 'learning_curve.png')
```

## 🎯 Example Workflow

### Baseball vs Hockey Classification

```bash
# 1. Initialize with document pipeline
curl -X POST http://localhost:8000/initialize \
  -H "Content-Type: application/json" \
  -d '{"pipe_type": "document", "alpha": 100.0}'

# 2. Load data
curl -X POST http://localhost:8000/load_data \
  -H "Content-Type: application/json" \
  -d '{"data_path": "/home/user/dualist-revamp/data/baseball-hockey.zip", "data_format": "zip"}'

# 3. Initialize LLM
curl -X POST http://localhost:8000/llm/initialize \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt-3.5-turbo", "api_key": "your-key"}'

# 4. Generate queries
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"n_instances": 5, "strategy": "margin"}'

# 5. LLM labels instance
curl -X POST http://localhost:8000/llm/label_instance \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The pitcher threw a curveball for strike three",
    "labels": ["baseball", "hockey"],
    "task_description": "Classify sports articles"
  }'

# 6. Add label and train
curl -X POST http://localhost:8000/label_instance \
  -H "Content-Type: application/json" \
  -d '{"instance_id": "doc_001", "label": "baseball"}'

curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"retrain": false}'

# Repeat 4-6 for several iterations...

# 7. Evaluate
curl -X POST http://localhost:8000/evaluate

# 8. Predict new instances
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["The defenseman scored in overtime"]}'
```

## 🔬 Advanced Configuration

### Custom LLM Prompts

Edit `/home/user/dualist-revamp/python-backend/app/core/llm_labeler.py`:

```python
def _build_instance_labeling_system_prompt(self, labels, task_description):
    # Customize your prompt here
    prompt = f"""Your custom instructions...
    TASK: {task_description}
    LABELS: {labels}
    """
    return prompt
```

### Custom Text Pipelines

Add a new pipeline in `/home/user/dualist-revamp/python-backend/app/pipelines/text_pipelines.py`:

```python
class CustomPipe(BasePipe):
    def process(self, text: str) -> Dict[str, int]:
        # Your custom preprocessing
        features = {}
        # ... extract features
        return features
```

### Model Persistence

```python
# Save trained model
classifier.save('/path/to/model.pkl')

# Load model
from app.core.naive_bayes_with_priors import NaiveBayesWithPriors
classifier = NaiveBayesWithPriors.load('/path/to/model.pkl')
```

## 🐛 Troubleshooting

### Backend Connection Issues

```bash
# Check if FastAPI is running
curl http://localhost:8000/

# Check logs
# In backend terminal, look for errors
```

### LLM Labeling Fails

```python
# Check API key
echo $OPENAI_API_KEY

# Test LLM directly
curl -X POST http://localhost:8000/llm/initialize \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt-3.5-turbo", "api_key": "your-key"}'
```

### Import Errors

```bash
# Make sure __init__.py files exist
find python-backend/app -name "__init__.py"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📚 Theory & Research

### Active Learning

Active learning is a semi-supervised learning paradigm where the algorithm can query an oracle (human or LLM) for labels on selected unlabeled instances. The goal is to minimize labeling costs while maximizing model performance.

**Key concepts:**
- **Uncertainty Sampling:** Query instances where the model is most uncertain
- **Query by Committee:** Use disagreement among ensemble models
- **Expected Model Change:** Query instances that would most change the model

### Dual Supervision

This system implements **dual supervision** from Settles (2011):
- **Instance labeling:** Traditional supervision (label documents)
- **Feature labeling:** Rationales (identify discriminative terms)

Feature labeling provides **prior knowledge** that guides the classifier, especially useful with limited labeled data.

### References

- Settles, B. (2011). "Closing the Loop: Fast, Interactive Semi-Supervised Annotation with Queries on Features and Instances." *EMNLP 2011*.
- Lewis, D. D., & Gale, W. A. (1994). "A Sequential Algorithm for Training Text Classifiers." *SIGIR 1994*.
- McCallum, A., & Nigam, K. (1998). "A Comparison of Event Models for Naive Bayes Text Classification." *AAAI-98 Workshop*.

## 🤝 Contributing

This is a research/demo system. For production use, consider:
- Proper state management (Redis, database)
- Authentication and authorization
- Rate limiting for LLM APIs
- Async processing for long-running tasks
- More sophisticated active learning strategies

## 📄 License

This project extends the original Dualist system. See the original license in the `core/` directory.

## 🙏 Acknowledgments

- Original Dualist system by Burr Settles (2011)
- MALLET toolkit for the Java implementation
- FastAPI, Django, Langchain, and OpenAI for the modern stack

---

**Built with ❤️ for efficient text classification** 🚀
