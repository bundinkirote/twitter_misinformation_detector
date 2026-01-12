# API Documentation

## Overview

The Twitter Misinformation Detection System provides a comprehensive web API for programmatic access to all system functionality. This document describes the available endpoints, request/response formats, and usage examples.

## Base URL

```
http://localhost:5000
```

## Authentication

The current implementation does NOT include user authentication or API keys. The system is designed for local research/academic use in a single-user or trusted network environment. All endpoints are publicly accessible without authentication. For production deployment, implement proper authentication layer before exposing to untrusted networks.

## Core Endpoints

### Dataset Management

#### Upload Dataset
```http
POST /upload
Content-Type: multipart/form-data

Parameters:
- dataset: File (CSV or Excel)
- dataset_name: String (optional)
```

**Response:**
```json
{
  "status": "success",
  "message": "Dataset uploaded successfully",
  "dataset_name": "dataset_20250105_123456",
  "redirect_url": "/dataset/dataset_20250105_123456"
}
```

#### Get Dataset Overview
```http
GET /dataset/{dataset_name}
```

**Response:**
```json
{
  "dataset_name": "example_dataset",
  "stats": {
    "total_samples": 1000,
    "features": 15,
    "misinformation_rate": 23.5,
    "missing_values": 12,
    "has_labels": true
  },
  "insights": [
    {
      "type": "quality",
      "message": "Dataset is well-balanced with good quality indicators",
      "confidence": 0.85
    }
  ]
}
```

### Data Processing

#### Process Dataset
```http
POST /api/process/{dataset_name}
```

**Response:**
```json
{
  "status": "success",
  "processing_time": 45.2,
  "processed_samples": 1000,
  "columns_mapped": {
    "text": "CONTENT",
    "label": "LABEL",
    "user_id": "USER_ID"
  }
}
```

### Feature Extraction

#### Extract Features
```http
POST /api/features/extract/{dataset_name}
Content-Type: application/json

{
  "feature_types": ["text", "behavioral", "network"],
  "text_features": {
    "tfidf": true,
    "sentiment": true,
    "embeddings": true
  },
  "behavioral_features": {
    "engagement": true,
    "temporal": true,
    "user_profile": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "extraction_time": 120.5,
  "features_extracted": {
    "text_features": 500,
    "behavioral_features": 25,
    "network_features": 15,
    "total_features": 540
  },
  "feature_importance": {
    "top_features": [
      {"name": "sentiment_score", "importance": 0.15},
      {"name": "retweet_count", "importance": 0.12},
      {"name": "tfidf_misinformation", "importance": 0.11}
    ]
  }
}
```

#### Get Feature Status
```http
GET /api/features/status/{dataset_name}
```

**Response:**
```json
{
  "status": "completed",
  "features_available": true,
  "feature_count": 540,
  "extraction_date": "2025-01-05T12:34:56Z",
  "feature_types": {
    "text": 500,
    "behavioral": 25,
    "network": 15
  }
}
```

### Model Training

#### Train Models
```http
POST /api/models/train/{dataset_name}
Content-Type: application/json

{
  "algorithms": ["logistic_regression", "random_forest", "naive_bayes"],
  "test_size": 0.2,
  "cross_validation": true,
  "hyperparameter_tuning": false
}
```

**Response:**
```json
{
  "status": "success",
  "training_time": 180.3,
  "models_trained": 3,
  "results": {
    "logistic_regression": {
      "accuracy": 0.89,
      "f1_score": 0.87,
      "roc_auc": 0.91,
      "training_time": 45.2
    },
    "random_forest": {
      "accuracy": 0.92,
      "f1_score": 0.90,
      "roc_auc": 0.94,
      "training_time": 89.1
    },
    "naive_bayes": {
      "accuracy": 0.85,
      "f1_score": 0.83,
      "roc_auc": 0.88,
      "training_time": 12.5
    }
  },
  "best_model": "random_forest"
}
```

#### Get Training Status
```http
GET /api/models/status/{dataset_name}
```

**Response:**
```json
{
  "status": "completed",
  "models_available": ["logistic_regression", "random_forest", "naive_bayes"],
  "best_model": "random_forest",
  "training_date": "2025-01-05T14:22:33Z",
  "performance_summary": {
    "best_accuracy": 0.92,
    "best_f1": 0.90,
    "best_roc_auc": 0.94
  }
}
```

### Predictions

#### Make Prediction
```http
POST /api/predict/{dataset_name}
Content-Type: application/json

{
  "text": "This is a sample tweet to classify",
  "model": "random_forest",
  "include_explanation": true
}
```

**Response:**
```json
{
  "prediction": {
    "class": 1,
    "probability": 0.78,
    "confidence": "high"
  },
  "explanation": {
    "top_features": [
      {"feature": "sentiment_negative", "contribution": 0.25},
      {"feature": "tfidf_fake", "contribution": 0.18},
      {"feature": "exclamation_count", "contribution": 0.12}
    ]
  },
  "processing_time": 0.15
}
```

#### Batch Predictions
```http
POST /api/predict/batch/{dataset_name}
Content-Type: application/json

{
  "texts": [
    "First tweet to classify",
    "Second tweet to classify",
    "Third tweet to classify"
  ],
  "model": "random_forest"
}
```

**Response:**
```json
{
  "predictions": [
    {"text": "First tweet...", "class": 0, "probability": 0.23},
    {"text": "Second tweet...", "class": 1, "probability": 0.89},
    {"text": "Third tweet...", "class": 0, "probability": 0.15}
  ],
  "summary": {
    "total_predictions": 3,
    "misinformation_detected": 1,
    "average_confidence": 0.42
  },
  "processing_time": 0.45
}
```

### Analysis Endpoints

#### Language Detection
```http
POST /api/analysis/language/{dataset_name}
```

**Response:**
```json
{
  "status": "success",
  "language_distribution": {
    "en": 0.75,
    "es": 0.15,
    "fr": 0.08,
    "other": 0.02
  },
  "dominant_language": "en",
  "multilingual_content": 0.25
}
```

#### Sentiment Analysis
```http
POST /api/analysis/sentiment/{dataset_name}
```

**Response:**
```json
{
  "status": "success",
  "sentiment_distribution": {
    "positive": 0.35,
    "negative": 0.45,
    "neutral": 0.20
  },
  "average_sentiment": -0.12,
  "emotion_analysis": {
    "anger": 0.25,
    "fear": 0.18,
    "joy": 0.15,
    "sadness": 0.12,
    "surprise": 0.10,
    "disgust": 0.20
  }
}
```

#### Network Analysis
```http
POST /api/analysis/network/{dataset_name}
```

**Response:**
```json
{
  "status": "success",
  "network_metrics": {
    "total_users": 500,
    "total_connections": 1250,
    "density": 0.005,
    "clustering_coefficient": 0.35,
    "average_path_length": 3.2
  },
  "communities": {
    "count": 8,
    "modularity": 0.42,
    "largest_community_size": 125
  },
  "influential_users": [
    {"user_id": "user123", "centrality": 0.15, "influence_score": 0.89},
    {"user_id": "user456", "centrality": 0.12, "influence_score": 0.76}
  ]
}
```

### Zero-Shot Classification

#### Zero-Shot Labeling
```http
POST /api/zero-shot/classify/{dataset_name}
Content-Type: application/json

{
  "labels": ["misinformation", "legitimate", "satire"],
  "hypothesis_template": "This text is {}",
  "batch_size": 100
}
```

**Response:**
```json
{
  "status": "success",
  "classification_results": {
    "total_classified": 1000,
    "label_distribution": {
      "misinformation": 0.23,
      "legitimate": 0.65,
      "satire": 0.12
    },
    "average_confidence": 0.78
  },
  "processing_time": 45.6
}
```

### Explainability

#### Get Model Explanations
```http
GET /api/explain/{dataset_name}/{model_name}
```

**Response:**
```json
{
  "model": "random_forest",
  "global_explanations": {
    "feature_importance": [
      {"feature": "sentiment_score", "importance": 0.15},
      {"feature": "retweet_count", "importance": 0.12},
      {"feature": "tfidf_misinformation", "importance": 0.11}
    ]
  },
  "shap_values": {
    "available": true,
    "summary_plot": "/static/plots/shap_summary.png"
  }
}
```

#### Explain Individual Prediction
```http
POST /api/explain/prediction/{dataset_name}
Content-Type: application/json

{
  "text": "Sample tweet text",
  "model": "random_forest"
}
```

**Response:**
```json
{
  "prediction": {
    "class": 1,
    "probability": 0.78
  },
  "explanation": {
    "shap_values": [
      {"feature": "sentiment_negative", "value": 0.25},
      {"feature": "tfidf_fake", "value": 0.18},
      {"feature": "exclamation_count", "value": 0.12}
    ],
    "base_value": 0.23,
    "explanation_plot": "/static/plots/explanation_123.png"
  }
}
```

### Visualization

#### Generate Visualizations
```http
POST /api/visualizations/generate/{dataset_name}
Content-Type: application/json

{
  "visualization_types": ["confusion_matrix", "roc_curve", "feature_importance"],
  "models": ["random_forest", "logistic_regression"]
}
```

**Response:**
```json
{
  "status": "success",
  "visualizations": {
    "confusion_matrix": "/static/plots/confusion_matrix.png",
    "roc_curve": "/static/plots/roc_curves.png",
    "feature_importance": "/static/plots/feature_importance.png"
  },
  "generation_time": 12.3
}
```

## Error Handling

### Error Response Format
```json
{
  "status": "error",
  "error_code": "DATASET_NOT_FOUND",
  "message": "The specified dataset was not found",
  "details": {
    "dataset_name": "invalid_dataset",
    "available_datasets": ["dataset1", "dataset2"]
  }
}
```

### Common Error Codes

- `DATASET_NOT_FOUND`: Dataset does not exist
- `INVALID_FORMAT`: Unsupported file format
- `PROCESSING_ERROR`: Error during data processing
- `MODEL_NOT_TRAINED`: Requested model not available
- `INSUFFICIENT_DATA`: Not enough data for operation
- `FEATURE_EXTRACTION_FAILED`: Feature extraction error
- `PREDICTION_ERROR`: Error during prediction

## Rate Limiting

Currently, no rate limiting is implemented for local deployment. For production deployment, consider implementing rate limiting based on your requirements.

## WebSocket Endpoints

### Real-time Training Progress
```javascript
const socket = new WebSocket('ws://localhost:5000/ws/training/{dataset_name}');

socket.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Training progress:', data.progress);
};
```

### Live Predictions
```javascript
const socket = new WebSocket('ws://localhost:5000/ws/predictions/{dataset_name}');

socket.send(JSON.stringify({
  'text': 'Tweet to classify',
  'model': 'random_forest'
}));
```

## SDK Examples

### Python SDK Example
```python
import requests
import json

class MisinformationDetectionAPI:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def upload_dataset(self, file_path, dataset_name=None):
        with open(file_path, 'rb') as f:
            files = {'dataset': f}
            data = {'dataset_name': dataset_name} if dataset_name else {}
            response = self.session.post(f"{self.base_url}/upload", 
                                       files=files, data=data)
        return response.json()
    
    def train_models(self, dataset_name, algorithms=None):
        if algorithms is None:
            algorithms = ["logistic_regression", "random_forest"]
        
        data = {"algorithms": algorithms}
        response = self.session.post(f"{self.base_url}/api/models/train/{dataset_name}",
                                   json=data)
        return response.json()
    
    def predict(self, dataset_name, text, model="random_forest"):
        data = {"text": text, "model": model}
        response = self.session.post(f"{self.base_url}/api/predict/{dataset_name}",
                                   json=data)
        return response.json()

# Usage
api = MisinformationDetectionAPI()
result = api.upload_dataset("my_dataset.csv", "test_dataset")
training_result = api.train_models("test_dataset")
prediction = api.predict("test_dataset", "This is a sample tweet")
```

### JavaScript SDK Example
```javascript
class MisinformationDetectionAPI {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
    }
    
    async uploadDataset(file, datasetName = null) {
        const formData = new FormData();
        formData.append('dataset', file);
        if (datasetName) {
            formData.append('dataset_name', datasetName);
        }
        
        const response = await fetch(`${this.baseUrl}/upload`, {
            method: 'POST',
            body: formData
        });
        return await response.json();
    }
    
    async trainModels(datasetName, algorithms = ['logistic_regression', 'random_forest']) {
        const response = await fetch(`${this.baseUrl}/api/models/train/${datasetName}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ algorithms })
        });
        return await response.json();
    }
    
    async predict(datasetName, text, model = 'random_forest') {
        const response = await fetch(`${this.baseUrl}/api/predict/${datasetName}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text, model })
        });
        return await response.json();
    }
}

// Usage
const api = new MisinformationDetectionAPI();
const result = await api.predict('my_dataset', 'This is a sample tweet');
```

## Testing the API

### Using curl
```bash
# Upload dataset
curl -X POST -F "dataset=@sample_data.csv" -F "dataset_name=test" \
     http://localhost:5000/upload

# Train models
curl -X POST -H "Content-Type: application/json" \
     -d '{"algorithms": ["random_forest", "logistic_regression"]}' \
     http://localhost:5000/api/models/train/test

# Make prediction
curl -X POST -H "Content-Type: application/json" \
     -d '{"text": "This is a sample tweet", "model": "random_forest"}' \
     http://localhost:5000/api/predict/test
```

### Using Postman
Import the provided Postman collection (`api_collection.json`) for easy testing of all endpoints.

## WebSocket Endpoints (Planned for Future Implementation)

Real-time endpoints for training progress and live predictions are planned for future versions but are not currently available. Current implementation uses HTTP-based requests only.

## API Versioning

Current API version: v1
All endpoints are prefixed with `/api/` for programmatic access.
Web interface endpoints do not use the `/api/` prefix.

## Support

For API support and questions:
1. Check the application logs for detailed error information
2. Verify that all required parameters are provided
3. Ensure the dataset exists and is properly processed
4. Check the system status endpoints for service availability