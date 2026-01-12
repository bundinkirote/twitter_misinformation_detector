"""
Model Evaluation Module

This module provides comprehensive model evaluation and comparison capabilities
including performance metrics calculation, visualization generation, statistical
analysis, and detailed reporting for machine learning model assessment and
selection in misinformation detection tasks.
"""

# Fix for macOS NSWindow threading issue
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score
)
import joblib
from datetime import datetime
from src.utils.file_manager import FileManager
from src.local_model_manager import get_model_manager
from src.feature_extractor import FeatureExtractor
from src.shap_explainer import SHAPExplainer
from src.model_compatibility import get_compatibility_manager

class ModelEvaluator:
    """
    Model Evaluation and Comparison Class
    
    Provides comprehensive evaluation capabilities for machine learning models
    including performance metrics calculation, statistical analysis, visualization
    generation, and comparative assessment. Supports multiple evaluation strategies
    and detailed reporting for model selection and validation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_manager = FileManager()
        self.feature_extractor = FeatureExtractor()
        self.model_manager = get_model_manager()
        self.shap_explainer = SHAPExplainer()
        self.compatibility_manager = get_compatibility_manager()
    
    def get_evaluation_results(self, dataset_name):
        """
        Retrieve comprehensive evaluation results for all trained models.
        
        Args:
            dataset_name: Name of the dataset to evaluate
            
        Returns:
            Dictionary containing detailed evaluation metrics and comparisons
        """
        self.logger.info(f"Getting evaluation results for dataset: {dataset_name}")
        
        try:
            # Load training results
            training_results = self.file_manager.load_results(dataset_name, 'model_training')
            optimization_results = self.file_manager.load_results(dataset_name, 'hyperparameter_optimization')
            
            evaluation_results = {
                'dataset_name': dataset_name,
                'evaluation_date': datetime.now().isoformat(),
                'models': {}
            }
            
            # Combine training and optimization results
            if training_results and 'models' in training_results:
                for model_name, results in training_results['models'].items():
                    if 'error' not in results:
                        evaluation_results['models'][model_name] = {
                            'type': 'baseline',
                            'metrics': results
                        }
            
            if optimization_results and 'models' in optimization_results:
                for model_name, results in optimization_results['models'].items():
                    if 'error' not in results:
                        evaluation_results['models'][f"{model_name}_optimized"] = {
                            'type': 'optimized',
                            'metrics': results
                        }
            
            # Generate visualizations
            self._create_evaluation_visualizations(dataset_name, evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error getting evaluation results: {e}")
            return None
    
    def get_available_models(self, dataset_name: str) -> dict:
        """Get all available trained models with their metrics, prioritizing combined models."""
        try:
            dataset_path = Path(self.file_manager.get_dataset_path(dataset_name))
            models_dir = dataset_path / 'models'
            if not models_dir.exists():
                return {}
            
            available_models = {}
            
            # Check for specialized models first (highest priority)
            specialized_types = ['combined', 'ugt', 'rct', 'rat']
            for framework_type in specialized_types:
                framework_dir = models_dir / framework_type
                if framework_dir.exists():
                    for model_file in framework_dir.glob('*.joblib'):
                        model_name = f"{framework_type}_{model_file.stem}"
                        metrics_path = framework_dir / f'{model_file.stem}_metrics.json'
                        
                        model_info = {
                            'name': model_name, 
                            'path': str(model_file),
                            'type': 'specialized',
                            'framework': framework_type,
                            'priority': 1 if framework_type == 'combined' else 2  # Combined gets highest priority
                        }
                        
                        # Load metrics if available
                        if metrics_path.exists():
                            try:
                                with open(metrics_path, 'r') as f:
                                    metrics = json.load(f)
                                    model_info['metrics'] = metrics
                                    model_info['f1_score'] = metrics.get('f1_score', 0.0)
                                    model_info['accuracy'] = metrics.get('test_accuracy', 0.0)
                            except Exception as e:
                                self.logger.warning(f"Error loading metrics for {model_name}: {e}")
                        
                        available_models[model_name] = model_info
            
            # Check for traditional models (lower priority)
            traditional_types = ['logistic_regression', 'random_forest', 'gradient_boosting', 
                               'svm', 'naive_bayes', 'neural_network', 'zero_shot']
            
            for model_name in traditional_types:
                model_path = models_dir / f'{model_name}.joblib'
                metrics_path = models_dir / f'{model_name}_metrics.json'
                
                if model_path.exists():
                    model_info = {
                        'name': model_name, 
                        'path': str(model_path),
                        'type': 'traditional',
                        'priority': 3  # Lower priority than specialized models
                    }
                    
                    # Load metrics if available
                    if metrics_path.exists():
                        try:
                            with open(metrics_path, 'r') as f:
                                metrics = json.load(f)
                                model_info['metrics'] = metrics
                                model_info['f1_score'] = metrics.get('f1_score', 0.0)
                                model_info['accuracy'] = metrics.get('test_accuracy', 0.0)
                        except Exception as e:
                            self.logger.warning(f"Error loading metrics for {model_name}: {e}")
                    
                    available_models[model_name] = model_info
            
            # Sort by priority (combined models first)
            available_models = dict(sorted(available_models.items(), 
                                         key=lambda x: (x[1].get('priority', 999), -x[1].get('accuracy', 0))))
            
            return available_models
            
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return {}
    
    def get_best_model(self, dataset_name: str) -> str:
        """Get the name of the best performing model, prioritizing combined models."""
        try:
            # First check for specialized training results
            specialized_results_path = Path(self.file_manager.get_dataset_path(dataset_name)) / 'results' / 'specialized_training_results.json'
            if specialized_results_path.exists():
                with open(specialized_results_path, 'r') as f:
                    results = json.load(f)
                    # Look for combined models first
                    if 'model_types' in results and 'combined' in results['model_types']:
                        combined_models = results['model_types']['combined'].get('models', {})
                        if combined_models:
                            # Find best combined model
                            best_combined = max(combined_models.items(), 
                                              key=lambda x: x[1].get('test_accuracy', 0) if isinstance(x[1], dict) else 0)
                            return f"combined_{best_combined[0]}"
            
            # Try to load from traditional training results
            results_path = self.file_manager.get_results_path(dataset_name, 'model_training')
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    return results.get('best_model', {}).get('name', 'logistic_regression')
            
            # Fallback: find best based on available metrics
            models = self.get_available_models(dataset_name)
            if not models:
                return 'logistic_regression'
            
            best_model = max(models.items(), 
                           key=lambda x: x[1].get('f1_score', 0.0))
            return best_model[0]
            
        except Exception as e:
            self.logger.error(f"Error finding best model: {e}")
            return 'logistic_regression'
    
    def load_complete_model(self, dataset_name: str, model_name: str) -> dict:
        """Load complete model with all components using compatibility layer."""
        try:
            model_path_str = self.file_manager.get_model_path(dataset_name, model_name)
            model_path = Path(model_path_str)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Use compatibility manager for safe loading
            self.logger.info(f"Loading model {model_name} with compatibility layer...")
            complete_model = self.compatibility_manager.safe_load_model(model_path)
            
            if complete_model is None:
                raise RuntimeError(f"Failed to load model {model_name} with all strategies")
            
            # Validate model compatibility
            if not self.compatibility_manager.validate_model_compatibility(complete_model):
                self.logger.warning(f"Model {model_name} may have compatibility issues")
            
            # Add model metadata
            if 'model_metadata' not in complete_model:
                complete_model['model_metadata'] = {}
            complete_model['model_metadata']['model_name'] = model_name
            complete_model['model_metadata']['dataset_name'] = dataset_name
            
            # Get expected feature count
            expected_features = self.compatibility_manager.get_model_feature_count(complete_model)
            if expected_features:
                complete_model['model_metadata']['expected_features'] = expected_features
                self.logger.info(f"Model expects {expected_features} features")
            
            return complete_model
                
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            raise

    def predict_with_model(self, dataset_name: str, text: str, model_name: str = None) -> dict:
        """Make enhanced prediction using specified model or best model."""
        try:
            # Determine which model to use
            if not model_name:
                model_name = self.get_best_model(dataset_name)
            
            # Load complete model
            complete_model = self.load_complete_model(dataset_name, model_name)
            model = complete_model['model']
            scaler = complete_model.get('scaler')
            metadata = complete_model.get('model_metadata', {})
            
            # Check if model is in fallback mode
            if complete_model.get('fallback_mode', False):
                self.logger.warning(f"Model {model_name} is in fallback mode, using alternative prediction")
                return self._create_fallback_prediction(text, model_name, metadata)
            
            # Handle zero-shot model separately
            if model_name == 'zero_shot':
                return self.predict_with_zero_shot(dataset_name, text)
            
            # Extract features using the same process as training
            text_features = self.feature_extractor.extract_text_features(dataset_name, [text])
            
            if text_features is None or text_features.shape[0] == 0:
                raise ValueError("Failed to extract features from text")
            
            # Handle feature dimension mismatch
            expected_features = metadata.get('expected_features')
            if expected_features and text_features.shape[1] != expected_features:
                self.logger.warning(f"Feature dimension mismatch: got {text_features.shape[1]}, expected {expected_features}")
                adapter = self.compatibility_manager.create_feature_adapter(expected_features, text_features.shape[1])
                text_features = adapter(text_features)
                self.logger.info(f"Adapted features to shape: {text_features.shape}")
            
            # Use scaler if available
            if scaler is not None:
                try:
                    text_features = scaler.transform(text_features)
                except Exception as e:
                    self.logger.warning(f"Scaler transform failed: {e}, proceeding without scaling")
            
            # Make prediction
            try:
                prediction = model.predict(text_features)[0]
                probability = model.predict_proba(text_features)[0] if hasattr(model, 'predict_proba') else None
            except Exception as e:
                self.logger.error(f"Prediction failed: {e}")
                # Fallback to zero-shot prediction
                return self.predict_with_zero_shot(dataset_name, text)
            confidence = float(max(probability)) if probability is not None else 0.5
            
            # Generate SHAP explanation
            explanation = self._generate_explanation(model, text_features, dataset_name, model_name)
            
            # Get zero-shot prediction for comparison
            zero_shot_result = self.predict_with_zero_shot(dataset_name, text)
            
            # Get fact-check validation
            fact_check_result = self._get_fact_check_validation(text)
            
            # Combine results
            comprehensive_result = {
                'primary_prediction': {
                    'model_name': model_name,
                    'prediction': int(prediction),
                    'prediction_label': 'Misinformation' if prediction == 1 else 'Legitimate',
                    'probability': probability.tolist() if probability is not None else None,
                    'confidence': confidence,
                    'explanation': explanation
                },
                'zero_shot_prediction': zero_shot_result,
                'fact_check_validation': fact_check_result,
                'model_metadata': metadata,
                'consensus_analysis': self._analyze_consensus([
                    prediction, 
                    zero_shot_result.get('prediction', 0),
                    1 if fact_check_result.get('verdict', 'unknown').lower() in ['false', 'misleading'] else 0
                ])
            }
            
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            raise
    
    def _create_fallback_prediction(self, text, model_name, metadata):
        """Create a fallback prediction when model loading fails."""
        try:
            # Use simple keyword-based analysis as fallback
            text_lower = text.lower()
            
            # Misinformation indicators
            misinformation_keywords = [
                'fake', 'hoax', 'conspiracy', 'lie', 'false', 'misleading',
                'unverified', 'rumor', 'scam', 'fraud'
            ]
            
            # Legitimate indicators  
            legitimate_keywords = [
                'official', 'confirmed', 'verified', 'study', 'research',
                'expert', 'scientist', 'government', 'university'
            ]
            
            misinformation_score = sum(1 for keyword in misinformation_keywords if keyword in text_lower)
            legitimate_score = sum(1 for keyword in legitimate_keywords if keyword in text_lower)
            
            # Simple scoring
            if misinformation_score > legitimate_score:
                prediction = 1
                confidence = min(0.6 + (misinformation_score * 0.1), 0.9)
            elif legitimate_score > misinformation_score:
                prediction = 0
                confidence = min(0.6 + (legitimate_score * 0.1), 0.9)
            else:
                prediction = 0  # Default to legitimate when uncertain
                confidence = 0.5
            
            return {
                'primary_prediction': {
                    'model_name': f"{model_name}_fallback",
                    'prediction': prediction,
                    'prediction_label': 'Misinformation' if prediction == 1 else 'Legitimate',
                    'probability': [1-confidence, confidence] if prediction == 1 else [confidence, 1-confidence],
                    'confidence': confidence,
                    'explanation': {'method': 'keyword_analysis_fallback', 'keywords_found': misinformation_score + legitimate_score}
                },
                'zero_shot_prediction': {'prediction': 0, 'confidence': 0.0, 'error': 'Not available in fallback mode'},
                'fact_check_validation': {'verdict': 'unknown', 'confidence': 0.0, 'method': 'fallback'},
                'model_metadata': {
                    **metadata,
                    'fallback_mode': True,
                    'fallback_reason': 'Model loading failed due to compatibility issues'
                },
                'consensus_analysis': {'consensus': 'legitimate' if prediction == 0 else 'misinformation', 'strength': 'weak', 'agreement_ratio': 0.5}
            }
            
        except Exception as e:
            self.logger.error(f"Fallback prediction failed: {e}")
            # Return minimal safe result
            return {
                'primary_prediction': {
                    'model_name': f"{model_name}_minimal_fallback",
                    'prediction': 0,
                    'prediction_label': 'Legitimate',
                    'probability': [0.5, 0.5],
                    'confidence': 0.5,
                    'explanation': {'method': 'minimal_fallback', 'error': str(e)}
                },
                'model_metadata': {'fallback_mode': True, 'error': str(e)},
                'consensus_analysis': {'consensus': 'unknown', 'strength': 'none', 'agreement_ratio': 0.0}
            }
    
    def predict_with_zero_shot(self, dataset_name: str, text: str) -> dict:
        """Make prediction using zero-shot classifier."""
        try:
            # Check if zero-shot model exists
            zs_model_path = self.file_manager.get_model_path(dataset_name, 'zero_shot')
            
            if zs_model_path.exists():
                # Load saved zero-shot model
                zs_model = joblib.load(zs_model_path)
                return {
                    'prediction': 0,  # Placeholder - would need actual implementation
                    'confidence': 0.5,
                    'model_name': zs_model.get('model_name', 'facebook/bart-large-mnli'),
                    'source': 'saved_zero_shot'
                }
            else:
                # Use live zero-shot prediction
                try:
                    from transformers import pipeline
                    classifier = pipeline("zero-shot-classification", 
                                       model="facebook/bart-large-mnli")
                    
                    candidate_labels = ["legitimate news", "misinformation", "fake news"]
                    result = classifier(text, candidate_labels)
                    
                    # Determine prediction
                    top_label = result['labels'][0]
                    top_score = result['scores'][0]
                    
                    prediction = 1 if top_label in ['misinformation', 'fake news'] else 0
                    
                    return {
                        'prediction': prediction,
                        'confidence': float(top_score),
                        'labels': result['labels'],
                        'scores': result['scores'],
                        'source': 'live_zero_shot'
                    }
                    
                except ImportError:
                    self.logger.warning("Transformers not available for zero-shot")
                    return {
                        'prediction': 0,
                        'confidence': 0.0,
                        'error': 'Zero-shot model not available'
                    }
                    
        except Exception as e:
            self.logger.error(f"Error in zero-shot prediction: {e}")
            return {
                'prediction': 0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _generate_explanation(self, model, features, dataset_name, model_name):
        """Generate SHAP explanation for the prediction."""
        try:
            # Load training data for SHAP explainer
            X_train, _, _ = self.feature_extractor.load_features(dataset_name)
            if X_train is not None:
                # Create explainer if not exists
                explainer_name = f"{dataset_name}_{model_name}"
                if explainer_name not in self.shap_explainer.explainers:
                    # Ensure X_train is numpy array, not DataFrame
                    if hasattr(X_train, 'values'):
                        X_train_array = X_train.values
                    else:
                        X_train_array = X_train
                    
                    explainer = self.shap_explainer.create_explainer(model, X_train_array[:100])
                    if explainer:
                        # Store with custom name
                        self.shap_explainer.explainers[explainer_name] = self.shap_explainer.explainers[type(model).__name__]
                
                # Get explanation
                shap_values = self.shap_explainer.explain_prediction(explainer_name, features)
                if shap_values is not None:
                    feature_names = self.feature_extractor.get_feature_names(dataset_name)
                    return self._format_explanation(shap_values[0], feature_names, top_k=10)
        except Exception as e:
            self.logger.warning(f"Could not generate explanation: {e}")
        
        return {'error': 'Explanation not available'}
    
    def _get_fact_check_validation(self, text):
        """Get fact-check validation from external sources."""
        try:
            from src.fact_check_validator import FactCheckValidator
            fact_checker = FactCheckValidator(self.file_manager)
            return fact_checker.validate_with_external_sources(text)
        except Exception as e:
            self.logger.warning(f"Fact-check validation failed: {e}")
            return {'verdict': 'unknown', 'confidence': 0.0, 'error': str(e)}
    
    def _analyze_consensus(self, predictions):
        """Analyze consensus among different prediction methods."""
        misinformation_count = sum(predictions)
        total_predictions = len(predictions)
        
        if misinformation_count >= 2:
            consensus = 'misinformation'
            strength = 'strong' if misinformation_count == total_predictions else 'moderate'
        elif misinformation_count == 1:
            consensus = 'mixed'
            strength = 'weak'
        else:
            consensus = 'legitimate'
            strength = 'strong'
        
        return {
            'consensus': consensus,
            'strength': strength,
            'agreement_ratio': (total_predictions - abs(misinformation_count - (total_predictions - misinformation_count))) / total_predictions
        }
    
    def _format_explanation(self, shap_values, feature_names, top_k=10):
        """Format SHAP explanation for display."""
        try:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(shap_values))]
            
            # Get top features by absolute importance
            importance_indices = np.argsort(np.abs(shap_values))[-top_k:][::-1]
            
            top_features = []
            for idx in importance_indices:
                if idx < len(feature_names):
                    top_features.append({
                        'feature': feature_names[idx],
                        'value': float(shap_values[idx]),
                        'direction': 'SUPPORTS misinformation' if shap_values[idx] > 0 else 'SUPPORTS legitimacy'
                    })
            
            return {'top_features': top_features}
            
        except Exception as e:
            self.logger.error(f"Error formatting explanation: {e}")
            return {'error': 'Could not format explanation'}
    
    def compare_datasets(self):
        """Compare model performance across different datasets."""
        try:
            datasets = self.file_manager.get_all_datasets()
            
            comparison_results = {
                'comparison_date': datetime.now().isoformat(),
                'datasets': {}
            }
            
            for dataset_info in datasets:
                dataset_name = dataset_info['name']
                
                # Get best model performance for each dataset
                optimization_results = self.file_manager.load_results(dataset_name, 'hyperparameter_optimization')
                
                if optimization_results and 'best_model' in optimization_results:
                    best_model = optimization_results['best_model']
                    best_score = optimization_results['best_score']
                    
                    comparison_results['datasets'][dataset_name] = {
                        'best_model': best_model,
                        'best_score': best_score,
                        'dataset_info': dataset_info
                    }
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Error comparing datasets: {e}")
            return None
    
    def get_performance_metrics(self, dataset_name):
        """Get detailed performance metrics for all models."""
        try:
            # Load training results directly - this contains the actual model performance data
            training_results = self.file_manager.load_results(dataset_name, 'model_training')
            
            if not training_results:
                self.logger.warning(f"No training results found for dataset: {dataset_name}")
                return None
            
            # Return training results directly as they have the correct structure
            # The template expects: models (dict), best_model, training_date, etc.
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return None
    
    def _create_evaluation_visualizations(self, dataset_name, evaluation_results):
        """Create evaluation visualizations."""
        try:
            viz_dir = Path('datasets') / dataset_name / 'visualizations'
            
            # Model comparison plot
            self._create_model_comparison_plot(evaluation_results, viz_dir)
            
            # Performance metrics plot
            self._create_performance_metrics_plot(evaluation_results, viz_dir)
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
    
    def _create_model_comparison_plot(self, evaluation_results, viz_dir):
        """Create model comparison plot."""
        try:
            models = []
            scores = []
            types = []
            
            for model_name, results in evaluation_results['models'].items():
                if 'metrics' in results:
                    metrics = results['metrics']
                    
                    if 'test_metrics' in metrics:
                        score = metrics['test_metrics'].get('f1_score', 0)
                    elif 'test_accuracy' in metrics:
                        score = metrics.get('test_accuracy', 0)
                    else:
                        continue
                    
                    models.append(model_name.replace('_', ' ').title())
                    scores.append(score)
                    types.append(results['type'])
            
            if models:
                plt.figure(figsize=(12, 6))
                colors = ['skyblue' if t == 'baseline' else 'lightcoral' for t in types]
                bars = plt.bar(models, scores, color=colors)
                
                plt.title('Model Performance Comparison')
                plt.ylabel('Score')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='skyblue', label='Baseline'),
                    Patch(facecolor='lightcoral', label='Optimized')
                ]
                plt.legend(handles=legend_elements)
                
                plt.tight_layout()
                plt.savefig(viz_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error creating model comparison plot: {e}")
    
    def _create_performance_metrics_plot(self, evaluation_results, viz_dir):
        """Create performance metrics plot."""
        try:
            # Extract metrics for optimized models
            model_names = []
            f1_scores = []
            accuracies = []
            precisions = []
            recalls = []
            
            for model_name, results in evaluation_results['models'].items():
                if results['type'] == 'optimized' and 'metrics' in results:
                    metrics = results['metrics']
                    if 'test_metrics' in metrics:
                        test_metrics = metrics['test_metrics']
                        
                        model_names.append(model_name.replace('_optimized', '').replace('_', ' ').title())
                        f1_scores.append(test_metrics.get('f1_score', 0))
                        accuracies.append(test_metrics.get('accuracy', 0))
                        precisions.append(test_metrics.get('precision', 0))
                        recalls.append(test_metrics.get('recall', 0))
            
            if model_names:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                x = np.arange(len(model_names))
                width = 0.2
                
                ax.bar(x - 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
                ax.bar(x - 0.5*width, accuracies, width, label='Accuracy', alpha=0.8)
                ax.bar(x + 0.5*width, precisions, width, label='Precision', alpha=0.8)
                ax.bar(x + 1.5*width, recalls, width, label='Recall', alpha=0.8)
                
                ax.set_xlabel('Models')
                ax.set_ylabel('Score')
                ax.set_title('Performance Metrics Comparison (Optimized Models)')
                ax.set_xticks(x)
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(viz_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error creating performance metrics plot: {e}")
    
    def get_model_performance_summary(self, dataset_name: str) -> dict:
        """Get streamlined model performance summary with best performer highlighted."""
        try:
            available_models = self.get_available_models(dataset_name)
            best_model_name = self.get_best_model(dataset_name)
            
            if not available_models:
                return {
                    'error': 'No trained models found',
                    'dataset_name': dataset_name,
                    'models_available': 0
                }
            
            # Prepare clean model summaries
            model_summaries = {}
            best_performance = {'f1_score': 0.0, 'accuracy': 0.0}
            
            for model_name, model_info in available_models.items():
                metrics = model_info.get('metrics', {})
                
                # Focus on main metrics only
                summary = {
                    'name': model_name,
                    'display_name': model_name.replace('_', ' ').title(),
                    'f1_score': round(metrics.get('f1_score', 0.0), 3),
                    'accuracy': round(metrics.get('test_accuracy', 0.0), 3),
                    'precision': round(metrics.get('precision', 0.0), 3),
                    'recall': round(metrics.get('recall', 0.0), 3),
                    'is_best': model_name == best_model_name,
                    'model_type': model_name,
                    'live_capable': model_name == 'zero_shot'
                }
                
                # Track best performance
                if summary['f1_score'] > best_performance['f1_score']:
                    best_performance = {
                        'f1_score': summary['f1_score'],
                        'accuracy': summary['accuracy'],
                        'model_name': model_name
                    }
                
                model_summaries[model_name] = summary
            
            return {
                'dataset_name': dataset_name,
                'models_count': len(available_models),
                'best_performer': {
                    'name': best_model_name,
                    'display_name': best_model_name.replace('_', ' ').title(),
                    'f1_score': best_performance['f1_score'],
                    'accuracy': best_performance['accuracy']
                },
                'models': model_summaries,
                'primary_metric': 'F1 Score',
                'zero_shot_available': 'zero_shot' in available_models
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model performance summary: {e}")
            return {
                'error': str(e),
                'dataset_name': dataset_name,
                'models_count': 0
            }