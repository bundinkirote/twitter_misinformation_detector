"""
SHAP Explainability Module

This module provides comprehensive model interpretability using SHAP (SHapley Additive
exPlanations) for machine learning model explanation. It implements multiple explainer
types, feature importance analysis, and visualization capabilities for transparent
and interpretable misinformation detection model analysis.
"""

import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')
from src.model_compatibility import get_compatibility_manager

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not available. Install with: pip install shap")

class SHAPExplainer:
    """
    SHAP-Based Model Explainability Class
    
    Implements comprehensive model interpretability using SHAP (SHapley Additive
    exPlanations) methodology. Provides multiple explainer types, feature importance
    analysis, and visualization capabilities for transparent machine learning model
    explanation and interpretation in misinformation detection tasks.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.explainers = {}
        self.shap_values_cache = {}
        self.compatibility_manager = get_compatibility_manager()
        
    def create_explainer(self, model, X_train, model_type='auto', feature_names=None):
        """
        Create SHAP explainer for the enhanced model with transformer + framework features.
        
        Args:
            model: Trained model (traditional or enhanced)
            X_train: Training data for background
            model_type: Type of explainer ('linear', 'tree', 'kernel', 'auto')
            feature_names: Names of features (including transformer + framework features)
        """
        if not SHAP_AVAILABLE:
            self.logger.error("SHAP not available")
            return None
            
        try:
            model_name = type(model).__name__
            self.logger.info(f"Creating SHAP explainer for {model_name} with {X_train.shape[1]} features")
            
            # Store feature names for enhanced interpretation
            self.feature_names = feature_names if feature_names is not None else [f'feature_{i}' for i in range(X_train.shape[1])]
            self.enhanced_features_info = self._analyze_feature_types(self.feature_names)
            
            # Ensure X_train is numpy array, not DataFrame
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            elif hasattr(X_train, 'values'):
                X_train = X_train.values
            elif hasattr(X_train, 'to_numpy'):
                X_train = X_train.to_numpy()
            
            # Ensure it's a numpy array
            if not isinstance(X_train, np.ndarray):
                X_train = np.array(X_train)
            
            # Auto-detect explainer type
            if model_type == 'auto':
                if hasattr(model, 'coef_'):  # Linear models
                    explainer = shap.LinearExplainer(model, X_train)
                    explainer_type = 'linear'
                elif hasattr(model, 'tree_'):  # Tree-based models
                    explainer = shap.TreeExplainer(model)
                    explainer_type = 'tree'
                else:  # General models
                    # Use a sample for kernel explainer to avoid memory issues
                    background_sample = shap.sample(X_train, min(100, len(X_train)))
                    explainer = shap.KernelExplainer(model.predict_proba, background_sample)
                    explainer_type = 'kernel'
            else:
                if model_type == 'linear':
                    explainer = shap.LinearExplainer(model, X_train)
                elif model_type == 'tree':
                    explainer = shap.TreeExplainer(model)
                elif model_type == 'kernel':
                    background_sample = shap.sample(X_train, min(100, len(X_train)))
                    explainer = shap.KernelExplainer(model.predict_proba, background_sample)
                else:
                    raise ValueError(f"Unknown explainer type: {model_type}")
                explainer_type = model_type
            
            self.explainers[model_name] = {
                'explainer': explainer,
                'type': explainer_type,
                'model': model
            }
            
            self.logger.info(f"SUCCESS: Created {explainer_type} explainer for {model_name}")
            return explainer
            
        except Exception as e:
            self.logger.error(f"Error creating SHAP explainer: {e}")
            return None
    
    def explain_predictions(self, model_name, X_test, max_samples=100):
        """
        Generate SHAP values for predictions.
        
        Args:
            model_name: Name of the model
            X_test: Test data
            max_samples: Maximum samples to explain (for performance)
        """
        if model_name not in self.explainers:
            self.logger.error(f"No explainer found for model: {model_name}")
            return None
            
        try:
            explainer_info = self.explainers[model_name]
            explainer = explainer_info['explainer']
            explainer_type = explainer_info['type']
            
            # Limit samples for performance
            if len(X_test) > max_samples:
                sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
                X_sample = X_test[sample_indices]
                self.logger.info(f"Explaining {max_samples} random samples out of {len(X_test)}")
            else:
                X_sample = X_test
                sample_indices = np.arange(len(X_test))
            
            # Generate SHAP values
            self.logger.info(f"Generating SHAP values using {explainer_type} explainer...")
            
            if explainer_type == 'linear':
                shap_values = explainer.shap_values(X_sample)
            elif explainer_type == 'tree':
                shap_values = explainer.shap_values(X_sample)
                # For binary classification, take the positive class
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
            else:  # kernel
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
            
            # Cache results
            self.shap_values_cache[model_name] = {
                'shap_values': shap_values,
                'X_sample': X_sample,
                'sample_indices': sample_indices,
                'explainer': explainer
            }
            
            self.logger.info(f"SUCCESS: Generated SHAP values for {model_name}")
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP values: {e}")
            return None
    
    def plot_feature_importance(self, model_name, feature_names=None, max_features=20, save_path=None):
        """
        Plot SHAP feature importance.
        
        Args:
            model_name: Name of the model
            feature_names: Names of features
            max_features: Maximum features to show
            save_path: Path to save the plot
        """
        if model_name not in self.shap_values_cache:
            self.logger.error(f"No SHAP values found for model: {model_name}")
            return None
            
        try:
            cache_data = self.shap_values_cache[model_name]
            shap_values = cache_data['shap_values']
            X_sample = cache_data['X_sample']
            
            plt.figure(figsize=(10, 8))
            
            # Create feature importance plot
            shap.summary_plot(
                shap_values, X_sample, 
                feature_names=feature_names,
                max_display=max_features,
                show=False
            )
            
            plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"SHAP plot saved to: {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            self.logger.error(f"Error plotting SHAP feature importance: {e}")
            return None
    
    def plot_waterfall(self, model_name, sample_idx=0, feature_names=None, save_path=None):
        """
        Plot SHAP waterfall chart for a single prediction.
        
        Args:
            model_name: Name of the model
            sample_idx: Index of sample to explain
            feature_names: Names of features
            save_path: Path to save the plot
        """
        if model_name not in self.shap_values_cache:
            self.logger.error(f"No SHAP values found for model: {model_name}")
            return None
            
        try:
            cache_data = self.shap_values_cache[model_name]
            shap_values = cache_data['shap_values']
            X_sample = cache_data['X_sample']
            explainer = cache_data['explainer']
            
            if sample_idx >= len(shap_values):
                sample_idx = 0
                
            plt.figure(figsize=(10, 6))
            
            # Create waterfall plot
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                    data=X_sample[sample_idx],
                    feature_names=feature_names
                ),
                show=False
            )
            
            plt.title(f'SHAP Waterfall - {model_name} (Sample {sample_idx})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"SHAP waterfall plot saved to: {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            self.logger.error(f"Error plotting SHAP waterfall: {e}")
            return None
    
    def get_top_features(self, model_name, n_features=10):
        """
        Get top contributing features based on SHAP values.
        
        Args:
            model_name: Name of the model
            n_features: Number of top features to return
        """
        if model_name not in self.shap_values_cache:
            self.logger.error(f"No SHAP values found for model: {model_name}")
            return None
            
        try:
            cache_data = self.shap_values_cache[model_name]
            shap_values = cache_data['shap_values']
            
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Get top features
            top_indices = np.argsort(mean_shap)[-n_features:][::-1]
            top_values = mean_shap[top_indices]
            
            return {
                'indices': top_indices,
                'values': top_values,
                'importance_scores': top_values / np.sum(top_values)  # Normalized
            }
            
        except Exception as e:
            self.logger.error(f"Error getting top features: {e}")
            return None
    
    def explain_single_prediction(self, model_name, sample, feature_names=None):
        """
        Explain a single prediction with detailed breakdown.
        
        Args:
            model_name: Name of the model
            sample: Single sample to explain
            feature_names: Names of features
        """
        if model_name not in self.explainers:
            self.logger.error(f"No explainer found for model: {model_name}")
            return None
            
        try:
            explainer_info = self.explainers[model_name]
            explainer = explainer_info['explainer']
            model = explainer_info['model']
            
            # Reshape sample if needed
            if len(sample.shape) == 1:
                sample = sample.reshape(1, -1)
            
            # Get prediction
            prediction = model.predict(sample)[0]
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(sample)[0]
            else:
                probability = [1-prediction, prediction]
            
            # Get SHAP values
            if explainer_info['type'] == 'linear':
                shap_vals = explainer.shap_values(sample)[0]
            else:
                shap_vals = explainer.shap_values(sample)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1][0]  # Positive class
                else:
                    shap_vals = shap_vals[0]
            
            # Create explanation
            explanation = {
                'prediction': int(prediction),
                'probability': {
                    'legitimate': float(probability[0]),
                    'misinformation': float(probability[1])
                },
                'confidence': float(max(probability)),
                'shap_values': shap_vals.tolist(),
                'feature_contributions': []
            }
            
            # Add feature contributions
            if feature_names:
                for i, (feature, shap_val) in enumerate(zip(feature_names, shap_vals)):
                    explanation['feature_contributions'].append({
                        'feature': feature,
                        'value': float(sample[0][i]),
                        'shap_value': float(shap_val),
                        'contribution': 'positive' if shap_val > 0 else 'negative'
                    })
                
                # Sort by absolute SHAP value
                explanation['feature_contributions'].sort(
                    key=lambda x: abs(x['shap_value']), reverse=True
                )
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining single prediction: {e}")
            return None
    
    def save_explainer(self, model_name, save_path):
        """Save SHAP explainer and values."""
        if model_name not in self.explainers:
            self.logger.error(f"No explainer found for model: {model_name}")
            return False
            
        try:
            save_data = {
                'explainer_info': self.explainers[model_name],
                'shap_values_cache': self.shap_values_cache.get(model_name, {})
            }
            
            joblib.dump(save_data, save_path)
            self.logger.info(f"SHAP explainer saved to: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving SHAP explainer: {e}")
            return False
    
    def load_explainer(self, model_name, load_path):
        """Load SHAP explainer and values."""
        try:
            # Use compatibility manager for safe loading
            save_data_dict = self.compatibility_manager.safe_load_model(load_path)
            if save_data_dict and 'model' in save_data_dict:
                save_data = save_data_dict['model']
            elif save_data_dict:
                save_data = save_data_dict
            else:
                save_data = joblib.load(load_path)
            
            self.explainers[model_name] = save_data['explainer_info']
            if save_data['shap_values_cache']:
                self.shap_values_cache[model_name] = save_data['shap_values_cache']
            
            self.logger.info(f"SHAP explainer loaded from: {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading SHAP explainer: {e}")
            return False
    
    def generate_explanations(self, dataset_name, training_results):
        """Generate comprehensive SHAP explanations for all trained models."""
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available for explanations")
            return None
            
        try:
            from src.utils.file_manager import FileManager
            from src.feature_extractor import FeatureExtractor
            
            file_manager = FileManager()
            feature_extractor = FeatureExtractor()
            
            # Load processed data
            processed_data = file_manager.load_processed_data(dataset_name)
            if not processed_data:
                self.logger.warning(f"No processed data found for {dataset_name}")
                return None
            
            X = processed_data['features']
            feature_names = processed_data.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
            
            explanations = {
                'dataset_name': dataset_name,
                'models': {},
                'global_feature_importance': {},
                'feature_names': feature_names
            }
            
            # Generate explanations for each model
            models_dir = Path('models') / dataset_name
            if models_dir.exists():
                for model_file in models_dir.glob('*.joblib'):
                    model_name = model_file.stem
                    
                    # Skip if model failed during training
                    if model_name in training_results.get('models', {}) and 'error' in training_results['models'][model_name]:
                        continue
                    
                    try:
                        # Load model
                        model_data = self.compatibility_manager.safe_load_model(model_file)
                        if not model_data or 'model' not in model_data:
                            continue
                            
                        model = model_data['model']
                        
                        # Create explainer for this model
                        explainer = self.create_explainer(model, X[:100])  # Use subset for background
                        if explainer:
                            # Generate SHAP values for sample
                            sample_size = min(50, len(X))
                            X_sample = X[:sample_size]
                            
                            # Ensure X_sample is numpy array
                            if isinstance(X_sample, pd.DataFrame):
                                X_sample = X_sample.values
                            elif hasattr(X_sample, 'values'):
                                X_sample = X_sample.values
                            
                            shap_values = explainer.shap_values(X_sample)
                            if isinstance(shap_values, list):
                                shap_values = shap_values[1]  # Use positive class for binary classification
                            
                            # Calculate feature importance
                            feature_importance = pd.DataFrame({
                                'feature': feature_names,
                                'importance': np.abs(shap_values).mean(axis=0)
                            }).sort_values('importance', ascending=False)
                            
                            explanations['models'][model_name] = {
                                'feature_importance': feature_importance.to_dict('records'),
                                'top_features': feature_importance.head(10).to_dict('records'),
                                'explainer_type': type(explainer).__name__,
                                'sample_size': sample_size
                            }
                            
                            self.logger.info(f"Generated SHAP explanations for {model_name}")
                            
                    except Exception as e:
                        self.logger.warning(f"Could not generate SHAP explanation for {model_name}: {e}")
                        continue
            
            return explanations if explanations['models'] else None
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanations: {e}")
            return None
    
    def explain_prediction(self, explainer_name, features):
        """Explain a single prediction using existing explainer."""
        try:
            if explainer_name not in self.explainers:
                self.logger.error(f"No explainer found: {explainer_name}")
                return None
            
            explainer_info = self.explainers[explainer_name]
            explainer = explainer_info['explainer']
            explainer_type = explainer_info['type']
            
            # Generate SHAP values
            if explainer_type == 'linear':
                shap_values = explainer.shap_values(features)
            elif explainer_type == 'tree':
                shap_values = explainer.shap_values(features)
                # For binary classification, take the positive class
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
            else:  # kernel
                shap_values = explainer.shap_values(features)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
            
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error explaining prediction: {e}")
            return None
    
    def explain_prediction_for_text(self, dataset_name, model_name, text):
        """Generate SHAP explanation for a single text prediction."""
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available for prediction explanation")
            return None
            
        try:
            from src.utils.file_manager import FileManager
            from src.feature_extractor import FeatureExtractor
            
            file_manager = FileManager()
            feature_extractor = FeatureExtractor()
            
            # Load the specific model
            model_path = Path('models') / dataset_name / f"{model_name}.joblib"
            if not model_path.exists():
                self.logger.warning(f"Model file not found: {model_path}")
                return None
            
            model_data = self.compatibility_manager.safe_load_model(model_path)
            if not model_data or 'model' not in model_data:
                self.logger.warning(f"Could not load model data for {model_name}")
                return None
                
            model = model_data['model']
            
            # Load processed data for background samples
            processed_df = file_manager.load_processed_data(dataset_name)
            if processed_df is None or processed_df.empty:
                self.logger.warning(f"No processed data found for {dataset_name}")
                return None
            
            # Get feature columns (exclude LABEL and TEXT columns)
            feature_columns = [col for col in processed_df.columns if col not in ['LABEL', 'TEXT', 'CONTENT']]
            if not feature_columns:
                self.logger.warning(f"No feature columns found in processed data for {dataset_name}")
                return None
            
            X_background = processed_df[feature_columns][:100].values  # Use subset for background
            feature_names = feature_columns
            
            # Extract features for the input text
            text_features = feature_extractor.extract_features_for_text(text, dataset_name)
            if text_features is None or len(text_features) != len(feature_names):
                self.logger.warning("Could not extract compatible features for text")
                return None
            
            # Create explainer if not exists
            explainer = self.create_explainer(model, X_background)
            if not explainer:
                self.logger.warning(f"Could not create SHAP explainer for {model_name}")
                return None
            
            # Generate SHAP values for the input
            import numpy as np
            text_features_array = np.array(text_features).reshape(1, -1)
            shap_values = explainer.shap_values(text_features_array)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
            
            # Create explanation data
            explanation = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'input_text': text,
                'prediction_shap_values': shap_values[0].tolist(),
                'feature_names': feature_names,
                'feature_values': text_features,
                'top_positive_features': [],
                'top_negative_features': [],
                'expected_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0
            }
            
            # Get top contributing features
            feature_importance = list(zip(feature_names, shap_values[0], text_features))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Split into positive and negative contributors
            positive_features = [(name, value, feature_val) for name, value, feature_val in feature_importance if value > 0][:10]
            negative_features = [(name, value, feature_val) for name, value, feature_val in feature_importance if value < 0][:10]
            
            explanation['top_positive_features'] = [
                {'feature': name, 'shap_value': float(shap_val), 'feature_value': float(feat_val)}
                for name, shap_val, feat_val in positive_features
            ]
            
            explanation['top_negative_features'] = [
                {'feature': name, 'shap_value': float(shap_val), 'feature_value': float(feat_val)}
                for name, shap_val, feat_val in negative_features
            ]
            
            self.logger.info(f"Generated SHAP explanation for {model_name} prediction")
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanation for prediction: {e}")
            return None
    
    def _analyze_feature_types(self, feature_names):
        """Analyze and categorize enhanced feature types."""
        try:
            feature_info = {
                'bert_features': [],
                'sentence_transformer_features': [],
                'ugt_features': [],
                'rct_features': [],
                'rat_features': [],
                'behavioral_features': [],
                'traditional_features': [],
                'total_features': len(feature_names)
            }
            
            for i, name in enumerate(feature_names):
                name_lower = name.lower()
                if 'bert_' in name_lower:
                    feature_info['bert_features'].append((i, name))
                elif 'sentence_' in name_lower:
                    feature_info['sentence_transformer_features'].append((i, name))
                elif 'ugt_' in name_lower:
                    feature_info['ugt_features'].append((i, name))
                elif 'rct_' in name_lower:
                    feature_info['rct_features'].append((i, name))
                elif 'rat_' in name_lower:
                    feature_info['rat_features'].append((i, name))
                elif any(pattern in name_lower for pattern in ['urgency', 'conspiracy', 'emotional', 'pattern']):
                    feature_info['behavioral_features'].append((i, name))
                else:
                    feature_info['traditional_features'].append((i, name))
            
            self.logger.info(f"Feature analysis: BERT({len(feature_info['bert_features'])}), "
                           f"Sentence({len(feature_info['sentence_transformer_features'])}), "
                           f"UGT({len(feature_info['ugt_features'])}), "
                           f"RCT({len(feature_info['rct_features'])}), "
                           f"RAT({len(feature_info['rat_features'])}), "
                           f"Behavioral({len(feature_info['behavioral_features'])}), "
                           f"Traditional({len(feature_info['traditional_features'])})")
            
            return feature_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature types: {e}")
            return {}
    
    def generate_enhanced_explanation(self, model, X_sample, feature_names=None, sample_text=None):
        """Generate enhanced SHAP explanation with transformer + framework insights."""
        if not SHAP_AVAILABLE:
            return None
            
        try:
            # Create explainer
            explainer = self.create_explainer(model, X_sample, feature_names=feature_names)
            if not explainer:
                return None
            
            # Generate SHAP values
            shap_values = self.explain_prediction(explainer, X_sample[:1], 'auto')
            if shap_values is None:
                return None
            
            # Create enhanced explanation
            explanation = {
                'sample_text': sample_text,
                'total_features': len(feature_names) if feature_names else X_sample.shape[1],
                'feature_breakdown': self._create_feature_breakdown(shap_values[0], feature_names),
                'transformer_insights': self._analyze_transformer_contributions(shap_values[0], feature_names),
                'framework_insights': self._analyze_framework_contributions(shap_values[0], feature_names),
                'behavioral_insights': self._analyze_behavioral_contributions(shap_values[0], feature_names),
                'top_contributors': self._get_top_contributors(shap_values[0], feature_names),
                'interpretation': self._interpret_enhanced_explanation(shap_values[0], feature_names)
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced explanation: {e}")
            return None
    
    def _create_feature_breakdown(self, shap_values, feature_names):
        """Create breakdown of SHAP contributions by feature type."""
        try:
            if not hasattr(self, 'enhanced_features_info'):
                return {}
            
            breakdown = {}
            
            # BERT contributions
            bert_indices = [i for i, name in self.enhanced_features_info['bert_features']]
            if bert_indices:
                bert_contributions = [shap_values[i] for i in bert_indices]
                breakdown['bert'] = {
                    'total_contribution': sum(bert_contributions),
                    'avg_contribution': np.mean(bert_contributions),
                    'feature_count': len(bert_indices),
                    'top_features': self._get_top_features_by_type(shap_values, bert_indices, feature_names, 'BERT')
                }
            
            # Sentence transformer contributions
            sentence_indices = [i for i, name in self.enhanced_features_info['sentence_transformer_features']]
            if sentence_indices:
                sentence_contributions = [shap_values[i] for i in sentence_indices]
                breakdown['sentence_transformer'] = {
                    'total_contribution': sum(sentence_contributions),
                    'avg_contribution': np.mean(sentence_contributions),
                    'feature_count': len(sentence_indices),
                    'top_features': self._get_top_features_by_type(shap_values, sentence_indices, feature_names, 'Sentence')
                }
            
            # Framework contributions
            for framework in ['ugt', 'rct', 'rat']:
                framework_indices = [i for i, name in self.enhanced_features_info[f'{framework}_features']]
                if framework_indices:
                    framework_contributions = [shap_values[i] for i in framework_indices]
                    breakdown[framework] = {
                        'total_contribution': sum(framework_contributions),
                        'avg_contribution': np.mean(framework_contributions),
                        'feature_count': len(framework_indices),
                        'top_features': self._get_top_features_by_type(shap_values, framework_indices, feature_names, framework.upper())
                    }
            
            # Behavioral contributions
            behavioral_indices = [i for i, name in self.enhanced_features_info['behavioral_features']]
            if behavioral_indices:
                behavioral_contributions = [shap_values[i] for i in behavioral_indices]
                breakdown['behavioral'] = {
                    'total_contribution': sum(behavioral_contributions),
                    'avg_contribution': np.mean(behavioral_contributions),
                    'feature_count': len(behavioral_indices),
                    'top_features': self._get_top_features_by_type(shap_values, behavioral_indices, feature_names, 'Behavioral')
                }
            
            return breakdown
            
        except Exception as e:
            self.logger.error(f"Error creating feature breakdown: {e}")
            return {}
    
    def _analyze_transformer_contributions(self, shap_values, feature_names):
        """Analyze transformer-specific contributions."""
        try:
            insights = {}
            
            if not hasattr(self, 'enhanced_features_info'):
                return insights
            
            # BERT analysis
            bert_indices = [i for i, name in self.enhanced_features_info['bert_features']]
            if bert_indices:
                bert_values = [shap_values[i] for i in bert_indices]
                insights['bert_analysis'] = {
                    'semantic_understanding_score': sum(bert_values),
                    'key_semantic_features': len([v for v in bert_values if abs(v) > 0.01]),
                    'interpretation': 'High semantic understanding' if sum(bert_values) > 0.1 else 'Low semantic impact'
                }
            
            # Sentence transformer analysis
            sentence_indices = [i for i, name in self.enhanced_features_info['sentence_transformer_features']]
            if sentence_indices:
                sentence_values = [shap_values[i] for i in sentence_indices]
                insights['sentence_analysis'] = {
                    'sentence_coherence_score': sum(sentence_values),
                    'key_sentence_features': len([v for v in sentence_values if abs(v) > 0.01]),
                    'interpretation': 'High sentence-level impact' if sum(sentence_values) > 0.1 else 'Low sentence-level impact'
                }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error analyzing transformer contributions: {e}")
            return {}
    
    def _analyze_framework_contributions(self, shap_values, feature_names):
        """Analyze theoretical framework contributions."""
        try:
            insights = {}
            
            if not hasattr(self, 'enhanced_features_info'):
                return insights
            
            frameworks = ['ugt', 'rct', 'rat']
            framework_names = ['Uses & Gratifications Theory', 'Rational Choice Theory', 'Routine Activity Theory']
            
            for framework, full_name in zip(frameworks, framework_names):
                framework_indices = [i for i, name in self.enhanced_features_info[f'{framework}_features']]
                if framework_indices:
                    framework_values = [shap_values[i] for i in framework_indices]
                    total_contribution = sum(framework_values)
                    
                    insights[f'{framework}_analysis'] = {
                        'framework_name': full_name,
                        'total_contribution': total_contribution,
                        'active_features': len([v for v in framework_values if abs(v) > 0.005]),
                        'interpretation': self._interpret_framework_contribution(framework, total_contribution)
                    }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error analyzing framework contributions: {e}")
            return {}
    
    def _analyze_behavioral_contributions(self, shap_values, feature_names):
        """Analyze behavioral pattern contributions."""
        try:
            if not hasattr(self, 'enhanced_features_info'):
                return {}
            
            behavioral_indices = [i for i, name in self.enhanced_features_info['behavioral_features']]
            if not behavioral_indices:
                return {}
            
            behavioral_values = [shap_values[i] for i in behavioral_indices]
            total_contribution = sum(behavioral_values)
            
            return {
                'behavioral_pattern_score': total_contribution,
                'active_behavioral_features': len([v for v in behavioral_values if abs(v) > 0.005]),
                'interpretation': 'Strong behavioral manipulation patterns detected' if total_contribution > 0.1 
                               else 'Weak behavioral patterns detected' if total_contribution > 0.05
                               else 'No significant behavioral patterns'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing behavioral contributions: {e}")
            return {}
    
    def _get_top_contributors(self, shap_values, feature_names, top_n=10):
        """Get top contributing features across all types."""
        try:
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(shap_values))]
            
            # Create list of (feature_name, shap_value, feature_type)
            contributors = []
            for i, (name, value) in enumerate(zip(feature_names, shap_values)):
                feature_type = self._get_feature_type(name)
                contributors.append({
                    'feature_name': name,
                    'shap_value': float(value),
                    'feature_type': feature_type,
                    'abs_contribution': abs(value)
                })
            
            # Sort by absolute contribution
            contributors.sort(key=lambda x: x['abs_contribution'], reverse=True)
            
            return contributors[:top_n]
            
        except Exception as e:
            self.logger.error(f"Error getting top contributors: {e}")
            return []
    
    def _get_feature_type(self, feature_name):
        """Determine the type of a feature based on its name."""
        name_lower = feature_name.lower()
        if 'bert_' in name_lower:
            return 'BERT Embedding'
        elif 'sentence_' in name_lower:
            return 'Sentence Transformer'
        elif 'ugt_' in name_lower:
            return 'UGT Framework'
        elif 'rct_' in name_lower:
            return 'RCT Framework'
        elif 'rat_' in name_lower:
            return 'RAT Framework'
        elif any(pattern in name_lower for pattern in ['urgency', 'conspiracy', 'emotional', 'pattern']):
            return 'Behavioral Pattern'
        else:
            return 'Traditional Feature'
    
    def _get_top_features_by_type(self, shap_values, indices, feature_names, feature_type):
        """Get top features for a specific feature type."""
        try:
            if not indices or feature_names is None:
                return []
            
            type_features = [(feature_names[i], shap_values[i]) for i in indices]
            type_features.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return [
                {'name': name, 'contribution': float(value), 'type': feature_type}
                for name, value in type_features[:5]
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting top features by type: {e}")
            return []
    
    def _interpret_framework_contribution(self, framework, contribution):
        """Interpret the contribution of a theoretical framework."""
        if framework == 'ugt':
            if contribution > 0.1:
                return "Strong gratification-seeking patterns detected - content appeals to user needs"
            elif contribution > 0.05:
                return "Moderate gratification patterns - some appeal to user motivations"
            else:
                return "Low gratification appeal - content doesn't strongly satisfy user needs"
        
        elif framework == 'rct':
            if contribution > 0.1:
                return "Strong rational choice indicators - high perceived benefit, low perceived cost"
            elif contribution > 0.05:
                return "Moderate rational choice patterns - some cost-benefit considerations"
            else:
                return "Low rational choice impact - unclear cost-benefit analysis"
        
        elif framework == 'rat':
            if contribution > 0.1:
                return "Strong routine activity patterns - opportunistic sharing behavior detected"
            elif contribution > 0.05:
                return "Moderate routine activity patterns - some habitual sharing indicators"
            else:
                return "Low routine activity impact - no clear habitual patterns"
        
        return "Framework contribution unclear"
    
    def _interpret_enhanced_explanation(self, shap_values, feature_names):
        """Create overall interpretation of the enhanced explanation."""
        try:
            if not hasattr(self, 'enhanced_features_info'):
                return "Standard SHAP explanation available"
            
            # Calculate contributions by type
            total_contribution = sum(abs(v) for v in shap_values)
            
            interpretations = []
            
            # Transformer interpretation
            bert_indices = [i for i, name in self.enhanced_features_info['bert_features']]
            sentence_indices = [i for i, name in self.enhanced_features_info['sentence_transformer_features']]
            
            if bert_indices or sentence_indices:
                transformer_contribution = sum(abs(shap_values[i]) for i in bert_indices + sentence_indices)
                transformer_pct = (transformer_contribution / total_contribution) * 100 if total_contribution > 0 else 0
                
                if transformer_pct > 40:
                    interpretations.append(f"Transformer embeddings are the primary drivers ({transformer_pct:.1f}% of total contribution)")
                elif transformer_pct > 20:
                    interpretations.append(f"Transformer embeddings play a significant role ({transformer_pct:.1f}% of total contribution)")
            
            # Framework interpretation
            framework_indices = []
            for framework in ['ugt', 'rct', 'rat']:
                framework_indices.extend([i for i, name in self.enhanced_features_info[f'{framework}_features']])
            
            if framework_indices:
                framework_contribution = sum(abs(shap_values[i]) for i in framework_indices)
                framework_pct = (framework_contribution / total_contribution) * 100 if total_contribution > 0 else 0
                
                if framework_pct > 30:
                    interpretations.append(f"Theoretical frameworks strongly influence the prediction ({framework_pct:.1f}% of total contribution)")
                elif framework_pct > 15:
                    interpretations.append(f"Theoretical frameworks moderately influence the prediction ({framework_pct:.1f}% of total contribution)")
            
            # Behavioral interpretation
            behavioral_indices = [i for i, name in self.enhanced_features_info['behavioral_features']]
            if behavioral_indices:
                behavioral_contribution = sum(abs(shap_values[i]) for i in behavioral_indices)
                behavioral_pct = (behavioral_contribution / total_contribution) * 100 if total_contribution > 0 else 0
                
                if behavioral_pct > 25:
                    interpretations.append(f"Behavioral patterns are highly influential ({behavioral_pct:.1f}% of total contribution)")
                elif behavioral_pct > 10:
                    interpretations.append(f"Behavioral patterns contribute moderately ({behavioral_pct:.1f}% of total contribution)")
            
            if not interpretations:
                interpretations.append("Traditional features dominate the prediction")
            
            return " | ".join(interpretations)
            
        except Exception as e:
            self.logger.error(f"Error interpreting enhanced explanation: {e}")
            return "Enhanced explanation interpretation unavailable"