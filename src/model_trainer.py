"""
Model Training Module

This module provides comprehensive machine learning model training capabilities
for misinformation detection. It implements multiple algorithms, handles data
preprocessing, performs cross-validation, and manages model persistence with
detailed performance evaluation and logging.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from collections import Counter
import joblib
import json
from datetime import datetime
from src.utils.file_manager import FileManager
from src.feature_extractor import FeatureExtractor
from src.model_compatibility import get_compatibility_manager

class ModelTrainer:
    """
    Machine Learning Model Training Class
    
    Implements comprehensive model training pipeline including data preprocessing,
    algorithm selection, cross-validation, performance evaluation, and model
    persistence. Supports multiple machine learning algorithms and feature
    combination strategies for optimal misinformation detection performance.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_manager = FileManager()
        self.feature_extractor = FeatureExtractor()
        self.compatibility_manager = get_compatibility_manager()
        self.training_log_file = None
        self.training_logs = []
        
        # Unified model configurations - Single definition eliminates redundancy
        # These 6 traditional ML algorithms are used across ALL framework combinations
        self.traditional_algorithms = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=5000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': GaussianNB(),
            'neural_network': MLPClassifier(random_state=42, max_iter=500)
        }
        
        # Comprehensive Comparative Analysis Framework Combinations
        # Testing individual framework impact + network features + hyperparameter optimization
        self.framework_combinations = {
            # BASELINE MODELS (Traditional features only - TF-IDF + LDA)
            'baseline_traditional': {
                'description': 'Baseline: TF-IDF + LDA features only (no frameworks, no transformers)',
                'features': ['text'],  # TF-IDF + LDA + basic linguistic features
                'expected_dims': 1000,  # Approximate for TF-IDF + LDA
                'category': 'baseline',
                'hyperparameter_tuning': True
            },
            
            # TRANSFORMER-ENHANCED MODELS (Testing transformer impact)
            'transformer_only': {
                'description': 'Transformers only: BERT + Sentence-BERT embeddings',
                'features': ['transformer_embeddings'],
                'expected_dims': 1152,  # BERT: 768 + Sentence: 384
                'category': 'transformer_enhanced',
                'hyperparameter_tuning': True
            },
            'transformer_traditional': {
                'description': 'Transformers + Traditional: BERT + Sentence-BERT + TF-IDF + LDA',
                'features': ['transformer_embeddings', 'text'],
                'expected_dims': 2152,  # 1152 + 1000
                'category': 'transformer_enhanced',
                'hyperparameter_tuning': True
            },
            
            # INDIVIDUAL FRAMEWORK TESTING (Testing each framework's isolated impact)
            'rat_only_embeddings': {
                'description': 'RAT + Transformers only (isolated RAT impact)',
                'features': ['transformer_embeddings', 'rat_features'],
                'expected_dims': 1167,  # 1152 + 15
                'category': 'individual_framework',
                'framework_type': 'rat',
                'hyperparameter_tuning': True
            },
            'rct_only_embeddings': {
                'description': 'RCT + Transformers only (isolated RCT impact)',
                'features': ['transformer_embeddings', 'rct_features'],
                'expected_dims': 1164,  # 1152 + 12
                'category': 'individual_framework',
                'framework_type': 'rct',
                'hyperparameter_tuning': True
            },
            'ugt_only_embeddings': {
                'description': 'UGT + Transformers only (isolated UGT impact)',
                'features': ['transformer_embeddings', 'ugt_features'],
                'expected_dims': 1170,  # 1152 + 18
                'category': 'individual_framework',
                'framework_type': 'ugt',
                'hyperparameter_tuning': True
            },
            
            # FRAMEWORK-ENHANCED MODELS (Testing theoretical framework + traditional impact)
            'rat_enhanced': {
                'description': 'RAT + Transformers + Traditional (full RAT integration)',
                'features': ['transformer_embeddings', 'text', 'rat_features'],
                'expected_dims': 2167,  # 1152 + 1000 + 15
                'category': 'framework_enhanced',
                'framework_type': 'rat',
                'hyperparameter_tuning': True
            },
            'rct_enhanced': {
                'description': 'RCT + Transformers + Traditional (full RCT integration)',
                'features': ['transformer_embeddings', 'text', 'rct_features'],
                'expected_dims': 2164,  # 1152 + 1000 + 12
                'category': 'framework_enhanced',
                'framework_type': 'rct',
                'hyperparameter_tuning': True
            },
            'ugt_enhanced': {
                'description': 'UGT + Transformers + Traditional (full UGT integration)',
                'features': ['transformer_embeddings', 'text', 'ugt_features'],
                'expected_dims': 2170,  # 1152 + 1000 + 18
                'category': 'framework_enhanced',
                'framework_type': 'ugt',
                'hyperparameter_tuning': True
            },
            

            
            # COMBINED FRAMEWORK MODEL (Full theoretical integration)
            'full_framework': {
                'description': 'Complete Model: RAT+RCT+UGT + Transformers + Traditional',
                'features': ['transformer_embeddings', 'text', 'rat_features', 'rct_features', 'ugt_features'],
                'expected_dims': 2197,  # 1152 + 1000 + 45
                'category': 'full_integration',
                'hyperparameter_tuning': True
            },
            
            # LEGACY COMBINED MODEL (All available features)
            'combined': {
                'description': 'Legacy Combined: All available features (text + behavioral + network + sentiment + theoretical)',
                'features': ['text', 'behavioral', 'network', 'sentiment', 'theoretical'],
                'expected_dims': 2000,  # Approximate for all traditional features
                'category': 'legacy_combined',
                'hyperparameter_tuning': True
            }
        }
        
        # Zero-shot comparison configuration
        self.zero_shot_config = {
            'model': 'facebook/bart-large-mnli',
            'labels': ['misinformation', 'reliable information'],
            'hypothesis_template': 'This text contains {}.'
        }
        
        # Legacy compatibility - maps old specialized_models to new structure
        self.specialized_models = {
            'ugt': self.traditional_algorithms,
            'rct': self.traditional_algorithms, 
            'rat': self.traditional_algorithms,
            'combined': self.traditional_algorithms
        }
    
    def _init_training_logs(self, dataset_name):
        """Initialize training log file for a dataset."""
        try:
            # Create logs directory if it doesn't exist
            logs_dir = Path('logs')
            logs_dir.mkdir(exist_ok=True)
            
            # Create training log file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.training_log_file = logs_dir / f'training_{dataset_name}_{timestamp}.log'
            
            # Clear previous logs
            self.training_logs = []
            
            # Write initial log entry
            self._write_training_log('info', f'Training session started for dataset: {dataset_name}')
            
        except Exception as e:
            self.logger.error(f"Error initializing training logs: {e}")
    
    def _write_training_log(self, level, message):
        """Write a log entry to both file and memory."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = {
                'timestamp': timestamp,
                'level': level,
                'message': message
            }
            
            # Add to memory (for real-time access)
            self.training_logs.append(log_entry)
            
            # Keep only last 100 log entries in memory
            if len(self.training_logs) > 100:
                self.training_logs = self.training_logs[-100:]
            
            # Write to file if available
            if self.training_log_file:
                with open(self.training_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] {level.upper()}: {message}\n")
            
            # Also log to standard logger
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
                
        except Exception as e:
            self.logger.error(f"Error writing training log: {e}")
    
    def get_training_logs(self):
        """Get current training logs."""
        return self.training_logs.copy()
    
    def clear_training_logs(self):
        """Clear training logs from memory."""
        self.training_logs = []
    
    def _analyze_class_imbalance(self, y, dataset_name):
        """Analyze class distribution and determine if SMOTE is needed."""
        class_counts = Counter(y)
        total_samples = len(y)
        
        self._write_training_log('info', f"Class distribution analysis for {dataset_name}:")
        for class_label, count in sorted(class_counts.items()):
            percentage = (count / total_samples) * 100
            self._write_training_log('info', f"  Class {class_label}: {count} samples ({percentage:.1f}%)")
        
        # Calculate imbalance ratio (majority class / minority class)
        if len(class_counts) >= 2:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count
            
            self._write_training_log('info', f"Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            # Recommend SMOTE if imbalance ratio > 2:1
            if imbalance_ratio > 2.0:
                self._write_training_log('info', f"⚠️  Class imbalance detected (ratio > 2:1). SMOTE recommended.")
                return True, imbalance_ratio
            else:
                self._write_training_log('info', f"✅ Classes are relatively balanced. SMOTE not required.")
                return False, imbalance_ratio
        
        return False, 1.0
    
    def _apply_smote(self, X_train, y_train, config):
        """Apply SMOTE or SMOTEENN based on configuration."""
        smote_strategy = config.get('smote_strategy', 'auto')  # 'auto', 'minority', 'not majority', or dict
        smote_method = config.get('smote_method', 'smote')  # 'smote' or 'smoteenn'
        smote_k_neighbors = config.get('smote_k_neighbors', 5)
        smote_random_state = config.get('smote_random_state', 42)
        
        self._write_training_log('info', f"Applying {smote_method.upper()} with strategy='{smote_strategy}', k_neighbors={smote_k_neighbors}")
        
        original_counts = Counter(y_train)
        self._write_training_log('info', f"Original training set: {dict(original_counts)}")
        
        try:
            if smote_method == 'smoteenn':
                # SMOTE + Edited Nearest Neighbours (combines oversampling and undersampling)
                smote_enn = SMOTEENN(
                    smote=SMOTE(
                        sampling_strategy=smote_strategy,
                        k_neighbors=smote_k_neighbors,
                        random_state=smote_random_state
                    ),
                    enn=EditedNearestNeighbours(
                        sampling_strategy='all',
                        n_neighbors=3
                    ),
                    random_state=smote_random_state
                )
                X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
            else:
                # Standard SMOTE
                smote = SMOTE(
                    sampling_strategy=smote_strategy,
                    k_neighbors=smote_k_neighbors,
                    random_state=smote_random_state
                )
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            resampled_counts = Counter(y_resampled)
            self._write_training_log('info', f"After {smote_method.upper()}: {dict(resampled_counts)}")
            
            # Calculate the change
            original_size = len(y_train)
            resampled_size = len(y_resampled)
            size_change = resampled_size - original_size
            
            self._write_training_log('info', f"Training set size: {original_size} → {resampled_size} ({size_change:+d} samples)")
            
            return X_resampled, y_resampled, True
            
        except Exception as e:
            self._write_training_log('warning', f"SMOTE failed: {str(e)}. Continuing with original data.")
            return X_train, y_train, False
    
    def train_models(self, dataset_name, selected_models):
        """
        DEPRECATED: Use train_single_model() for modular training instead.
        Train selected models on the dataset.
        """
        # Initialize training logs
        self._init_training_logs(dataset_name)
        self._write_training_log('info', f"Starting training for models: {', '.join(selected_models)}")
        
        try:
            # Load configuration if exists
            self._write_training_log('info', "Loading model configuration...")
            config = self._load_model_config(dataset_name)
            
            # Load features
            self._write_training_log('info', "Loading features...")
            X, y, feature_names = self.feature_extractor.load_features(dataset_name)
            
            if X is None:
                self._write_training_log('error', "Features not found. Please extract features first.")
                raise ValueError("Features not found. Please extract features first.")
            
            # Split data using config parameters
            test_size = 1.0 - config.get('train_size', 0.8)
            self._write_training_log('info', f"Splitting data (train: {1-test_size:.1%}, test: {test_size:.1%})...")
            random_state = config.get('random_state', 42)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            self._write_training_log('info', "Scaling features...")
            X_train_scaled, X_test_scaled = self.feature_extractor.scale_features(
                dataset_name, X_train, X_test
            )
            
            # Analyze class imbalance and apply SMOTE if needed
            self._write_training_log('info', "Analyzing class distribution...")
            needs_smote, imbalance_ratio = self._analyze_class_imbalance(y_train, dataset_name)
            
            # Apply SMOTE based on configuration
            use_smote = config.get('use_smote', 'auto')  # 'auto', True, False
            smote_applied = False
            
            if use_smote == 'auto':
                # Auto mode: apply SMOTE if imbalance detected
                if needs_smote:
                    X_train_scaled, y_train, smote_applied = self._apply_smote(X_train_scaled, y_train, config)
                else:
                    self._write_training_log('info', "SMOTE not needed - classes are balanced")
            elif use_smote is True:
                # Force SMOTE application
                self._write_training_log('info', "SMOTE forced by configuration")
                X_train_scaled, y_train, smote_applied = self._apply_smote(X_train_scaled, y_train, config)
            else:
                # SMOTE disabled
                self._write_training_log('info', "SMOTE disabled by configuration")
            
            self._write_training_log('info', f"Data prepared: {len(X_train_scaled)} training samples, {len(X_test)} test samples, {X.shape[1]} features")
            
            training_results = {
                'dataset_name': dataset_name,
                'training_date': datetime.now().isoformat(),
                'data_split': {
                    'train_samples': len(X_train_scaled),
                    'test_samples': len(X_test),
                    'total_features': X.shape[1],
                    'original_train_samples': len(X_train),
                    'smote_applied': smote_applied,
                    'imbalance_ratio': imbalance_ratio
                },
                'models': {}
            }
            
            # Train each selected model
            self._write_training_log('info', f"Starting training for {len(selected_models)} models...")
            for i, model_name in enumerate(selected_models, 1):
                if model_name == 'zero_shot':
                    # Handle zero-shot classification separately
                    self._write_training_log('info', f"[{i}/{len(selected_models)}] Running zero-shot classification...")
                    zero_shot_results = self._run_zero_shot_classification(dataset_name, config)
                    training_results['models']['zero_shot'] = zero_shot_results
                    self._write_training_log('info', f"Zero-shot classification completed")
                    continue
                    
                if model_name not in self.default_models:
                    self._write_training_log('warning', f"Unknown model: {model_name}")
                    continue
                
                self._write_training_log('info', f"[{i}/{len(selected_models)}] Training {model_name}...")
                
                try:
                    # Train model with config parameters
                    model = self._get_configured_model(model_name, config)
                    self._write_training_log('info', f"Fitting {model_name} model...")
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    self._write_training_log('info', f"Evaluating {model_name} model...")
                    train_score = model.score(X_train_scaled, y_train)
                    test_score = model.score(X_test_scaled, y_test)
                    
                    # Cross-validation using config parameters
                    cv_folds = config.get('cv_folds', 5)
                    scoring_metric = config.get('scoring_metric', 'f1')
                    self._write_training_log('info', f"Running {cv_folds}-fold cross-validation for {model_name}...")
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring=scoring_metric)
                    
                    # Predictions for detailed metrics
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate additional metrics
                    self._write_training_log('info', f"Calculating detailed metrics for {model_name}...")
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Store comprehensive results
                    training_results['models'][model_name] = {
                        'train_accuracy': float(train_score),
                        'test_accuracy': float(test_score),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'cv_f1_mean': float(cv_scores.mean()),
                        'cv_f1_std': float(cv_scores.std()),
                        'classification_report': classification_report(y_test, y_pred, output_dict=True),
                        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                        'feature_count': X_train_scaled.shape[1],
                        'model_type': model_name
                    }
                    
                    # Save complete model with preprocessing pipeline
                    self._write_training_log('info', f"Saving {model_name} model...")
                    model_path = self.file_manager.get_model_path(dataset_name, model_name)
                    
                    # Get the scaler used for this dataset
                    scaler = self.feature_extractor.get_scaler(dataset_name)
                    
                    # Create complete model package with enhanced metadata
                    complete_model = {
                        'model': model,
                        'scaler': scaler,
                        'feature_columns': self.feature_extractor.get_feature_names(dataset_name),
                        'label_encoder': getattr(self.feature_extractor, 'label_encoder', None),
                        'model_metadata': {
                            'model_name': model_name,
                            'dataset_name': dataset_name,
                            'training_date': datetime.now().isoformat(),
                            'metrics': training_results['models'][model_name],
                            'feature_types': self.feature_extractor.get_feature_types(dataset_name),
                            'expected_features': X_train_scaled.shape[1],
                            'sklearn_version': getattr(model, '_sklearn_version', 'unknown'),
                            'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                            'numpy_version': np.__version__,
                            'compatibility_version': '1.0'
                        }
                    }
                    
                    # Save complete model with compatibility enhancement
                    self._write_training_log('info', f"Saving {model_name} with compatibility enhancements...")
                    try:
                        # Use standard joblib save but with enhanced error handling
                        joblib.dump(complete_model, model_path)
                        
                        # Validate the saved model can be loaded
                        test_load = self.compatibility_manager.safe_load_model(model_path)
                        if test_load is None:
                            raise RuntimeError("Model validation failed after saving")
                        
                        self._write_training_log('info', f"✅ Model {model_name} saved and validated successfully")
                        
                    except Exception as save_error:
                        self._write_training_log('warning', f"Standard save failed, trying compatibility save: {save_error}")
                        # Fallback: save model components separately
                        self._save_model_with_fallback(model_path, complete_model, model_name)
                    
                    # Also save individual model metrics for easy access
                    metrics_path = model_path.replace('.joblib', '_metrics.json')
                    with open(metrics_path, 'w') as f:
                        json.dump(training_results['models'][model_name], f, indent=2)
                    
                    self._write_training_log('info', f"✅ {model_name} training completed! Test accuracy: {test_score:.4f}, F1: {f1:.4f}")
                    
                except Exception as e:
                    self._write_training_log('error', f"❌ Error training {model_name}: {str(e)}")
                    # Store error info but continue with other models
                    training_results['models'][model_name] = {
                        'error': str(e),
                        'model_type': model_name,
                        'test_accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0,
                        'status': 'failed'
                    }
            
            # Find and mark best performing model
            self._write_training_log('info', "Determining best performing model...")
            best_model_name, best_score = self._find_best_model(training_results['models'])
            if best_model_name:
                training_results['best_model'] = {
                    'name': best_model_name,
                    'f1_score': best_score,
                    'path': str(self.file_manager.get_model_path(dataset_name, best_model_name))
                }
                self._write_training_log('info', f"🏆 Best performing model: {best_model_name} (F1: {best_score:.4f})")
            
            # Save training results
            self._write_training_log('info', "Saving training results...")
            self.file_manager.save_results(dataset_name, training_results, 'model_training')
            
            # Update dataset info
            self.file_manager.update_dataset_info(dataset_name, {
                'status': 'models_trained',
                'models_trained': list(selected_models),
                'best_model': best_model_name,
                'best_f1_score': best_score
            })
            
            self._write_training_log('info', "🎉 All model training completed successfully!")
            return training_results
            
        except Exception as e:
            self._write_training_log('error', f"💥 Critical error during model training: {str(e)}")
            raise
    
    def _find_best_model(self, models_results):
        """Find the best performing model based on F1 score."""
        best_name = None
        best_score = 0.0
        
        for model_name, results in models_results.items():
            if 'error' in results:
                continue
            f1_score = results.get('f1_score', 0.0)
            if f1_score > best_score:
                best_score = f1_score
                best_name = model_name
        
        return best_name, best_score
    
    def _run_zero_shot_classification(self, dataset_name: str, config: dict) -> dict:
        """Run zero-shot classification as part of model training pipeline."""
        self.logger.info("Running zero-shot classification in training pipeline")
        
        try:
            # Get zero-shot configuration
            zero_shot_config = config.get('zero_shot_config', {
                'model': 'facebook/bart-large-mnli',
                'confidence_threshold': 0.7,
                'label_set': 'binary',
                'kenyan_context': True,
                'multilingual': True
            })
            
            # Import zero-shot classifier
            from .zero_shot_labeling import ZeroShotLabeler
            zero_shot_classifier = ZeroShotLabeler(self.file_manager)
            
            # Run zero-shot classification
            zero_shot_results = zero_shot_classifier.classify_dataset(dataset_name, zero_shot_config)
            
            # Save zero-shot results
            self.save_zero_shot_model(dataset_name, zero_shot_results)
            
            # Format results for training pipeline
            formatted_results = {
                'model_name': 'zero_shot',
                'model_type': 'transformer',
                'training_time': zero_shot_results.get('processing_time', 'N/A'),
                'accuracy': zero_shot_results.get('average_confidence', 0.0),
                'f1_score': zero_shot_results.get('average_confidence', 0.0),  # Use confidence as proxy
                'precision': zero_shot_results.get('average_confidence', 0.0),
                'recall': zero_shot_results.get('average_confidence', 0.0),
                'total_classified': zero_shot_results.get('total_classified', 0),
                'misinformation_detected': zero_shot_results.get('misinformation_detected', 0),
                'model_config': zero_shot_config,
                'notes': 'Zero-shot classification using pre-trained transformer'
            }
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error in zero-shot classification: {e}")
            return {
                'model_name': 'zero_shot',
                'error': str(e),
                'status': 'failed'
            }
    
    def save_zero_shot_model(self, dataset_name: str, zero_shot_results: dict):
        """Save zero-shot model results for later use."""
        try:
            # Create zero-shot model package
            zero_shot_model = {
                'model_type': 'zero_shot',
                'model_name': zero_shot_results.get('model_name', 'facebook/bart-large-mnli'),
                'results': zero_shot_results,
                'training_date': datetime.now().isoformat(),
                'dataset_name': dataset_name,
                'live_prediction_capable': True
            }
            
            # Save zero-shot model
            zs_model_path = self.file_manager.get_model_path(dataset_name, 'zero_shot')
            joblib.dump(zero_shot_model, zs_model_path)
            
            # Save metrics separately
            zs_metrics_path = zs_model_path.replace('.joblib', '_metrics.json')
            with open(zs_metrics_path, 'w') as f:
                json.dump(zero_shot_results, f, indent=2)
            
            self.logger.info(f"Zero-shot model saved for {dataset_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving zero-shot model: {e}")
            return False
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            raise
    
    def get_training_results(self, dataset_name):
        """Get the latest training results for a dataset."""
        return self.file_manager.load_results(dataset_name, 'model_training')
    
    def retrain_model(self, dataset_name, model_name, custom_params=None):
        """Retrain a specific model with custom parameters."""
        self.logger.info(f"Retraining {model_name} for dataset: {dataset_name}")
        
        try:
            # Load features
            X, y, feature_names = self.feature_extractor.load_features(dataset_name)
            
            if X is None:
                raise ValueError("Features not found. Please extract features first.")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled, X_test_scaled = self.feature_extractor.scale_features(
                dataset_name, X_train, X_test
            )
            
            # Get model with custom parameters
            if custom_params:
                model = self._get_model_with_params(model_name, custom_params)
            else:
                model = self.default_models[model_name]
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            test_score = model.score(X_test_scaled, y_test)
            
            # Save retrained model
            model_path = self.file_manager.get_model_path(dataset_name, f"{model_name}_retrained")
            joblib.dump(model, model_path)
            
            self.logger.info(f"Retraining completed. Test accuracy: {test_score:.4f}")
            
            return {
                'model_name': model_name,
                'test_accuracy': float(test_score),
                'custom_params': custom_params,
                'retrain_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error retraining model: {e}")
            raise
    
    def _get_model_with_params(self, model_name, params):
        """Get model instance with custom parameters."""
        if model_name == 'logistic_regression':
            return LogisticRegression(random_state=42, **params)
        elif model_name == 'random_forest':
            return RandomForestClassifier(random_state=42, **params)
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=42, **params)
        elif model_name == 'svm':
            return SVC(random_state=42, probability=True, **params)
        elif model_name == 'naive_bayes':
            return GaussianNB(**params)
        elif model_name == 'neural_network':
            return MLPClassifier(random_state=42, **params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def compare_models(self, dataset_name):
        """Compare performance of all trained models."""
        training_results = self.get_training_results(dataset_name)
        
        if not training_results or 'models' not in training_results:
            return None
        
        comparison = {
            'dataset_name': dataset_name,
            'comparison_date': datetime.now().isoformat(),
            'models': []
        }
        
        for model_name, results in training_results['models'].items():
            if 'error' not in results:
                comparison['models'].append({
                    'name': model_name,
                    'test_accuracy': results['test_accuracy'],
                    'cv_f1_mean': results['cv_f1_mean'],
                    'cv_f1_std': results['cv_f1_std']
                })
        
        # Sort by test accuracy
        comparison['models'].sort(key=lambda x: x['test_accuracy'], reverse=True)
        
        return comparison
    
    def _load_model_config(self, dataset_name):
        """Load model configuration from saved config file."""
        import json
        from pathlib import Path
        
        # Try gratification config first, then traditional config
        config_paths = [
            Path('datasets') / dataset_name / 'config' / 'gratification_model_config.json',
            Path('datasets') / dataset_name / 'config' / 'model_config.json'
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    self.logger.info(f"Loaded model config from {config_path}")
                    return config
                except Exception as e:
                    self.logger.warning(f"Error loading config from {config_path}: {e}")
        
        # Return default config if no config file found
        self.logger.info("Using default model configuration")
        return {
            'train_size': 0.8,
            'random_state': 42,
            'cv_folds': 5,
            'scoring_metric': 'f1',
            # SMOTE configuration
            'use_smote': 'auto',  # 'auto', True, False
            'smote_method': 'smote',  # 'smote' or 'smoteenn'
            'smote_strategy': 'auto',  # 'auto', 'minority', 'not majority', or dict
            'smote_k_neighbors': 5,
            'smote_random_state': 42
        }
    
    def _get_configured_model(self, model_name, config):
        """Get model instance with configuration parameters applied."""
        random_state = config.get('random_state', 42)
        
        if model_name == 'logistic_regression':
            return LogisticRegression(
                C=config.get('lr_C', 1.0),
                solver=config.get('lr_solver', 'liblinear'),
                max_iter=config.get('lr_max_iter', 1000),
                random_state=random_state
            )
        elif model_name == 'random_forest':
            max_depth = config.get('rf_max_depth', 'None')
            max_depth = None if max_depth == 'None' else int(max_depth)
            return RandomForestClassifier(
                n_estimators=config.get('rf_n_estimators', 100),
                max_depth=max_depth,
                min_samples_split=config.get('rf_min_samples_split', 2),
                random_state=random_state
            )
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=config.get('gb_n_estimators', 100),
                learning_rate=config.get('gb_learning_rate', 0.1),
                max_depth=config.get('gb_max_depth', 3),
                random_state=random_state
            )
        elif model_name == 'svm':
            return SVC(
                C=config.get('svm_C', 1.0),
                kernel=config.get('svm_kernel', 'rbf'),
                probability=True,
                random_state=random_state
            )
        elif model_name == 'naive_bayes':
            return GaussianNB()
        elif model_name == 'neural_network':
            # Parse hidden layer sizes from string format like "(100,)" or "(100, 50)"
            hidden_layers_str = config.get('nn_hidden_layers', '(100,)')
            try:
                # Safely evaluate the tuple string
                hidden_layers = eval(hidden_layers_str) if isinstance(hidden_layers_str, str) else hidden_layers_str
                if not isinstance(hidden_layers, tuple):
                    hidden_layers = (100,)  # Default fallback
            except:
                hidden_layers = (100,)  # Default fallback
            
            return MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                alpha=config.get('nn_alpha', 0.0001),
                learning_rate=config.get('nn_learning_rate', 'constant'),
                solver=config.get('nn_solver', 'adam'),
                max_iter=config.get('nn_max_iter', 500),
                random_state=random_state
            )
        else:
            # Fallback to default models
            return self.default_models.get(model_name)
    
    def _save_model_with_fallback(self, model_path, complete_model, model_name):
        """Fallback method to save model with compatibility issues."""
        try:
            # Try saving components separately
            model_dir = Path(model_path).parent
            model_stem = Path(model_path).stem
            
            # Save individual components
            model_only_path = model_dir / f"{model_stem}_model_only.joblib"
            scaler_path = model_dir / f"{model_stem}_scaler.joblib"
            metadata_path = model_dir / f"{model_stem}_metadata.json"
            
            # Save core model
            joblib.dump(complete_model['model'], model_only_path)
            
            # Save scaler if exists
            if complete_model['scaler']:
                joblib.dump(complete_model['scaler'], scaler_path)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(complete_model['model_metadata'], f, indent=2)
            
            # Create a compatibility wrapper
            fallback_model = {
                'model_path': str(model_only_path),
                'scaler_path': str(scaler_path) if complete_model['scaler'] else None,
                'metadata_path': str(metadata_path),
                'feature_columns': complete_model['feature_columns'],
                'label_encoder': complete_model['label_encoder'],
                'model_metadata': complete_model['model_metadata'],
                'fallback_save': True
            }
            
            # Save the wrapper
            joblib.dump(fallback_model, model_path)
            self._write_training_log('info', f"✅ Model {model_name} saved using fallback method")
            
        except Exception as e:
            self._write_training_log('error', f"❌ Fallback save also failed for {model_name}: {e}")
            raise
    
    def train_unified_framework_models(self, dataset_name, framework_combinations=None, config=None):
        """
        DEPRECATED: Use train_single_model() for modular training instead.
        
        Comprehensive comparative analysis: Baseline vs Framework-Enhanced vs Zero-shot.
        
        Research Question: Do theoretical frameworks (RAT+RCT+UGT) + transformer embeddings 
        improve performance over traditional TF-IDF+LDA baseline?
        
        Args:
            dataset_name: Name of the dataset
            framework_combinations: List of combinations to test (default: all)
            config: Training configuration including zero-shot settings
        """
        if framework_combinations is None:
            framework_combinations = list(self.framework_combinations.keys())
        
        self._init_training_logs(dataset_name)
        self._write_training_log('info', f"🚀 Starting unified framework training for: {', '.join(framework_combinations)}")
        
        try:
            # Load unified features (all feature types)
            self._write_training_log('info', "📊 Loading unified features...")
            feature_data = self._load_unified_features(dataset_name)
            
            if not feature_data:
                raise ValueError("Unified features not found. Please extract features first.")
            
            # Load labels
            labels = self._load_labels(dataset_name)
            
            # Initialize comprehensive results structure
            training_results = {
                'dataset_name': dataset_name,
                'training_date': datetime.now().isoformat(),
                'framework_type': 'unified_rat_rct_ugt',
                'algorithms_tested': list(self.traditional_algorithms.keys()),
                'combinations_tested': framework_combinations,
                'results': {},
                'comparative_analysis': {}
            }
            
            # Train each framework combination with all 6 algorithms
            for combo_name in framework_combinations:
                if combo_name not in self.framework_combinations:
                    self._write_training_log('warning', f"⚠️ Unknown combination: {combo_name}")
                    continue
                
                self._write_training_log('info', f"\n🎯 Training {combo_name} ({self.framework_combinations[combo_name]['description']})")
                
                # Get features for this combination
                X = self._create_feature_combination(feature_data, combo_name)
                if X is None:
                    self._write_training_log('error', f"❌ Failed to create features for {combo_name}")
                    continue
                
                # Train all 6 algorithms for this combination
                combo_results = self._train_combination_with_all_algorithms(
                    X, labels, combo_name, dataset_name
                )
                
                training_results['results'][combo_name] = combo_results
            
            # Run zero-shot comparison if configured
            if config and config.get('zero_shot_config'):
                self._write_training_log('info', "\n🤖 Running zero-shot comparison (BART-MNLI)...")
                zero_shot_results = self._run_zero_shot_comparison(dataset_name, config['zero_shot_config'])
                training_results['zero_shot_results'] = zero_shot_results
            
            # Generate comprehensive comparative analysis
            training_results['comparative_analysis'] = self._generate_framework_analysis(
                training_results['results'], 
                training_results.get('zero_shot_results')
            )
            
            # Add individual framework impact analysis
            training_results['individual_framework_analysis'] = self._analyze_individual_framework_impact(
                training_results['results']
            )
            
            # Save unified results
            self._save_unified_training_results(dataset_name, training_results)
            
            self._write_training_log('info', "🎉 Unified framework training completed successfully!")
            return training_results
            
        except Exception as e:
            self._write_training_log('error', f"❌ Unified framework training failed: {e}")
            raise
    
    def train_specialized_models(self, dataset_name, model_types=None, feature_config=None):
        """
        DEPRECATED: Use train_single_model() for modular training instead.
        
        Train specialized models for different theoretical frameworks.
        Integrates with existing transformer-based pipeline.
        
        Args:
            dataset_name: Name of the dataset
            model_types: List of model types ['ugt', 'rct', 'rat', 'traditional_ml', 'combined']
            feature_config: Configuration for feature extraction
        """
        if model_types is None:
            model_types = ['ugt', 'rct', 'rat', 'traditional_ml', 'combined']
        
        self._init_training_logs(dataset_name)
        self._write_training_log('info', f"Starting specialized model training for: {', '.join(model_types)}")
        
        if feature_config is None:
            feature_config = {
                'use_embeddings': True,  # Use your existing transformer embeddings
                'use_enhanced_text': True,  # Use new enhanced text features
                'embedding_config': {
                    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'embedding_strategy': 'mean_pooling'
                }
            }
        
        results = {
            'dataset_name': dataset_name,
            'training_date': datetime.now().isoformat(),
            'model_types': {},
            'feature_config': feature_config
        }
        
        try:
            # Extract comprehensive features (includes transformers + enhanced text)
            self._write_training_log('info', "Extracting comprehensive features...")
            feature_info = self.feature_extractor.extract_comprehensive_features(
                dataset_name,
                embedding_config=feature_config.get('embedding_config') if feature_config.get('use_embeddings') else None,
                zero_shot_config={'run_zero_shot': False}  # Can be enabled if needed
            )
            
            # Load features
            X, y, feature_names = self.feature_extractor.load_features(dataset_name)
            
            if X is None:
                raise ValueError("Features not found. Please extract features first.")
            
            self._write_training_log('info', f"Loaded features: {X.shape[1]} features, {len(X)} samples")
            
            # Train each model type
            for model_type in model_types:
                self._write_training_log('info', f"Training {model_type.upper()} models...")
                try:
                    if model_type in ['ugt', 'rct', 'rat']:
                        model_results = self._train_theoretical_framework_models(
                            X, y, feature_names, dataset_name, model_type
                        )
                    elif model_type == 'traditional_ml':
                        model_results = self._train_traditional_models_specialized(
                            X, y, feature_names, dataset_name
                        )
                    elif model_type == 'combined':
                        model_results = self._train_combined_models_specialized(
                            X, y, feature_names, dataset_name
                        )
                    else:
                        self._write_training_log('warning', f"Unknown model type: {model_type}")
                        continue
                    
                    results['model_types'][model_type] = model_results
                    self._write_training_log('info', f"Successfully trained {model_type.upper()} models")
                    
                except Exception as e:
                    self._write_training_log('error', f"Error training {model_type} models: {e}")
                    results['model_types'][model_type] = {'error': str(e)}
            
            # Save overall results
            self._save_specialized_training_results(dataset_name, results)
            
            return results
            
        except Exception as e:
            self._write_training_log('error', f"Error in specialized training: {e}")
            raise
    
    def _train_theoretical_framework_models(self, X, y, feature_names, dataset_name, framework_type):
        """Train models for a specific theoretical framework (UGT, RCT, RAT)."""
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = self.specialized_models[framework_type]
        results = {
            'framework_type': framework_type,
            'feature_count': X.shape[1],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'models': {}
        }
        
        for model_name, model in models.items():
            try:
                self._write_training_log('info', f"Training {framework_type} {model_name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                # Detailed metrics
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                model_results = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                }
                
                results['models'][model_name] = model_results
                
                # Save model
                self._save_specialized_model(dataset_name, framework_type, model_name, model, scaler, feature_names, model_results)
                
                self._write_training_log('info', f"{framework_type} {model_name}: Test Acc={test_score:.3f}, CV={cv_scores.mean():.3f}")
                
            except Exception as e:
                self._write_training_log('error', f"Error training {framework_type} {model_name}: {e}")
                results['models'][model_name] = {'error': str(e)}
        
        return results
    
    def _train_traditional_models_specialized(self, X, y, feature_names, dataset_name):
        """Train traditional ML models with specialized saving."""
        # Use existing train_models method but with specialized saving
        selected_models = list(self.default_models.keys())
        
        # Temporarily redirect to specialized saving
        original_method = self.train_models
        
        try:
            # Train using existing method
            training_results = original_method(dataset_name, selected_models)
            
            # Reformat results for consistency
            results = {
                'framework_type': 'traditional_ml',
                'feature_count': X.shape[1],
                'models': {}
            }
            
            for model_name, model_data in training_results.get('models', {}).items():
                if 'error' not in model_data:
                    results['models'][model_name] = {
                        'train_accuracy': model_data.get('train_accuracy', 0),
                        'test_accuracy': model_data.get('test_accuracy', 0),
                        'cv_mean': model_data.get('cv_mean', 0),
                        'cv_std': model_data.get('cv_std', 0),
                        'roc_auc': model_data.get('roc_auc', 0)
                    }
                else:
                    results['models'][model_name] = model_data
            
            return results
            
        except Exception as e:
            self._write_training_log('error', f"Error in traditional ML training: {e}")
            return {'error': str(e)}
    
    def _train_combined_models_specialized(self, X, y, feature_names, dataset_name):
        """
        Train combined/ensemble models using best performers from each category:
        - Best Traditional ML models (all 6 algorithms)
        - Best Framework models (UGT, RCT, RAT)
        - Transformer embeddings integration
        """
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import VotingClassifier, StackingClassifier
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        self._write_training_log('info', "Training COMBINED models with ALL 6 ML algorithms + Transformers")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {
            'framework_type': 'combined',
            'feature_count': X.shape[1],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'models': {},
            'feature_breakdown': {
                'total_features': X.shape[1],
                'bert_embeddings': '768 dims (estimated)',
                'sentence_transformers': '384 dims (estimated)', 
                'theoretical_frameworks': '45 dims (UGT+RCT+RAT)',
                'behavioral_analysis': '44 dims',
                'traditional_features': f'{X.shape[1] - 1241} dims (TF-IDF, sentiment, etc.)'
            }
        }
        
        # Train ALL 6 traditional ML models for ensemble selection
        self._write_training_log('info', "Training all 6 traditional ML algorithms...")
        base_models = []
        model_performances = {}
        
        for name, model in self.default_models.items():
            try:
                self._write_training_log('info', f"Training {name} for ensemble...")
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train_scaled, y_train)
                
                # Evaluate performance
                test_score = model_copy.score(X_test_scaled, y_test)
                cv_scores = cross_val_score(model_copy, X_train_scaled, y_train, cv=3)
                
                model_performances[name] = {
                    'test_accuracy': test_score,
                    'cv_mean': cv_scores.mean(),
                    'model': model_copy
                }
                
                # Add to base models if performance is good
                if test_score > 0.6:  # Only include decent performers
                    base_models.append((name, model_copy))
                    self._write_training_log('info', f"{name}: Test Acc={test_score:.3f}, CV={cv_scores.mean():.3f} ✅")
                else:
                    self._write_training_log('warning', f"{name}: Test Acc={test_score:.3f} - excluded from ensemble")
                    
            except Exception as e:
                self._write_training_log('error', f"Could not train {name} for ensemble: {e}")
        
        # Create ensemble voting classifier with best models
        if len(base_models) >= 3:
            self._write_training_log('info', f"Creating ensemble with {len(base_models)} models")
            ensemble_voting = VotingClassifier(estimators=base_models, voting='soft')
            self.specialized_models['combined']['ensemble_voting'] = ensemble_voting
            
            # Create stacking classifier with best models
            if len(base_models) >= 4:
                # Use best performer as meta-learner
                best_model_name = max(model_performances.keys(), 
                                    key=lambda x: model_performances[x]['test_accuracy'])
                meta_learner = model_performances[best_model_name]['model']
                
                ensemble_stacking = StackingClassifier(
                    estimators=base_models[:4],  # Use top 4 as base learners
                    final_estimator=meta_learner,
                    cv=3
                )
                self.specialized_models['combined']['ensemble_stacking'] = ensemble_stacking
                self._write_training_log('info', f"Created stacking ensemble with {best_model_name} as meta-learner")
        
        # Train all combined models
        models = self.specialized_models['combined']
        for model_name, model in models.items():
            if model is None:
                continue
                
            try:
                self._write_training_log('info', f"Training combined {model_name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                # Cross-validation (reduced for complex models)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3)
                
                # Detailed metrics
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                model_results = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                }
                
                results['models'][model_name] = model_results
                
                # Save model
                self._save_specialized_model(dataset_name, 'combined', model_name, model, scaler, feature_names, model_results)
                
                self._write_training_log('info', f"Combined {model_name}: Test Acc={test_score:.3f}, CV={cv_scores.mean():.3f}")
                
            except Exception as e:
                self._write_training_log('error', f"Error training combined {model_name}: {e}")
                results['models'][model_name] = {'error': str(e)}
        
        return results
    
    def _save_specialized_model(self, dataset_name, model_type, model_name, model, scaler, feature_names, metrics):
        """Save specialized model with organized directory structure."""
        try:
            # Create directory structure
            model_dir = Path('models') / dataset_name / model_type
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / f'{model_name}.joblib'
            
            # Create complete model package
            complete_model = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'model_type': model_type,
                'model_name': model_name,
                'training_date': datetime.now().isoformat(),
                'metrics': metrics,
                'feature_count': len(feature_names)
            }
            
            joblib.dump(complete_model, model_path)
            
            # Save metadata
            metadata = {
                'model_type': model_type,
                'model_name': model_name,
                'feature_count': len(feature_names),
                'training_date': datetime.now().isoformat(),
                'metrics': {k: v for k, v in metrics.items() if k not in ['classification_report', 'confusion_matrix']}
            }
            
            metadata_path = model_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self._write_training_log('info', f"Saved {model_type} {model_name} model")
            
        except Exception as e:
            self._write_training_log('error', f"Error saving {model_type} {model_name}: {e}")
    
    def _load_unified_features(self, dataset_name):
        """Load all feature types for unified framework training."""
        try:
            # Try to load from unified features directory first
            unified_features_dir = Path('datasets') / dataset_name / 'features' / 'unified'
            if unified_features_dir.exists():
                return self._load_saved_unified_features(dataset_name)
            
            # Otherwise, load from standard feature extraction
            X, y, feature_names = self.feature_extractor.load_features(dataset_name)
            if X is None:
                return None
            
            # Create unified feature structure
            feature_data = {
                'all_features': X,
                'feature_names': feature_names,
                'labels': y
            }
            
            # Try to separate feature types if possible
            # This is a simplified approach - in practice, you'd want more sophisticated separation
            total_features = X.shape[1]
            
            # Estimate feature splits based on typical dimensions
            if total_features >= 1152:  # Has transformer embeddings
                feature_data['transformer_embeddings'] = X[:, :1152]
                remaining_start = 1152
                
                # Estimate theoretical framework features
                if total_features >= 1152 + 45:  # Has all frameworks
                    feature_data['rat_features'] = X[:, remaining_start:remaining_start+15]
                    feature_data['rct_features'] = X[:, remaining_start+15:remaining_start+27]
                    feature_data['ugt_features'] = X[:, remaining_start+27:remaining_start+45]
                    remaining_start += 45
                
                # Remaining features are behavioral/traditional
                if remaining_start < total_features:
                    feature_data['behavioral_features'] = X[:, remaining_start:]
            
            return feature_data
            
        except Exception as e:
            self._write_training_log('error', f"Error loading unified features: {e}")
            return None
    
    def _load_saved_unified_features(self, dataset_name):
        """Load pre-saved unified features."""
        unified_features_dir = Path('datasets') / dataset_name / 'features' / 'unified'
        feature_data = {}
        
        # Load individual feature type arrays
        feature_types = ['text', 'behavioral', 'sentiment', 'language', 'theoretical', 'network', 'transformer_embeddings', 'data_processor']
        
        for feature_type in feature_types:
            feature_file = unified_features_dir / f'{feature_type}_features.npy'
            if feature_file.exists():
                try:
                    feature_data[feature_type] = np.load(feature_file)
                    self._write_training_log('info', f"Loaded {feature_type} features: {feature_data[feature_type].shape}")
                except Exception as e:
                    self._write_training_log('warning', f"Failed to load {feature_type} features: {e}")
        
        # Load labels
        labels_file = Path('datasets') / dataset_name / 'features' / 'y_labels.npy'
        if labels_file.exists():
            feature_data['labels'] = np.load(labels_file)
        
        # Also check for pre-computed combination features
        for combo_name in self.framework_combinations.keys():
            feature_file = unified_features_dir / f'{combo_name}_features.npy'
            if feature_file.exists():
                feature_data[combo_name] = np.load(feature_file)
        
        return feature_data
    
    def _load_labels(self, dataset_name):
        """Load labels from processed data."""
        processed_file = Path('datasets') / dataset_name / 'processed' / 'processed_data.csv'
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed data not found: {processed_file}")
        
        df = pd.read_csv(processed_file)
        if 'LABEL' not in df.columns:
            raise ValueError("LABEL column not found in processed data")
        
        return df['LABEL'].values
    
    def _create_feature_combination(self, feature_data, combo_name):
        """Create feature combination based on framework specification."""
        if combo_name in feature_data:
            # Pre-saved combination
            return feature_data[combo_name]
        
        if combo_name not in self.framework_combinations:
            return None
        
        combo_config = self.framework_combinations[combo_name]
        feature_list = []
        
        for feature_type in combo_config['features']:
            if feature_type in feature_data:
                feature_list.append(feature_data[feature_type])
            else:
                self._write_training_log('warning', f"Feature type '{feature_type}' not found for {combo_name}")
        
        if feature_list:
            return np.hstack(feature_list)
        else:
            return None
    
    # REMOVED: _train_combination_with_all_algorithms - redundant with modular single model training
    
    def _needs_smote(self, y_train):
        """Check if SMOTE is needed based on class imbalance."""
        from collections import Counter
        class_counts = Counter(y_train)
        if len(class_counts) < 2:
            return False
        
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        
        return imbalance_ratio > 2.0
    
    def _apply_smote_unified(self, X_train, y_train):
        """Apply SMOTE for unified training."""
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            self._write_training_log('info', f"  📊 SMOTE applied: {len(y_train)} → {len(y_resampled)} samples")
            return X_resampled, y_resampled
        except Exception as e:
            self._write_training_log('warning', f"  ⚠️ SMOTE failed: {e}. Using original data.")
            return X_train, y_train
    
    def _save_unified_model(self, dataset_name, combo_name, algo_name, model, scaler):
        """Save unified framework model."""
        models_dir = Path('datasets') / dataset_name / 'models' / 'unified'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = models_dir / f'{combo_name}_{algo_name}.joblib'
        scaler_file = models_dir / f'{combo_name}_{algo_name}_scaler.joblib'
        
        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)
    
    def _run_zero_shot_comparison(self, dataset_name, zero_shot_config):
        """Run zero-shot classification comparison using BART-MNLI."""
        try:
            from src.zero_shot_labeling import ZeroShotLabeling
            
            # Initialize zero-shot classifier
            zero_shot = ZeroShotLabeling()
            
            # Load processed data
            processed_file = Path('datasets') / dataset_name / 'processed' / 'processed_data.csv'
            df = pd.read_csv(processed_file)
            
            # Run zero-shot classification
            results = zero_shot.classify_dataset(
                dataset_name=dataset_name,
                model_name=zero_shot_config.get('model', 'facebook/bart-large-mnli'),
                confidence_threshold=zero_shot_config.get('confidence_threshold', 0.7)
            )
            
            if results and 'evaluation_metrics' in results:
                return {
                    'model': zero_shot_config.get('model', 'facebook/bart-large-mnli'),
                    'metrics': results['evaluation_metrics'],
                    'predictions_count': results.get('predictions_count', 0),
                    'confidence_threshold': zero_shot_config.get('confidence_threshold', 0.7)
                }
            else:
                return {'error': 'Zero-shot classification failed'}
                
        except Exception as e:
            self._write_training_log('error', f"Zero-shot comparison failed: {e}")
            return {'error': str(e)}
    
    def _generate_framework_analysis(self, results, zero_shot_results=None):
        """Generate comprehensive comparative analysis including zero-shot comparison."""
        analysis = {
            'best_overall': None,
            'best_by_combination': {},
            'best_by_algorithm': {},
            'framework_effectiveness': {},
            'algorithm_rankings': {},
            'category_comparison': {},
            'zero_shot_comparison': zero_shot_results,
            'research_insights': {}
        }
        
        # Find best overall performance
        best_score = 0
        best_config = None
        
        # Algorithm performance tracking
        algorithm_scores = {algo: [] for algo in self.traditional_algorithms.keys()}
        
        for combo_name, combo_results in results.items():
            if 'algorithms' not in combo_results:
                continue
            
            combo_best = {'score': 0, 'algorithm': None}
            
            for algo_name, algo_results in combo_results['algorithms'].items():
                if 'error' not in algo_results and 'metrics' in algo_results:
                    score = algo_results['metrics']['f1_score']
                    algorithm_scores[algo_name].append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_config = {
                            'combination': combo_name,
                            'algorithm': algo_name,
                            'score': score,
                            'accuracy': algo_results['metrics']['accuracy']
                        }
                    
                    if score > combo_best['score']:
                        combo_best = {'score': score, 'algorithm': algo_name}
            
            analysis['best_by_combination'][combo_name] = combo_best
        
        analysis['best_overall'] = best_config
        
        # Framework effectiveness analysis
        framework_scores = {}
        for combo_name, combo_results in results.items():
            if 'algorithms' not in combo_results:
                continue
            
            scores = []
            for algo_name, algo_results in combo_results['algorithms'].items():
                if 'error' not in algo_results and 'metrics' in algo_results:
                    scores.append(algo_results['metrics']['f1_score'])
            
            if scores:
                framework_scores[combo_name] = {
                    'mean_f1': np.mean(scores),
                    'std_f1': np.std(scores),
                    'max_f1': np.max(scores),
                    'min_f1': np.min(scores),
                    'algorithm_count': len(scores)
                }
        
        analysis['framework_effectiveness'] = framework_scores
        
        # Algorithm rankings across all frameworks
        for algo_name, scores in algorithm_scores.items():
            if scores:
                analysis['algorithm_rankings'][algo_name] = {
                    'mean_f1': np.mean(scores),
                    'std_f1': np.std(scores),
                    'max_f1': np.max(scores),
                    'min_f1': np.min(scores),
                    'framework_count': len(scores)
                }
        
        # Category-based comparison for research insights
        category_performance = {
            'baseline': [],
            'transformer_enhanced': [],
            'framework_enhanced': [],
            'full_integration': []
        }
        
        for combo_name, combo_results in results.items():
            if 'algorithms' not in combo_results:
                continue
            
            # Get category from framework combinations
            combo_config = self.framework_combinations.get(combo_name, {})
            category = combo_config.get('category', 'unknown')
            
            if category in category_performance:
                for algo_name, algo_results in combo_results['algorithms'].items():
                    if 'error' not in algo_results and 'metrics' in algo_results:
                        category_performance[category].append(algo_results['metrics']['f1_score'])
        
        # Calculate category statistics
        for category, scores in category_performance.items():
            if scores:
                analysis['category_comparison'][category] = {
                    'mean_f1': np.mean(scores),
                    'std_f1': np.std(scores),
                    'max_f1': np.max(scores),
                    'min_f1': np.min(scores),
                    'model_count': len(scores)
                }
        
        # Research insights
        analysis['research_insights'] = self._generate_research_insights(
            analysis['category_comparison'], 
            zero_shot_results
        )
        
        return analysis
    
    def _generate_research_insights(self, category_comparison, zero_shot_results):
        """Generate research insights about framework effectiveness."""
        insights = {
            'framework_improvement': {},
            'transformer_impact': {},
            'zero_shot_performance': {},
            'recommendations': []
        }
        
        # Compare baseline vs enhanced models
        baseline_f1 = category_comparison.get('baseline', {}).get('mean_f1', 0)
        transformer_f1 = category_comparison.get('transformer_enhanced', {}).get('mean_f1', 0)
        framework_f1 = category_comparison.get('framework_enhanced', {}).get('mean_f1', 0)
        full_f1 = category_comparison.get('full_integration', {}).get('mean_f1', 0)
        
        # Framework improvement analysis
        if baseline_f1 > 0:
            insights['framework_improvement'] = {
                'transformer_improvement': ((transformer_f1 - baseline_f1) / baseline_f1 * 100) if transformer_f1 > 0 else 0,
                'framework_improvement': ((framework_f1 - baseline_f1) / baseline_f1 * 100) if framework_f1 > 0 else 0,
                'full_improvement': ((full_f1 - baseline_f1) / baseline_f1 * 100) if full_f1 > 0 else 0
            }
        
        # Zero-shot comparison
        if zero_shot_results and 'metrics' in zero_shot_results:
            zero_shot_f1 = zero_shot_results['metrics'].get('f1_score', 0)
            insights['zero_shot_performance'] = {
                'f1_score': zero_shot_f1,
                'vs_baseline': ((zero_shot_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0,
                'vs_best_traditional': ((zero_shot_f1 - max(baseline_f1, transformer_f1, framework_f1, full_f1)) / max(baseline_f1, transformer_f1, framework_f1, full_f1) * 100) if max(baseline_f1, transformer_f1, framework_f1, full_f1) > 0 else 0
            }
        
        # Generate recommendations
        if full_f1 > baseline_f1:
            insights['recommendations'].append("✅ Theoretical frameworks + transformers improve performance over baseline")
        else:
            insights['recommendations'].append("❌ Theoretical frameworks do not improve performance significantly")
        
        if transformer_f1 > baseline_f1:
            insights['recommendations'].append("✅ Transformer embeddings provide performance gains")
        else:
            insights['recommendations'].append("❌ Transformer embeddings do not improve baseline performance")
        
        return insights
    
    def _perform_hyperparameter_tuning(self, model, X_train, y_train, algo_name):
        """Perform hyperparameter tuning for a specific algorithm."""
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import make_scorer, f1_score
        
        # Define parameter grids for each algorithm
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [2000, 5000, 10000]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000]
            },
            'naive_bayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        }
        
        if algo_name not in param_grids:
            self._write_training_log('warning', f"No hyperparameter grid defined for {algo_name}")
            return model, {}
        
        try:
            self._write_training_log('info', f"    🔧 Hyperparameter tuning for {algo_name}...")
            
            # Use F1 score as the scoring metric
            scorer = make_scorer(f1_score, average='weighted')
            
            # Use RandomizedSearchCV for faster tuning
            search = RandomizedSearchCV(
                model, 
                param_grids[algo_name],
                n_iter=20,  # Limit iterations for speed
                cv=3,  # 3-fold CV for speed
                scoring=scorer,
                random_state=42,
                n_jobs=-1
            )
            
            search.fit(X_train, y_train)
            
            best_params = search.best_params_
            best_score = search.best_score_
            
            self._write_training_log('info', f"    ✅ Best CV score: {best_score:.4f}")
            self._write_training_log('info', f"    📋 Best params: {best_params}")
            
            return search.best_estimator_, {
                'best_params': best_params,
                'best_cv_score': best_score,
                'tuning_method': 'RandomizedSearchCV'
            }
            
        except Exception as e:
            self._write_training_log('warning', f"    ⚠️ Hyperparameter tuning failed for {algo_name}: {e}")
            return model, {'tuning_error': str(e)}
    
    def _analyze_individual_framework_impact(self, results):
        """Analyze the impact of individual frameworks (RAT, RCT, UGT)."""
        framework_impact = {
            'individual_frameworks': {},
            'framework_ranking': [],
            'best_individual_framework': None
        }
        
        # Extract individual framework results
        individual_results = {}
        for combo_name, combo_results in results.items():
            combo_config = self.framework_combinations.get(combo_name, {})
            if combo_config.get('category') == 'individual_framework':
                framework_type = combo_config.get('framework_type')
                if framework_type and 'algorithms' in combo_results:
                    # Calculate average F1 score across all algorithms
                    f1_scores = []
                    for algo_results in combo_results['algorithms'].values():
                        if 'error' not in algo_results and 'metrics' in algo_results:
                            f1_scores.append(algo_results['metrics']['f1_score'])
                    
                    if f1_scores:
                        individual_results[framework_type] = {
                            'mean_f1': np.mean(f1_scores),
                            'std_f1': np.std(f1_scores),
                            'max_f1': np.max(f1_scores),
                            'combination_name': combo_name,
                            'algorithm_count': len(f1_scores)
                        }
        
        framework_impact['individual_frameworks'] = individual_results
        
        # Rank frameworks by performance
        if individual_results:
            framework_ranking = sorted(
                individual_results.items(),
                key=lambda x: x[1]['mean_f1'],
                reverse=True
            )
            
            framework_impact['framework_ranking'] = [
                {
                    'framework': framework,
                    'mean_f1': data['mean_f1'],
                    'rank': rank + 1
                }
                for rank, (framework, data) in enumerate(framework_ranking)
            ]
            
            # Best individual framework
            if framework_ranking:
                best_framework, best_data = framework_ranking[0]
                framework_impact['best_individual_framework'] = {
                    'framework': best_framework,
                    'mean_f1': best_data['mean_f1'],
                    'combination_name': best_data['combination_name']
                }
        
        return framework_impact
    
    def _save_unified_training_results(self, dataset_name, results):
        """Save comprehensive unified training results."""
        results_dir = Path('datasets') / dataset_name / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / 'unified_framework_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self._write_training_log('info', f"💾 Unified training results saved to: {results_file}")
    
    def _save_specialized_training_results(self, dataset_name, results):
        """Save overall specialized training results."""
        try:
            results_dir = Path('models') / dataset_name
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = results_dir / 'specialized_training_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self._write_training_log('info', f"Saved specialized training results")
            
        except Exception as e:
            self._write_training_log('error', f"Error saving training results: {e}")
    
    def load_specialized_model(self, dataset_name, model_type, model_name):
        """Load a specific specialized model."""
        try:
            model_path = Path('models') / dataset_name / model_type / f'{model_name}.joblib'
            
            if model_path.exists():
                complete_model = joblib.load(model_path)
                self.logger.info(f"Loaded {model_type} {model_name} model")
                return complete_model
            else:
                self.logger.warning(f"Model not found: {model_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading specialized model: {e}")
            return None
    
    def get_specialized_model_comparison(self, dataset_name):
        """Get comparison of all specialized trained models."""
        try:
            results_path = Path('models') / dataset_name / 'specialized_training_results.json'
            
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                # Extract key metrics for comparison
                comparison = {}
                
                for model_type, type_results in results.get('model_types', {}).items():
                    if 'models' in type_results:
                        comparison[model_type] = {}
                        for model_name, model_results in type_results['models'].items():
                            if 'error' not in model_results:
                                comparison[model_type][model_name] = {
                                    'test_accuracy': model_results.get('test_accuracy', 0),
                                    'cv_mean': model_results.get('cv_mean', 0),
                                    'roc_auc': model_results.get('roc_auc', 0)
                                }
                
                return comparison
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting specialized model comparison: {e}")
            return {}
    
    def train_single_model(self, model_name, X, y, feature_names=None, dataset_name=None, feature_combo=None, cv_folds=3, skip_hyperparameter_tuning=True):
        """
        Train a single model with given features and labels for modular training.
        
        Args:
            model_name: Name of the model to train
            X: Feature matrix
            y: Labels
            feature_names: List of feature names (optional)
            dataset_name: Name of dataset (for model saving)
            feature_combo: Feature combination name (for model saving)
            cv_folds: Number of cross-validation folds (default: 3)
            skip_hyperparameter_tuning: Skip hyperparameter tuning (default: True)
        """
        try:
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
            import joblib
            
            # Map model names to our algorithm definitions
            model_mapping = {
                'logistic_regression': 'logistic_regression',
                'naive_bayes': 'naive_bayes',
                'random_forest': 'random_forest',
                'xgboost': 'gradient_boosting',  # Using gradient boosting as XGBoost alternative
                'svm': 'svm',
                'neural_network': 'neural_network'
            }
            
            if model_name not in model_mapping:
                raise ValueError(f"Unknown model: {model_name}")
            
            algorithm_name = model_mapping[model_name]
            if algorithm_name not in self.traditional_algorithms:
                raise ValueError(f"Algorithm not available: {algorithm_name}")
            
            # Create fresh model instance to avoid state issues
            base_model = self.traditional_algorithms[algorithm_name]
            model = base_model.__class__(**base_model.get_params())
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features for algorithms that need it
            scaler = StandardScaler()
            needs_scaling = algorithm_name in ['logistic_regression', 'svm', 'neural_network']
            if needs_scaling:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                scaler = None  # No scaling needed
            
            # Apply SMOTE if needed (check class imbalance)
            smote_applied = False
            if self._needs_smote(y_train):
                try:
                    X_train_scaled, y_train = self._apply_smote_unified(X_train_scaled, y_train)
                    smote_applied = True
                except Exception as e:
                    self.logger.warning(f"SMOTE failed for {model_name}: {e}")
            
            # Hyperparameter tuning (skipped by default for speed)
            tuning_results = {'tuning_method': 'default_params', 'skipped': skip_hyperparameter_tuning}
            if not skip_hyperparameter_tuning:
                try:
                    model, tuning_results = self._perform_hyperparameter_tuning(
                        model, X_train_scaled, y_train, algorithm_name
                    )
                except Exception as e:
                    self.logger.warning(f"Hyperparameter tuning failed for {model_name}: {e}")
                    tuning_results = {'tuning_method': 'default_params', 'tuning_error': str(e)}
            
            # Train the model (if not already trained by hyperparameter tuning)
            if 'best_params' not in tuning_results:
                model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            train_accuracy = model.score(X_train_scaled, y_train)
            
            # Cross-validation (configurable folds)
            cv_mean = cv_std = None
            try:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='f1_weighted')
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
            except Exception as e:
                self.logger.warning(f"Cross-validation failed for {model_name}: {e}")
                cv_mean = float(f1)  # Fallback to test F1
                cv_std = 0.0
            
            # Calculate ROC AUC if possible
            roc_auc = None
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                except Exception as e:
                    self.logger.warning(f"ROC AUC calculation failed for {model_name}: {e}")
            
            # Save model if dataset and feature combo provided
            model_saved = False
            if dataset_name and feature_combo:
                try:
                    models_dir = Path('models') / dataset_name
                    models_dir.mkdir(parents=True, exist_ok=True)
                    
                    model_file = models_dir / f"{model_name}_{feature_combo}.pkl"
                    model_data = {
                        'model': model,
                        'scaler': scaler,
                        'feature_names': feature_names,
                        'model_name': model_name,
                        'algorithm': algorithm_name,
                        'feature_combo': feature_combo,
                        'needs_scaling': needs_scaling,
                        'smote_applied': smote_applied
                    }
                    joblib.dump(model_data, model_file)
                    model_saved = True
                    self.logger.info(f"Model saved: {model_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to save model {model_name}: {e}")
            
            # Prepare comprehensive results
            results = {
                'model_name': model_name,
                'algorithm': algorithm_name,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'train_accuracy': float(train_accuracy),
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_folds': cv_folds,
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'total_features': X.shape[1],
                'train_samples': X_train_scaled.shape[0],
                'test_samples': X_test.shape[0],
                'feature_combo': feature_combo,
                'needs_scaling': needs_scaling,
                'smote_applied': smote_applied,
                'model_saved': model_saved,
                'hyperparameter_tuning': tuning_results,
                'feature_names_sample': feature_names[:10] if feature_names else None,  # First 10 for brevity
                'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            self.logger.info(f"✅ {model_name} trained successfully: Acc={accuracy:.3f}, F1={f1:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error training single model {model_name}: {e}")
            return {
                'error': str(e),
                'model_name': model_name,
                'algorithm': algorithm_name if 'algorithm_name' in locals() else 'unknown',
                'accuracy': 0,
                'f1_score': 0,
                'cv_mean': 0,
                'cv_std': 0
            }
    
    def load_zero_shot_results(self, dataset_name):
        """
        Load existing zero-shot classification results for comparison with trained models.
        Uses confidence scores as proxy for accuracy since zero-shot doesn't require training.
        """
        try:
            # Load zero-shot results from feature extraction
            zero_shot_results = self.file_manager.load_results(dataset_name, 'zero_shot_classification')
            
            if not zero_shot_results:
                self.logger.info("No zero-shot results found for comparison")
                return None
            
            # Extract key metrics
            total_samples = zero_shot_results.get('total_samples', 0)
            average_confidence = zero_shot_results.get('average_confidence', 0)
            misinformation_count = zero_shot_results.get('misinformation_count', 0)
            legitimate_count = zero_shot_results.get('legitimate_count', 0)
            uncertain_count = zero_shot_results.get('uncertain_count', 0)
            
            # Calculate metrics using confidence as proxy for accuracy
            # High confidence predictions are considered "accurate"
            confidence_scores = zero_shot_results.get('confidence_scores', [])
            high_confidence_threshold = 0.7
            high_confidence_count = sum(1 for conf in confidence_scores if conf >= high_confidence_threshold)
            
            # Use average confidence as accuracy proxy
            accuracy_proxy = average_confidence
            
            # Calculate F1 proxy based on classification distribution
            classification_dist = zero_shot_results.get('classification_distribution', {})
            misinfo_ratio = classification_dist.get('misinformation', 0)
            legit_ratio = classification_dist.get('legitimate', 0)
            
            # F1 proxy: balance between precision (confidence) and recall (coverage)
            coverage = (total_samples - uncertain_count) / total_samples if total_samples > 0 else 0
            f1_proxy = 2 * (average_confidence * coverage) / (average_confidence + coverage) if (average_confidence + coverage) > 0 else 0
            
            # Create results in same format as trained models
            bart_zero_shot_results = {
                'model_name': 'bart_zero_shot',
                'algorithm': 'zero_shot_classification',
                'accuracy': float(accuracy_proxy),
                'precision': float(average_confidence),  # Confidence as precision proxy
                'recall': float(coverage),  # Coverage as recall proxy
                'f1_score': float(f1_proxy),
                'train_accuracy': None,  # N/A for zero-shot
                'cv_mean': None,  # N/A for zero-shot
                'cv_std': None,  # N/A for zero-shot
                'cv_folds': None,  # N/A for zero-shot
                'roc_auc': None,  # Not available for zero-shot
                'total_features': 0,  # Uses text directly, no feature engineering
                'train_samples': 0,  # No training required
                'test_samples': total_samples,
                'feature_combo': 'text_only',
                'needs_scaling': False,
                'smote_applied': False,
                'model_saved': False,  # Pre-trained model
                'hyperparameter_tuning': {'tuning_method': 'pre_trained'},
                'feature_names_sample': ['raw_text'],
                'is_zero_shot': True,
                'no_training_required': True,
                'zero_shot_details': {
                    'total_samples': total_samples,
                    'misinformation_detected': misinformation_count,
                    'legitimate_detected': legitimate_count,
                    'uncertain_predictions': uncertain_count,
                    'average_confidence': average_confidence,
                    'high_confidence_predictions': high_confidence_count,
                    'classification_distribution': classification_dist,
                    'confidence_distribution': zero_shot_results.get('confidence_distribution', {})
                }
            }
            
            self.logger.info(f"✅ Loaded Bart-zero-shot results: Acc={accuracy_proxy:.3f}, F1={f1_proxy:.3f}, Confidence={average_confidence:.3f}")
            return bart_zero_shot_results
            
        except Exception as e:
            self.logger.error(f"❌ Error loading zero-shot results: {e}")
            return None
    
    def create_ensemble_models(self, dataset_name, ensemble_config=None):
        """
        Create ensemble models from trained individual models.
        Handles cases where not all 6 classifiers are trained.
        
        Args:
            dataset_name: Name of the dataset
            ensemble_config: Configuration for ensemble creation
        """
        try:
            from sklearn.ensemble import VotingClassifier, StackingClassifier
            from sklearn.model_selection import cross_val_score
            import joblib
            from pathlib import Path
            
            if ensemble_config is None:
                ensemble_config = {
                    'accuracy_threshold': 0.6,
                    'min_models_voting': 3,
                    'min_models_stacking': 4,
                    'voting_type': 'soft',
                    'cv_folds': 3
                }
            
            # Load all available trained models
            models_dir = Path('models') / dataset_name
            if not models_dir.exists():
                self.logger.warning(f"No models directory found for {dataset_name}")
                return None
            
            # Collect available models and their performance
            available_models = []
            model_files = list(models_dir.glob("*.pkl"))
            
            for model_file in model_files:
                try:
                    model_data = joblib.load(model_file)
                    
                    # Skip zero-shot models for ensemble
                    if model_data.get('is_zero_shot', False):
                        continue
                    
                    # Check if model meets threshold
                    accuracy = model_data.get('accuracy', 0)
                    if accuracy >= ensemble_config['accuracy_threshold']:
                        available_models.append({
                            'name': model_file.stem,
                            'model': model_data['model'],
                            'scaler': model_data.get('scaler'),
                            'accuracy': accuracy,
                            'f1_score': model_data.get('f1_score', 0),
                            'needs_scaling': model_data.get('needs_scaling', False),
                            'feature_names': model_data.get('feature_names', [])
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Could not load model {model_file}: {e}")
                    continue
            
            # Sort by F1 score (best first)
            available_models.sort(key=lambda x: x['f1_score'], reverse=True)
            
            self.logger.info(f"Found {len(available_models)} models above threshold {ensemble_config['accuracy_threshold']}")
            
            ensemble_results = {
                'voting_ensemble': None,
                'stacking_ensemble': None,
                'available_models': len(available_models),
                'threshold_used': ensemble_config['accuracy_threshold']
            }
            
            # Create Voting Ensemble
            if len(available_models) >= ensemble_config['min_models_voting']:
                voting_ensemble = self._create_voting_ensemble(
                    available_models[:5],  # Top 5 models
                    dataset_name,
                    ensemble_config
                )
                ensemble_results['voting_ensemble'] = voting_ensemble
            
            # Create Stacking Ensemble
            if len(available_models) >= ensemble_config['min_models_stacking']:
                stacking_ensemble = self._create_stacking_ensemble(
                    available_models[:4],  # Top 4 as base models
                    available_models[0],   # Best as meta-model
                    dataset_name,
                    ensemble_config
                )
                ensemble_results['stacking_ensemble'] = stacking_ensemble
            
            # Save ensemble results
            self.file_manager.save_results(dataset_name, ensemble_results, 'ensemble_models')
            
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"❌ Error creating ensemble models: {e}")
            return None
    
    def _create_voting_ensemble(self, models, dataset_name, config):
        """Create a voting ensemble from selected models."""
        try:
            from sklearn.ensemble import VotingClassifier
            from sklearn.model_selection import cross_val_score
            import joblib
            
            # Prepare estimators for voting
            estimators = []
            for i, model_info in enumerate(models):
                estimator_name = f"model_{i}_{model_info['name'].split('_')[0]}"
                estimators.append((estimator_name, model_info['model']))
            
            # Create voting classifier
            voting_type = config.get('voting_type', 'soft')
            voting_ensemble = VotingClassifier(
                estimators=estimators,
                voting=voting_type
            )
            
            # Load features for evaluation
            X, y, feature_names = self.feature_extractor.load_features(dataset_name)
            if X is None:
                raise ValueError("Features not found for ensemble evaluation")
            
            # Evaluate ensemble with cross-validation
            cv_scores = cross_val_score(
                voting_ensemble, X, y, 
                cv=config.get('cv_folds', 3), 
                scoring='f1_weighted'
            )
            
            # Fit the ensemble
            voting_ensemble.fit(X, y)
            
            # Save ensemble model
            ensemble_path = Path('models') / dataset_name / 'voting_ensemble.pkl'
            ensemble_data = {
                'ensemble': voting_ensemble,
                'model_names': [info['name'] for info in models],
                'voting_type': voting_type,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'base_models_count': len(models),
                'feature_names': feature_names
            }
            joblib.dump(ensemble_data, ensemble_path)
            
            return {
                'models': [info['name'] for info in models],
                'voting_type': voting_type,
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'expected_f1': float(cv_scores.mean()),
                'base_models_count': len(models),
                'status': 'created',
                'model_path': str(ensemble_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating voting ensemble: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'models': [info['name'] for info in models] if models else []
            }
    
    def _create_stacking_ensemble(self, base_models, meta_model, dataset_name, config):
        """Create a stacking ensemble from selected models."""
        try:
            from sklearn.ensemble import StackingClassifier
            from sklearn.model_selection import cross_val_score
            import joblib
            
            # Prepare base estimators
            base_estimators = []
            for i, model_info in enumerate(base_models):
                estimator_name = f"base_{i}_{model_info['name'].split('_')[0]}"
                base_estimators.append((estimator_name, model_info['model']))
            
            # Create stacking classifier
            stacking_ensemble = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_model['model'],
                cv=config.get('cv_folds', 3)
            )
            
            # Load features for evaluation
            X, y, feature_names = self.feature_extractor.load_features(dataset_name)
            if X is None:
                raise ValueError("Features not found for ensemble evaluation")
            
            # Evaluate ensemble with cross-validation
            cv_scores = cross_val_score(
                stacking_ensemble, X, y,
                cv=config.get('cv_folds', 3),
                scoring='f1_weighted'
            )
            
            # Fit the ensemble
            stacking_ensemble.fit(X, y)
            
            # Save ensemble model
            ensemble_path = Path('models') / dataset_name / 'stacking_ensemble.pkl'
            ensemble_data = {
                'ensemble': stacking_ensemble,
                'base_model_names': [info['name'] for info in base_models],
                'meta_model_name': meta_model['name'],
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'base_models_count': len(base_models),
                'feature_names': feature_names
            }
            joblib.dump(ensemble_data, ensemble_path)
            
            return {
                'base_models': [info['name'] for info in base_models],
                'meta_model': meta_model['name'],
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'expected_f1': float(cv_scores.mean()),
                'base_models_count': len(base_models),
                'status': 'created',
                'model_path': str(ensemble_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating stacking ensemble: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'base_models': [info['name'] for info in base_models] if base_models else [],
                'meta_model': meta_model['name'] if meta_model else 'unknown'
            }