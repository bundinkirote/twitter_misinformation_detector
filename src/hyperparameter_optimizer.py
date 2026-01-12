"""
Hyperparameter Optimization Module

This module provides comprehensive hyperparameter optimization capabilities for
machine learning models. It implements grid search and randomized search strategies
with cross-validation to find optimal parameter configurations for improved model
performance in misinformation detection tasks.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import joblib
from datetime import datetime
from src.utils.file_manager import FileManager
from src.feature_extractor import FeatureExtractor

class HyperparameterOptimizer:
    """
    Hyperparameter Optimization Class
    
    Implements systematic hyperparameter optimization using grid search and
    randomized search methodologies. Provides comprehensive parameter space
    exploration with cross-validation for optimal model configuration discovery
    and performance enhancement.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_manager = FileManager()
        self.feature_extractor = FeatureExtractor()
        
        # Define parameter grids for different models
        self.param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            },
            'naive_bayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000]
            }
        }
    
    def optimize_models(self, dataset_name, selected_models, optimization_method='grid_search', config=None):
        """Optimize hyperparameters for selected models."""
        self.logger.info(f"Starting hyperparameter optimization for dataset: {dataset_name}")
        
        try:
            # Load features
            X, y, feature_names = self.feature_extractor.load_features(dataset_name)
            
            if X is None:
                raise ValueError("Features not found. Please extract features first.")
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled, X_test_scaled = self.feature_extractor.scale_features(
                dataset_name, X_train, X_test
            )
            
            # Update parameter grids with config if provided
            if config:
                self._update_param_grids_from_config(config)
            
            optimization_results = {
                'dataset_name': dataset_name,
                'optimization_method': optimization_method,
                'optimization_date': datetime.now().isoformat(),
                'config_used': config,
                'models': {}
            }
            
            # Optimize each selected model
            for model_name in selected_models:
                full_model_name = self._map_model_name(model_name)
                self.logger.info(f"Optimizing {full_model_name}...")
                
                try:
                    if optimization_method == 'grid_search':
                        result = self._grid_search_optimization(
                            full_model_name, X_train_scaled, y_train, X_test_scaled, y_test
                        )
                    elif optimization_method == 'random_search':
                        result = self._random_search_optimization(
                            full_model_name, X_train_scaled, y_train, X_test_scaled, y_test
                        )
                    else:
                        raise ValueError(f"Unknown optimization method: {optimization_method}")
                    
                    optimization_results['models'][full_model_name] = result
                    
                    # Save optimized model
                    model_path = self.file_manager.get_model_path(dataset_name, f"{full_model_name}_optimized")
                    joblib.dump(result['best_estimator'], model_path)
                    
                    self.logger.info(f"Optimization completed for {full_model_name}. Best score: {result['best_score']:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error optimizing {full_model_name}: {e}")
                    optimization_results['models'][full_model_name] = {'error': str(e)}
            
            # Find best overall model
            best_model_name = None
            best_score = 0
            
            for model_name, result in optimization_results['models'].items():
                if 'test_metrics' in result and result['test_metrics']['f1_score'] > best_score:
                    best_score = result['test_metrics']['f1_score']
                    best_model_name = model_name
            
            if best_model_name:
                optimization_results['best_model'] = best_model_name
                optimization_results['best_score'] = best_score
                
                # Save best model separately
                best_model = optimization_results['models'][best_model_name]['best_estimator']
                best_model_path = self.file_manager.get_model_path(dataset_name, 'best_model')
                joblib.dump(best_model, best_model_path)
            
            # Save optimization results
            self.file_manager.save_results(dataset_name, optimization_results, 'hyperparameter_optimization')
            
            # Update dataset info
            self.file_manager.update_dataset_info(dataset_name, {
                'status': 'hyperparameters_optimized',
                'best_model': best_model_name,
                'best_score': best_score
            })
            
            self.logger.info("Hyperparameter optimization completed successfully")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {e}")
            raise
    
    def _grid_search_optimization(self, model_name, X_train, y_train, X_test, y_test):
        """Perform grid search optimization."""
        model = self._get_model_instance(model_name)
        param_grid = self.param_grids.get(model_name, {})
        
        if not param_grid:
            raise ValueError(f"No parameter grid defined for {model_name}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        test_metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0)
        }
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': best_model,
            'test_metrics': test_metrics,
            'cv_results': grid_search.cv_results_
        }
    
    def _random_search_optimization(self, model_name, X_train, y_train, X_test, y_test):
        """Perform random search optimization."""
        model = self._get_model_instance(model_name)
        param_distributions = self.param_grids.get(model_name, {})
        
        if not param_distributions:
            raise ValueError(f"No parameter distributions defined for {model_name}")
        
        # Perform random search
        random_search = RandomizedSearchCV(
            model, param_distributions, n_iter=50, cv=5, scoring='f1',
            n_jobs=-1, random_state=42, verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        # Evaluate on test set
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        test_metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0)
        }
        
        return {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_estimator': best_model,
            'test_metrics': test_metrics,
            'cv_results': random_search.cv_results_
        }
    
    def _map_model_name(self, short_name):
        """Map short model names from UI to full names."""
        name_mapping = {
            'lr': 'logistic_regression',
            'nb': 'naive_bayes',
            'gb': 'gradient_boosting',
            'svm': 'svm',
            'nn': 'neural_network',
            'rf': 'random_forest'
        }
        return name_mapping.get(short_name, short_name)
    
    def _get_model_instance(self, model_name):
        """Get model instance by name."""
        # Map short name to full name if needed
        full_name = self._map_model_name(model_name)
        
        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42),
            'naive_bayes': GaussianNB(),
            'neural_network': MLPClassifier(random_state=42)
        }
        
        if full_name not in models:
            raise ValueError(f"Unknown model: {full_name}")
        
        return models[full_name]
    
    def get_optimization_results(self, dataset_name):
        """Get the latest optimization results for a dataset."""
        return self.file_manager.load_results(dataset_name, 'hyperparameter_optimization')
    
    def compare_optimization_methods(self, dataset_name, selected_models):
        """Compare different optimization methods."""
        self.logger.info("Comparing optimization methods...")
        
        comparison_results = {
            'dataset_name': dataset_name,
            'comparison_date': datetime.now().isoformat(),
            'methods': {}
        }
        
        methods = ['grid_search', 'random_search']
        
        for method in methods:
            try:
                results = self.optimize_models(dataset_name, selected_models, method)
                comparison_results['methods'][method] = results
            except Exception as e:
                self.logger.error(f"Error with {method}: {e}")
                comparison_results['methods'][method] = {'error': str(e)}
        
        # Save comparison results
        self.file_manager.save_results(dataset_name, comparison_results, 'optimization_comparison')
        
        return comparison_results
    
    def _update_param_grids_from_config(self, config):
        """Update parameter grids based on user configuration."""
        # Update logistic regression parameters
        if 'logistic_regression' in config:
            lr_config = config['logistic_regression']
            self.param_grids['logistic_regression'] = {
                'C': [lr_config.get('C', 1.0)] if 'C' in lr_config else [0.01, 0.1, 1, 10, 100],
                'penalty': [lr_config.get('penalty', 'l2')] if 'penalty' in lr_config else ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000]
            }
        
        # Update random forest parameters
        if 'random_forest' in config:
            rf_config = config['random_forest']
            self.param_grids['random_forest'] = {
                'n_estimators': [rf_config.get('n_estimators', 100)] if 'n_estimators' in rf_config else [50, 100, 200],
                'max_depth': [rf_config.get('max_depth', None)] if 'max_depth' in rf_config else [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        
        # Update gradient boosting parameters
        if 'gradient_boosting' in config:
            gb_config = config['gradient_boosting']
            self.param_grids['gradient_boosting'] = {
                'n_estimators': [gb_config.get('n_estimators', 100)] if 'n_estimators' in gb_config else [50, 100, 200],
                'learning_rate': [gb_config.get('learning_rate', 0.1)] if 'learning_rate' in gb_config else [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        # Update SVM parameters
        if 'svm' in config:
            svm_config = config['svm']
            self.param_grids['svm'] = {
                'C': [svm_config.get('C', 1.0)] if 'C' in svm_config else [0.1, 1, 10, 100],
                'kernel': [svm_config.get('kernel', 'rbf')] if 'kernel' in svm_config else ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        
        # Update neural network parameters
        if 'neural_network' in config:
            nn_config = config['neural_network']
            hidden_size = nn_config.get('hidden_layer_sizes', 100)
            self.param_grids['neural_network'] = {
                'hidden_layer_sizes': [(hidden_size,)] if 'hidden_layer_sizes' in nn_config else [(50,), (100,), (200,)],
                'alpha': [nn_config.get('alpha', 0.0001)] if 'alpha' in nn_config else [0.0001, 0.001, 0.01],
                'activation': [nn_config.get('activation', 'relu')] if 'activation' in nn_config else ['relu', 'tanh', 'logistic'],
                'max_iter': [500, 1000]
            }