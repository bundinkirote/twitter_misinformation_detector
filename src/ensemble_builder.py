"""
Ensemble Builder Module

This module provides functionality for creating and managing ensemble models
from individual trained classifiers. It supports both voting and stacking
ensemble methods and packages them for deployment and portability.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from src.utils.file_manager import FileManager

class EnsembleBuilder:
    """
    Ensemble Builder Class
    
    Constructs ensemble models from individual trained classifiers using
    voting and stacking methods. Provides functionality for model packaging,
    version management, and deployment preparation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_manager = FileManager()
        
        # Define supported machine learning algorithms
        self.supported_algorithms = [
            'logistic_regression', 'random_forest', 'naive_bayes', 
            'xgboost', 'svm', 'neural_network'
        ]
        
        # Define feature combination priority order for model selection
        self.feature_priority = [
            'complete_model',
            'framework_embeddings_all', 
            'transformer_embeddings',
            'behavioral_features',
            'base_model'
        ]
    
    def get_available_models(self, dataset_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve all available trained models for a specified dataset.
        
        Args:
            dataset_name: Name of the dataset to search for models
            
        Returns:
            Dictionary containing available models organized by algorithm and feature combination
        """
        models_dir = Path('models') / dataset_name
        
        if not models_dir.exists():
            return {}
        
        available_models = {}
        
        for algorithm in self.supported_algorithms:
            algorithm_models = {}
            
            # Iterate through feature combinations in priority order
            for feature_combo in self.feature_priority:
                model_file = models_dir / f"{algorithm}_{feature_combo}.pkl"
                
                if model_file.exists():
                    try:
                        # Load and validate model file
                        model_data = joblib.load(model_file)
                        
                        if 'model' in model_data and model_data['model'] is not None:
                            algorithm_models[feature_combo] = {
                                'file_path': model_file,
                                'model_data': model_data,
                                'feature_names': model_data.get('feature_names', []),
                                'scaler': model_data.get('scaler'),
                                'training_date': model_data.get('training_date', ''),
                                'performance': model_data.get('performance', {})
                            }
                    except Exception as e:
                        self.logger.warning(f"Could not load {model_file}: {e}")
                        continue
            
            if algorithm_models:
                available_models[algorithm] = algorithm_models
        
        return available_models
    
    def select_best_models(self, available_models: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Select the optimal model for each algorithm based on feature priority and performance.
        
        Args:
            available_models: Dictionary of available models by algorithm
            
        Returns:
            Dictionary containing the best model for each algorithm
        """
        best_models = {}
        
        for algorithm, models in available_models.items():
            # Prioritize complete_model if available
            if 'complete_model' in models:
                best_models[algorithm] = models['complete_model']
                continue
            
            # Select based on feature combination priority
            for feature_combo in self.feature_priority:
                if feature_combo in models:
                    best_models[algorithm] = models[feature_combo]
                    break
        
        return best_models
    
    def create_voting_ensemble(self, dataset_name: str, voting_type: str = 'soft') -> Optional[Dict[str, Any]]:
        """
        Create a voting ensemble classifier from available individual models.
        
        Args:
            dataset_name: Name of the dataset
            voting_type: Type of voting ('soft' or 'hard')
            
        Returns:
            Dictionary containing the voting ensemble and metadata, or None if creation fails
        """
        self.logger.info(f"Creating {voting_type} voting ensemble for {dataset_name}")
        
        # Get available models
        available_models = self.get_available_models(dataset_name)
        
        if len(available_models) < 2:
            self.logger.warning(f"Need at least 2 algorithms for ensemble, found {len(available_models)}")
            return None
        
        # Select best models
        best_models = self.select_best_models(available_models)
        
        if len(best_models) < 2:
            self.logger.warning(f"Could not select enough models for ensemble")
            return None
        
        # Create estimators list for VotingClassifier
        estimators = []
        model_info = {}
        
        for algorithm, model_data in best_models.items():
            try:
                model = model_data['model_data']['model']
                estimators.append((algorithm, model))
                
                model_info[algorithm] = {
                    'feature_combination': self._get_feature_combo_from_models(model_data),
                    'feature_names': model_data['feature_names'],
                    'performance': model_data['performance']
                }
                
            except Exception as e:
                self.logger.error(f"Error preparing {algorithm} for ensemble: {e}")
                continue
        
        if len(estimators) < 2:
            self.logger.error("Not enough valid models for ensemble")
            return None
        
        # Create voting ensemble
        try:
            voting_ensemble = VotingClassifier(
                estimators=estimators,
                voting=voting_type
            )
            
            # Get common feature names (use the most complete set)
            all_feature_names = []
            for model_data in best_models.values():
                features = model_data['feature_names']
                if len(features) > len(all_feature_names):
                    all_feature_names = features
            
            # Get common scaler (use from complete_model if available)
            scaler = None
            for model_data in best_models.values():
                if model_data.get('scaler') is not None:
                    scaler = model_data['scaler']
                    break
            
            ensemble_package = {
                'ensemble': voting_ensemble,
                'ensemble_type': f'{voting_type}_voting',
                'base_models_info': model_info,
                'base_models_count': len(estimators),
                'feature_names': all_feature_names,
                'scaler': scaler,
                'dataset_name': dataset_name,
                'creation_date': datetime.now().isoformat(),
                'algorithms_used': list(best_models.keys())
            }
            
            self.logger.info(f"Created {voting_type} voting ensemble with {len(estimators)} models")
            return ensemble_package
            
        except Exception as e:
            self.logger.error(f"Error creating voting ensemble: {e}")
            return None
    
    def create_stacking_ensemble(self, dataset_name: str, meta_learner=None) -> Optional[Dict[str, Any]]:
        """Create a stacking ensemble from available models."""
        self.logger.info(f"Creating stacking ensemble for {dataset_name}")
        
        # Get available models
        available_models = self.get_available_models(dataset_name)
        
        if len(available_models) < 3:
            self.logger.warning(f"Need at least 3 algorithms for stacking, found {len(available_models)}")
            return None
        
        # Select best models
        best_models = self.select_best_models(available_models)
        
        if len(best_models) < 3:
            self.logger.warning(f"Could not select enough models for stacking ensemble")
            return None
        
        # Create estimators list for StackingClassifier
        estimators = []
        model_info = {}
        
        for algorithm, model_data in best_models.items():
            try:
                model = model_data['model_data']['model']
                estimators.append((algorithm, model))
                
                model_info[algorithm] = {
                    'feature_combination': self._get_feature_combo_from_models(model_data),
                    'feature_names': model_data['feature_names'],
                    'performance': model_data['performance']
                }
                
            except Exception as e:
                self.logger.error(f"Error preparing {algorithm} for stacking: {e}")
                continue
        
        if len(estimators) < 3:
            self.logger.error("Not enough valid models for stacking ensemble")
            return None
        
        # Create stacking ensemble
        try:
            if meta_learner is None:
                meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            
            stacking_ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=3,
                stack_method='predict_proba'
            )
            
            # Get common feature names and scaler
            all_feature_names = []
            for model_data in best_models.values():
                features = model_data['feature_names']
                if len(features) > len(all_feature_names):
                    all_feature_names = features
            
            scaler = None
            for model_data in best_models.values():
                if model_data.get('scaler') is not None:
                    scaler = model_data['scaler']
                    break
            
            ensemble_package = {
                'ensemble': stacking_ensemble,
                'ensemble_type': 'stacking',
                'base_models_info': model_info,
                'base_models_count': len(estimators),
                'meta_learner': str(meta_learner),
                'feature_names': all_feature_names,
                'scaler': scaler,
                'dataset_name': dataset_name,
                'creation_date': datetime.now().isoformat(),
                'algorithms_used': list(best_models.keys())
            }
            
            self.logger.info(f"Created stacking ensemble with {len(estimators)} models")
            return ensemble_package
            
        except Exception as e:
            self.logger.error(f"Error creating stacking ensemble: {e}")
            return None
    
    def create_complete_package(self, dataset_name: str, package_type: str = 'base') -> bool:
        """Create a complete ensemble package with all models and components."""
        self.logger.info(f"Creating complete {package_type} ensemble package for {dataset_name}")
        
        try:
            # Get available models
            available_models = self.get_available_models(dataset_name)
            
            if len(available_models) < 2:
                self.logger.warning(f"Need at least 2 algorithms for ensemble package, found {len(available_models)}")
                return False
            
            # Load training data for ensemble training
            from src.feature_extractor import FeatureExtractor
            feature_extractor = FeatureExtractor()
            
            # Try to load complete_model features first
            X, y, feature_names = feature_extractor.load_features(dataset_name, 'complete_model')
            
            if X is None:
                # Fallback to any available features
                X, y, feature_names = feature_extractor.load_features(dataset_name)
            
            if X is None or y is None:
                self.logger.error(f"Could not load features for {dataset_name}")
                return False
            
            # Create ensemble models
            voting_package = self.create_voting_ensemble(dataset_name, 'soft')
            stacking_package = self.create_stacking_ensemble(dataset_name)
            
            # Train ensembles if created successfully
            trained_voting = None
            trained_stacking = None
            
            if voting_package:
                try:
                    voting_ensemble = voting_package['ensemble']
                    if voting_package['scaler']:
                        X_scaled = voting_package['scaler'].transform(X)
                    else:
                        X_scaled = X
                    
                    voting_ensemble.fit(X_scaled, y)
                    
                    # Evaluate
                    from sklearn.model_selection import cross_val_score
                    train_score = voting_ensemble.score(X_scaled, y)
                    cv_scores = cross_val_score(voting_ensemble, X_scaled, y, cv=3, scoring='f1_weighted')
                    
                    voting_package['performance'] = {
                        'train_accuracy': float(train_score),
                        'cv_f1_mean': float(cv_scores.mean()),
                        'cv_f1_std': float(cv_scores.std())
                    }
                    trained_voting = voting_package
                    
                except Exception as e:
                    self.logger.error(f"Error training voting ensemble: {e}")
            
            if stacking_package:
                try:
                    stacking_ensemble = stacking_package['ensemble']
                    if stacking_package['scaler']:
                        X_scaled = stacking_package['scaler'].transform(X)
                    else:
                        X_scaled = X
                    
                    stacking_ensemble.fit(X_scaled, y)
                    
                    # Evaluate
                    train_score = stacking_ensemble.score(X_scaled, y)
                    cv_scores = cross_val_score(stacking_ensemble, X_scaled, y, cv=3, scoring='f1_weighted')
                    
                    stacking_package['performance'] = {
                        'train_accuracy': float(train_score),
                        'cv_f1_mean': float(cv_scores.mean()),
                        'cv_f1_std': float(cv_scores.std())
                    }
                    trained_stacking = stacking_package
                    
                except Exception as e:
                    self.logger.error(f"Error training stacking ensemble: {e}")
            
            # Create complete package
            complete_package = {
                'dataset_name': dataset_name,
                'package_type': package_type,
                'creation_date': datetime.now().isoformat(),
                'voting_ensemble': trained_voting,
                'stacking_ensemble': trained_stacking,
                'individual_models': {},
                'feature_extractor_config': {
                    'feature_names': feature_names,
                    'feature_count': len(feature_names) if feature_names else X.shape[1]
                },
                'scaler': None,
                'metadata': {
                    'training_samples': X.shape[0],
                    'training_features': X.shape[1],
                    'algorithms_available': list(available_models.keys()),
                    'ensemble_count': sum([1 for x in [trained_voting, trained_stacking] if x is not None]),
                    'best_ensemble': None,
                    'package_version': '1.0'
                }
            }
            
            # Add individual models to package
            best_models = self.select_best_models(available_models)
            for algorithm, model_data in best_models.items():
                complete_package['individual_models'][algorithm] = {
                    'model': model_data['model_data']['model'],
                    'scaler': model_data['model_data'].get('scaler'),
                    'feature_names': model_data['feature_names'],
                    'performance': model_data['performance'],
                    'feature_combination': self._get_feature_combo_from_models(model_data)
                }
            
            # Set common scaler (use from complete_model if available)
            for model_data in best_models.values():
                if model_data.get('model_data', {}).get('scaler') is not None:
                    complete_package['scaler'] = model_data['model_data']['scaler']
                    break
            
            # Determine best ensemble based on performance
            if trained_stacking and trained_voting:
                stacking_f1 = trained_stacking['performance']['cv_f1_mean']
                voting_f1 = trained_voting['performance']['cv_f1_mean']
                complete_package['metadata']['best_ensemble'] = 'stacking' if stacking_f1 >= voting_f1 else 'voting'
            elif trained_stacking:
                complete_package['metadata']['best_ensemble'] = 'stacking'
            elif trained_voting:
                complete_package['metadata']['best_ensemble'] = 'voting'
            else:
                complete_package['metadata']['best_ensemble'] = 'individual'
            
            # Save complete package
            package_filename = f"{package_type}_ensemble_package.pkl"
            package_path = Path('models') / dataset_name / package_filename
            package_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(complete_package, package_path)
            
            # Save metadata separately for easy inspection
            metadata_path = package_path.with_suffix('.json')
            metadata = {
                'dataset_name': dataset_name,
                'package_type': package_type,
                'creation_date': complete_package['creation_date'],
                'algorithms_available': complete_package['metadata']['algorithms_available'],
                'ensemble_count': complete_package['metadata']['ensemble_count'],
                'best_ensemble': complete_package['metadata']['best_ensemble'],
                'voting_performance': trained_voting['performance'] if trained_voting else None,
                'stacking_performance': trained_stacking['performance'] if trained_stacking else None,
                'individual_models_count': len(complete_package['individual_models'])
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Complete {package_type} ensemble package saved to {package_path}")
            self.logger.info(f"Package contains: {complete_package['metadata']['ensemble_count']} ensembles, {len(complete_package['individual_models'])} individual models")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating complete ensemble package: {e}")
            return False

    def train_and_save_ensemble(self, dataset_name: str, ensemble_type: str = 'both') -> Dict[str, bool]:
        """Train ensemble models on the dataset and save them."""
        self.logger.info(f"Training and saving ensemble models for {dataset_name}")
        
        results = {
            'voting_ensemble': False,
            'stacking_ensemble': False
        }
        
        # Load training data
        try:
            from src.feature_extractor import FeatureExtractor
            feature_extractor = FeatureExtractor()
            
            # Try to load complete_model features first
            X, y, feature_names = feature_extractor.load_features(dataset_name, 'complete_model')
            
            if X is None:
                # Fallback to any available features
                X, y, feature_names = feature_extractor.load_features(dataset_name)
            
            if X is None or y is None:
                self.logger.error(f"Could not load features for {dataset_name}")
                return results
            
            self.logger.info(f"Loaded training data: {X.shape[0]} samples, {X.shape[1]} features")
            
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            return results
        
        # Create and train voting ensemble
        if ensemble_type in ['voting', 'both']:
            voting_package = self.create_voting_ensemble(dataset_name, 'soft')
            
            if voting_package:
                try:
                    # Train the ensemble
                    self.logger.info("Training voting ensemble...")
                    voting_ensemble = voting_package['ensemble']
                    
                    # Use the scaler if available
                    if voting_package['scaler']:
                        X_scaled = voting_package['scaler'].transform(X)
                    else:
                        X_scaled = X
                    
                    voting_ensemble.fit(X_scaled, y)
                    
                    # Evaluate ensemble
                    train_score = voting_ensemble.score(X_scaled, y)
                    cv_scores = cross_val_score(voting_ensemble, X_scaled, y, cv=3, scoring='f1_weighted')
                    
                    # Add performance metrics
                    voting_package['performance'] = {
                        'train_accuracy': float(train_score),
                        'cv_f1_mean': float(cv_scores.mean()),
                        'cv_f1_std': float(cv_scores.std()),
                        'training_samples': X.shape[0],
                        'training_features': X.shape[1]
                    }
                    
                    # Save voting ensemble
                    voting_path = Path('models') / dataset_name / 'voting_ensemble.pkl'
                    voting_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    joblib.dump(voting_package, voting_path)
                    
                    # Save metadata
                    metadata_path = voting_path.with_suffix('.json')
                    with open(metadata_path, 'w') as f:
                        metadata = voting_package.copy()
                        del metadata['ensemble']  # Remove non-serializable ensemble
                        if 'scaler' in metadata:
                            del metadata['scaler']  # Remove non-serializable scaler
                        json.dump(metadata, f, indent=2)
                    
                    results['voting_ensemble'] = True
                    self.logger.info(f"Voting ensemble saved to {voting_path}")
                    
                except Exception as e:
                    self.logger.error(f"Error training/saving voting ensemble: {e}")
        
        # Create and train stacking ensemble
        if ensemble_type in ['stacking', 'both']:
            stacking_package = self.create_stacking_ensemble(dataset_name)
            
            if stacking_package:
                try:
                    # Train the ensemble
                    self.logger.info("Training stacking ensemble...")
                    stacking_ensemble = stacking_package['ensemble']
                    
                    # Use the scaler if available
                    if stacking_package['scaler']:
                        X_scaled = stacking_package['scaler'].transform(X)
                    else:
                        X_scaled = X
                    
                    stacking_ensemble.fit(X_scaled, y)
                    
                    # Evaluate ensemble
                    train_score = stacking_ensemble.score(X_scaled, y)
                    cv_scores = cross_val_score(stacking_ensemble, X_scaled, y, cv=3, scoring='f1_weighted')
                    
                    # Add performance metrics
                    stacking_package['performance'] = {
                        'train_accuracy': float(train_score),
                        'cv_f1_mean': float(cv_scores.mean()),
                        'cv_f1_std': float(cv_scores.std()),
                        'training_samples': X.shape[0],
                        'training_features': X.shape[1]
                    }
                    
                    # Save stacking ensemble
                    stacking_path = Path('models') / dataset_name / 'stacking_ensemble.pkl'
                    stacking_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    joblib.dump(stacking_package, stacking_path)
                    
                    # Save metadata
                    metadata_path = stacking_path.with_suffix('.json')
                    with open(metadata_path, 'w') as f:
                        metadata = stacking_package.copy()
                        del metadata['ensemble']  # Remove non-serializable ensemble
                        if 'scaler' in metadata:
                            del metadata['scaler']  # Remove non-serializable scaler
                        json.dump(metadata, f, indent=2)
                    
                    results['stacking_ensemble'] = True
                    self.logger.info(f"Stacking ensemble saved to {stacking_path}")
                    
                except Exception as e:
                    self.logger.error(f"Error training/saving stacking ensemble: {e}")
        
        return results
    
    def _get_feature_combo_from_models(self, model_data: Dict[str, Any]) -> str:
        """Extract feature combination from model data."""
        # Try to determine from file path or metadata
        file_path = model_data.get('file_path', '')
        if file_path:
            for combo in self.feature_priority:
                if combo in str(file_path):
                    return combo
        return 'unknown'
    
    def get_ensemble_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about available ensemble models."""
        models_dir = Path('models') / dataset_name
        
        ensemble_info = {
            'dataset_name': dataset_name,
            'voting_ensemble': None,
            'stacking_ensemble': None,
            'individual_models_count': 0,
            'ensemble_ready': False
        }
        
        if not models_dir.exists():
            return ensemble_info
        
        # Check for ensemble models
        voting_path = models_dir / 'voting_ensemble.pkl'
        stacking_path = models_dir / 'stacking_ensemble.pkl'
        
        if voting_path.exists():
            try:
                voting_data = joblib.load(voting_path)
                ensemble_info['voting_ensemble'] = {
                    'available': True,
                    'algorithms_count': voting_data.get('base_models_count', 0),
                    'algorithms_used': voting_data.get('algorithms_used', []),
                    'performance': voting_data.get('performance', {}),
                    'creation_date': voting_data.get('creation_date', '')
                }
            except Exception as e:
                self.logger.error(f"Error loading voting ensemble info: {e}")
        
        if stacking_path.exists():
            try:
                stacking_data = joblib.load(stacking_path)
                ensemble_info['stacking_ensemble'] = {
                    'available': True,
                    'algorithms_count': stacking_data.get('base_models_count', 0),
                    'algorithms_used': stacking_data.get('algorithms_used', []),
                    'performance': stacking_data.get('performance', {}),
                    'creation_date': stacking_data.get('creation_date', ''),
                    'meta_learner': stacking_data.get('meta_learner', '')
                }
            except Exception as e:
                self.logger.error(f"Error loading stacking ensemble info: {e}")
        
        # Count individual models
        individual_models = self.get_available_models(dataset_name)
        ensemble_info['individual_models_count'] = len(individual_models)
        ensemble_info['ensemble_ready'] = len(individual_models) >= 2
        
        return ensemble_info
    
    def build_all_ensembles(self) -> Dict[str, Dict[str, bool]]:
        """Build ensemble models for all datasets that have sufficient individual models."""
        self.logger.info("Building ensemble models for all ready datasets...")
        
        results = {}
        models_dir = Path('models')
        
        if not models_dir.exists():
            self.logger.warning("No models directory found")
            return results
        
        # Check each dataset
        for dataset_dir in models_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            self.logger.info(f"Checking dataset: {dataset_name}")
            
            # Check if dataset has enough models for ensemble
            available_models = self.get_available_models(dataset_name)
            
            if len(available_models) >= 2:
                self.logger.info(f"Building ensembles for {dataset_name} ({len(available_models)} algorithms)")
                results[dataset_name] = self.train_and_save_ensemble(dataset_name, 'both')
            else:
                self.logger.info(f"Skipping {dataset_name} - only {len(available_models)} algorithms available")
                results[dataset_name] = {'voting_ensemble': False, 'stacking_ensemble': False}
        
        return results
    
    def get_available_packages(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about available ensemble packages."""
        models_dir = Path('models') / dataset_name
        
        package_info = {
            'dataset_name': dataset_name,
            'base_package': None,
            'tuned_package': None,
            'active_package': None,
            'has_packages': False
        }
        
        if not models_dir.exists():
            return package_info
        
        # Check for base package
        base_path = models_dir / 'base_ensemble_package.pkl'
        if base_path.exists():
            try:
                # Load metadata instead of full package for efficiency
                metadata_path = base_path.with_suffix('.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    package_info['base_package'] = metadata
                else:
                    # Fallback: load package header
                    package = joblib.load(base_path)
                    package_info['base_package'] = {
                        'creation_date': package.get('creation_date', ''),
                        'best_ensemble': package.get('metadata', {}).get('best_ensemble', ''),
                        'ensemble_count': package.get('metadata', {}).get('ensemble_count', 0)
                    }
                package_info['has_packages'] = True
            except Exception as e:
                self.logger.error(f"Error loading base package info: {e}")
        
        # Check for tuned package
        tuned_path = models_dir / 'tuned_ensemble_package.pkl'
        if tuned_path.exists():
            try:
                metadata_path = tuned_path.with_suffix('.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    package_info['tuned_package'] = metadata
                else:
                    package = joblib.load(tuned_path)
                    package_info['tuned_package'] = {
                        'creation_date': package.get('creation_date', ''),
                        'best_ensemble': package.get('metadata', {}).get('best_ensemble', ''),
                        'ensemble_count': package.get('metadata', {}).get('ensemble_count', 0)
                    }
                package_info['has_packages'] = True
            except Exception as e:
                self.logger.error(f"Error loading tuned package info: {e}")
        
        # Determine active package (prefer tuned over base)
        if package_info['tuned_package']:
            package_info['active_package'] = 'tuned'
        elif package_info['base_package']:
            package_info['active_package'] = 'base'
        
        return package_info
    
    def load_package_for_prediction(self, dataset_name: str, prefer_tuned: bool = True) -> Optional[Dict[str, Any]]:
        """Load the best available package for prediction."""
        models_dir = Path('models') / dataset_name
        
        if not models_dir.exists():
            return None
        
        # Determine which package to load
        package_to_load = None
        
        if prefer_tuned:
            tuned_path = models_dir / 'tuned_ensemble_package.pkl'
            if tuned_path.exists():
                package_to_load = tuned_path
        
        if package_to_load is None:
            base_path = models_dir / 'base_ensemble_package.pkl'
            if base_path.exists():
                package_to_load = base_path
        
        if package_to_load is None:
            self.logger.warning(f"No ensemble packages found for {dataset_name}")
            return None
        
        try:
            package = joblib.load(package_to_load)
            self.logger.info(f"Loaded {package['package_type']} package for {dataset_name}")
            return package
        except Exception as e:
            self.logger.error(f"Error loading package {package_to_load}: {e}")
            return None
    
    def compare_packages(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Compare base and tuned packages if both exist."""
        package_info = self.get_available_packages(dataset_name)
        
        if not (package_info['base_package'] and package_info['tuned_package']):
            return None
        
        comparison = {
            'dataset_name': dataset_name,
            'base_package': package_info['base_package'],
            'tuned_package': package_info['tuned_package'],
            'recommendation': 'tuned',  # Default to tuned if available
            'improvements': {}
        }
        
        # Compare performance if available
        base_perf = package_info['base_package']
        tuned_perf = package_info['tuned_package']
        
        # Add performance comparison logic here if needed
        # For now, always recommend tuned if available
        
        return comparison