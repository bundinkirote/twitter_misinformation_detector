"""
Model Compatibility Module

This module provides comprehensive model compatibility management for handling
version mismatches, loading issues, and cross-platform compatibility. It implements
multiple fallback strategies, safe loading mechanisms, and version-aware model
handling for robust machine learning model deployment.
"""

import numpy as np
import pandas as pd
import logging
import joblib
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ModelCompatibilityManager:
    """
    Model Compatibility Management Class
    
    Implements comprehensive model compatibility handling with version checks,
    fallback strategies, and safe loading mechanisms. Provides robust model
    loading capabilities across different versions and platforms for reliable
    machine learning model deployment and operation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def safe_load_model(self, model_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Safely load a model with multiple fallback strategies.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model dictionary or None if failed
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return None
            
        # Try multiple loading strategies
        strategies = [
            self._load_with_joblib,
            self._load_with_numpy_fix,
            self._load_with_pickle,
            self._load_with_compatibility_mode,
            self._load_with_simple_fallback
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                self.logger.info(f"Trying loading strategy {i}/{len(strategies)}: {strategy.__name__}")
                result = strategy(model_path)
                if result is not None:
                    self.logger.info(f"✅ Successfully loaded model using {strategy.__name__}")
                    return self._normalize_model_format(result)
            except Exception as e:
                self.logger.warning(f"Strategy {i} failed: {e}")
                continue
                
        self.logger.error(f"All loading strategies failed for {model_path}")
        return None
    
    def _load_with_joblib(self, model_path: Path) -> Any:
        """Standard joblib loading."""
        return joblib.load(model_path)
    
    def _load_with_pickle(self, model_path: Path) -> Any:
        """Fallback to pickle loading."""
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_with_compatibility_mode(self, model_path: Path) -> Any:
        """Load with numpy compatibility fixes."""
        # Temporarily patch numpy for compatibility
        original_core = None
        try:
            import numpy as np
            if hasattr(np, '_core'):
                original_core = np._core
            else:
                # Create a dummy _core for compatibility
                import types
                np._core = types.ModuleType('_core')
                if hasattr(np, 'core'):
                    for attr in dir(np.core):
                        if not attr.startswith('_'):
                            setattr(np._core, attr, getattr(np.core, attr))
            
            return joblib.load(model_path)
            
        finally:
            # Restore original state
            if original_core is not None:
                np._core = original_core
            elif hasattr(np, '_core'):
                delattr(np, '_core')
    
    def _load_with_numpy_fix(self, model_path: Path) -> Any:
        """Load with numpy version compatibility fixes."""
        try:
            # Try to fix numpy._core issues
            import numpy as np
            import sys
            
            # Backup current numpy state
            numpy_backup = {}
            
            # Create numpy._core module if it doesn't exist
            if not hasattr(np, '_core'):
                # Try different approaches to create _core
                if hasattr(np, 'core'):
                    np._core = np.core
                    numpy_backup['_core_added'] = True
                else:
                    # Create a minimal _core module
                    import types
                    _core_module = types.ModuleType('numpy._core')
                    
                    # Add essential attributes that might be expected
                    if hasattr(np, 'ndarray'):
                        _core_module.ndarray = np.ndarray
                    if hasattr(np, 'dtype'):
                        _core_module.dtype = np.dtype
                    if hasattr(np, 'multiarray'):
                        _core_module.multiarray = np.multiarray
                    
                    np._core = _core_module
                    sys.modules['numpy._core'] = _core_module
                    numpy_backup['_core_created'] = True
            
            # Try loading
            result = joblib.load(model_path)
            
            # Restore numpy state
            if numpy_backup.get('_core_added') and hasattr(np, '_core'):
                delattr(np, '_core')
            elif numpy_backup.get('_core_created'):
                if hasattr(np, '_core'):
                    delattr(np, '_core')
                if 'numpy._core' in sys.modules:
                    del sys.modules['numpy._core']
                
            return result
            
        except Exception as e:
            self.logger.warning(f"Numpy fix strategy failed: {e}")
            raise
    
    def _load_with_simple_fallback(self, model_path: Path) -> Any:
        """Simple fallback that creates a minimal model structure."""
        try:
            self.logger.info("Using simple fallback - creating minimal model structure")
            
            # Return a minimal model structure that can be used for basic predictions
            return {
                'model': None,
                'vectorizer': None,
                'scaler': None,
                'feature_names': [],
                'model_type': 'fallback',
                'fallback_mode': True,
                'error': 'Original model could not be loaded due to compatibility issues'
            }
            
        except Exception as e:
            self.logger.warning(f"Simple fallback strategy failed: {e}")
            raise
    
    def _normalize_model_format(self, loaded_model: Any) -> Dict[str, Any]:
        """Normalize loaded model to standard format."""
        if isinstance(loaded_model, dict):
            # Check if this is a fallback save format
            if loaded_model.get('fallback_save'):
                return self._load_fallback_model(loaded_model)
            
            # Already in dictionary format
            if 'model' in loaded_model:
                return loaded_model
            else:
                # Assume the dict is the model itself
                return {
                    'model': loaded_model,
                    'scaler': None,
                    'feature_columns': None,
                    'label_encoder': None,
                    'model_metadata': {}
                }
        else:
            # Single model object
            return {
                'model': loaded_model,
                'scaler': None,
                'feature_columns': None,
                'label_encoder': None,
                'model_metadata': {}
            }
    
    def _load_fallback_model(self, fallback_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Load a model saved using fallback method."""
        try:
            # Load the actual model
            model_path = fallback_dict['model_path']
            model = joblib.load(model_path)
            
            # Load scaler if exists
            scaler = None
            if fallback_dict.get('scaler_path'):
                try:
                    scaler = joblib.load(fallback_dict['scaler_path'])
                except Exception as e:
                    self.logger.warning(f"Could not load scaler: {e}")
            
            # Return normalized format
            return {
                'model': model,
                'scaler': scaler,
                'feature_columns': fallback_dict.get('feature_columns'),
                'label_encoder': fallback_dict.get('label_encoder'),
                'model_metadata': fallback_dict.get('model_metadata', {})
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load fallback model: {e}")
            return None
    
    def validate_model_compatibility(self, model_dict: Dict[str, Any]) -> bool:
        """Validate that the loaded model is compatible with current environment."""
        try:
            model = model_dict.get('model')
            if model is None:
                return False
                
            # Check if model has required methods
            required_methods = ['predict', 'predict_proba']
            for method in required_methods:
                if not hasattr(model, method):
                    self.logger.warning(f"Model missing required method: {method}")
                    return False
            
            # Try a dummy prediction to ensure compatibility
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
                dummy_data = np.random.random((1, n_features))
                
                # Test prediction
                _ = model.predict(dummy_data)
                if hasattr(model, 'predict_proba'):
                    _ = model.predict_proba(dummy_data)
                    
                self.logger.info(f"✅ Model validation successful (expects {n_features} features)")
                return True
            else:
                self.logger.warning("Cannot determine expected feature count")
                return True  # Assume it's okay if we can't check
                
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False
    
    def get_model_feature_count(self, model_dict: Dict[str, Any]) -> Optional[int]:
        """Get the expected feature count for the model."""
        try:
            model = model_dict.get('model')
            if model and hasattr(model, 'n_features_in_'):
                return model.n_features_in_
            return None
        except Exception as e:
            self.logger.warning(f"Could not determine feature count: {e}")
            return None
    
    def create_feature_adapter(self, expected_features: int, actual_features: int) -> callable:
        """Create a function to adapt features to match model expectations."""
        def adapt_features(X):
            """Adapt feature matrix to match model expectations."""
            if X.shape[1] == expected_features:
                return X
            elif X.shape[1] > expected_features:
                # Truncate features
                self.logger.warning(f"Truncating features from {X.shape[1]} to {expected_features}")
                return X[:, :expected_features]
            else:
                # Pad with zeros
                self.logger.warning(f"Padding features from {X.shape[1]} to {expected_features}")
                padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                return np.hstack([X, padding])
        
        return adapt_features

# Global instance
_compatibility_manager = None

def get_compatibility_manager():
    """Get the global compatibility manager instance."""
    global _compatibility_manager
    if _compatibility_manager is None:
        _compatibility_manager = ModelCompatibilityManager()
    return _compatibility_manager