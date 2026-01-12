"""
Model Manager Module

This module provides comprehensive model management capabilities for handling
embeddings and model operations. It implements model loading, embedding generation,
and model lifecycle management for transformer models in machine learning pipelines.
"""

import numpy as np
import logging
from typing import Optional, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ModelManager:
    """
    Model Management Class
    
    Implements comprehensive model management for embeddings and model operations.
    Provides model loading, embedding generation, and lifecycle management for
    transformer models with efficient resource utilization and error handling.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._bert_model = None
        self._tokenizer = None
        
    def get_embeddings(self, text: str, model_name: str = 'bert-base-uncased') -> Optional[np.ndarray]:
        """
        Get embeddings for text using specified model.
        
        Args:
            text: Input text
            model_name: Model name (currently supports BERT variants)
            
        Returns:
            Embeddings array or None if failed
        """
        try:
            # Try to use transformers if available
            if self._bert_model is None:
                self._load_bert_model(model_name)
            
            if self._bert_model is None:
                return None
                
            # Tokenize and get embeddings
            inputs = self._tokenizer(text, return_tensors='pt', truncation=True, 
                                   padding=True, max_length=512)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = self._bert_model(**inputs)
                
            # Use [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
            return embeddings.flatten()
            
        except Exception as e:
            self.logger.warning(f"Could not generate embeddings: {e}")
            return None
    
    def _load_bert_model(self, model_name: str):
        """Load BERT model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.logger.info(f"Loading {model_name} model...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._bert_model = AutoModel.from_pretrained(model_name)
            self.logger.info(f"Successfully loaded {model_name}")
            
        except ImportError:
            self.logger.warning("Transformers library not available for embeddings")
            self._bert_model = None
            self._tokenizer = None
        except Exception as e:
            self.logger.warning(f"Could not load BERT model: {e}")
            self._bert_model = None
            self._tokenizer = None
    
    def get_model_info(self, model_name: str) -> dict:
        """Get information about a model."""
        return {
            'name': model_name,
            'type': 'transformer',
            'embedding_dim': 768 if 'base' in model_name else 1024,
            'max_length': 512
        }