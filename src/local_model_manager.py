"""
Local Model Manager Module

This module provides comprehensive local model management capabilities for transformer
models within the project structure. It implements model downloading, caching, version
control, and portability features for efficient transformer model lifecycle management
in machine learning pipelines.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import requests
from tqdm import tqdm

# Transformers imports with error handling
try:
    import warnings
    # Suppress transformers warnings about unused weights
    warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
    
    from transformers import AutoTokenizer, AutoModel
    from transformers import logging as transformers_logging
    # Set transformers logging to ERROR to suppress warnings
    transformers_logging.set_verbosity_error()
    
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class LocalModelManager:
    """
    Local Model Management Class
    
    Implements comprehensive local management of transformer models including
    downloading, caching, version control, and portability features. Provides
    efficient model lifecycle management within project structure for optimal
    performance and deployment flexibility.
    """
    
    def __init__(self, models_dir: str = "local_models"):
        self.logger = logging.getLogger(__name__)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'bert-base-uncased': {
                'type': 'transformer',
                'description': 'BERT base model for English text',
                'size_mb': 440,
                'use_case': 'English text embeddings'
            },
            'xlm-roberta-base': {
                'type': 'transformer', 
                'description': 'XLM-RoBERTa base model for multilingual text',
                'size_mb': 1100,
                'use_case': 'Multilingual text embeddings (Swahili, English, etc.)'
            },
            'distilbert-base-multilingual-cased': {
                'type': 'transformer',
                'description': 'DistilBERT multilingual model (lighter than XLM-RoBERTa)',
                'size_mb': 540,
                'use_case': 'Multilingual text embeddings (lighter alternative)'
            },
            'all-MiniLM-L6-v2': {
                'type': 'sentence_transformer',
                'description': 'Sentence transformer for semantic similarity',
                'size_mb': 90,
                'use_case': 'Sentence embeddings and similarity'
            }
        }
        
        # Load model registry
        self.model_registry = self._load_model_registry()
        
        self.logger.info(f"Local model manager initialized at: {self.models_dir}")
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load model registry from local file."""
        registry_file = self.models_dir / "model_registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading model registry: {e}")
        
        return {
            'models': {},
            'last_updated': None,
            'total_size_mb': 0
        }
    
    def _save_model_registry(self):
        """Save model registry to local file."""
        registry_file = self.models_dir / "model_registry.json"
        
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving model registry: {e}")
    
    def download_model(self, model_name: str, force_download: bool = False) -> bool:
        """
        Download and save model locally.
        
        Args:
            model_name: Name of the model to download
            force_download: Force re-download even if model exists
        """
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers not available. Install with: pip install transformers sentence-transformers")
            return False
        
        if model_name not in self.model_configs:
            self.logger.error(f"Unknown model: {model_name}")
            return False
        
        model_dir = self.models_dir / model_name
        
        # Check if model already exists
        if model_dir.exists() and not force_download:
            if self._verify_model_integrity(model_name):
                self.logger.info(f"Model {model_name} already exists and is valid")
                return True
            else:
                self.logger.warning(f"Model {model_name} exists but is corrupted, re-downloading...")
                shutil.rmtree(model_dir)
        
        try:
            # First, try to copy from Hugging Face cache
            if not force_download and self._copy_from_hf_cache(model_name, model_dir):
                self.logger.info(f"SUCCESS: Copied {model_name} from Hugging Face cache")
                
                # Update registry
                self.model_registry['models'][model_name] = {
                    'downloaded_at': datetime.now().isoformat(),
                    'model_dir': str(model_dir),
                    'config': self.model_configs[model_name],
                    'verified': True,
                    'source': 'huggingface_cache'
                }
                
                # Calculate actual size
                actual_size = self._calculate_directory_size(model_dir)
                self.model_registry['models'][model_name]['actual_size_mb'] = actual_size
                
                self.model_registry['last_updated'] = datetime.now().isoformat()
                self._save_model_registry()
                
                return True
            
            # If cache copy failed, download from internet
            self.logger.info(f"Downloading model from internet: {model_name}")
            model_dir.mkdir(exist_ok=True)
            
            config = self.model_configs[model_name]
            
            if config['type'] == 'transformer':
                # Download transformer model
                self._download_transformer_model(model_name, model_dir)
            elif config['type'] == 'sentence_transformer':
                # Download sentence transformer model
                self._download_sentence_transformer_model(model_name, model_dir)
            
            # Update registry
            self.model_registry['models'][model_name] = {
                'downloaded_at': datetime.now().isoformat(),
                'model_dir': str(model_dir),
                'config': config,
                'verified': True,
                'source': 'internet_download'
            }
            
            # Calculate actual size
            actual_size = self._calculate_directory_size(model_dir)
            self.model_registry['models'][model_name]['actual_size_mb'] = actual_size
            
            self.model_registry['last_updated'] = datetime.now().isoformat()
            self._save_model_registry()
            
            self.logger.info(f"SUCCESS: Successfully downloaded {model_name} ({actual_size:.1f} MB)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading {model_name}: {e}")
            if model_dir.exists():
                shutil.rmtree(model_dir)
            return False
    
    def _download_transformer_model(self, model_name: str, model_dir: Path):
        """Download transformer model (BERT, XLM-RoBERTa)."""
        try:
            # Download tokenizer
            self.logger.info(f"  Downloading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(model_dir)
            
            # Download model
            self.logger.info(f"  Downloading model weights for {model_name}...")
            model = AutoModel.from_pretrained(model_name)
            model.save_pretrained(model_dir)
            
            # Save model info
            model_info = {
                'model_name': model_name,
                'model_type': 'transformer',
                'tokenizer_class': 'AutoTokenizer',
                'model_class': 'AutoModel',
                'downloaded_at': datetime.now().isoformat()
            }
            
            with open(model_dir / 'model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
                
        except Exception as e:
            raise Exception(f"Failed to download transformer model {model_name}: {e}")
    
    def _download_sentence_transformer_model(self, model_name: str, model_dir: Path):
        """Download sentence transformer model."""
        try:
            self.logger.info(f"  Downloading sentence transformer {model_name}...")
            model = SentenceTransformer(model_name)
            model.save(str(model_dir))
            
            # Save model info
            model_info = {
                'model_name': model_name,
                'model_type': 'sentence_transformer',
                'model_class': 'SentenceTransformer',
                'downloaded_at': datetime.now().isoformat()
            }
            
            with open(model_dir / 'model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
                
        except Exception as e:
            raise Exception(f"Failed to download sentence transformer {model_name}: {e}")
    
    def _verify_model_integrity(self, model_name: str) -> bool:
        """Verify that a downloaded model is complete and valid."""
        try:
            model_dir = self.models_dir / model_name
            
            # Check if model_info.json exists
            model_info_file = model_dir / 'model_info.json'
            if not model_info_file.exists():
                return False
            
            # Load model info
            with open(model_info_file, 'r') as f:
                model_info = json.load(f)
            
            model_type = model_info.get('model_type')
            
            if model_type == 'transformer':
                # Check for essential transformer files
                essential_files = ['config.json']
                model_files = ['pytorch_model.bin', 'model.safetensors', 'tf_model.h5']
                tokenizer_files = ['tokenizer.json', 'vocab.txt', 'tokenizer_config.json']
                
                # Must have config.json
                for file_name in essential_files:
                    if not (model_dir / file_name).exists():
                        self.logger.warning(f"Missing essential file: {file_name}")
                        return False
                
                # Must have at least one model file
                has_model_file = any((model_dir / f).exists() for f in model_files)
                if not has_model_file:
                    self.logger.warning(f"Missing model file. Expected one of: {model_files}")
                    return False
                
                # Should have tokenizer files (but not strictly required for some models)
                has_tokenizer = any((model_dir / f).exists() for f in tokenizer_files)
                if not has_tokenizer:
                    self.logger.info(f"No tokenizer files found, but continuing...")
                    # Don't return False - some models might not need tokenizer files
            
            elif model_type == 'sentence_transformer':
                # Check for sentence transformer files
                essential_files = ['config.json']
                model_files = ['pytorch_model.bin', 'model.safetensors', 'tf_model.h5']
                
                # Must have config.json
                for file_name in essential_files:
                    if not (model_dir / file_name).exists():
                        self.logger.warning(f"Missing essential file: {file_name}")
                        return False
                
                # Must have at least one model file
                has_model_file = any((model_dir / f).exists() for f in model_files)
                if not has_model_file:
                    self.logger.warning(f"Missing model file. Expected one of: {model_files}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying model {model_name}: {e}")
            return False
    
    def _copy_from_hf_cache(self, model_name: str, target_dir: Path) -> bool:
        """Copy model from Hugging Face cache if available."""
        try:
            # Common Hugging Face cache locations
            hf_cache_locations = [
                Path.home() / ".cache" / "huggingface" / "transformers",
                Path.home() / ".cache" / "huggingface" / "hub",
                Path.home() / ".cache" / "torch" / "sentence_transformers"
            ]
            
            self.logger.info(f"Searching for {model_name} in Hugging Face cache...")
            
            # Search for model in cache locations
            for cache_dir in hf_cache_locations:
                if not cache_dir.exists():
                    continue
                
                # Look for model directories
                model_cache_dirs = self._find_model_in_cache(model_name, cache_dir)
                
                if model_cache_dirs:
                    self.logger.info(f"Found {model_name} in cache: {cache_dir}")
                    
                    # Copy the best match
                    source_dir = model_cache_dirs[0]
                    
                    # Create target directory
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy all files
                    self._copy_model_files(source_dir, target_dir, model_name)
                    
                    # Verify the copied model
                    if self._verify_copied_model(model_name, target_dir):
                        self.logger.info(f"SUCCESS: Successfully copied {model_name} from cache")
                        return True
                    else:
                        self.logger.warning(f"Copied model {model_name} failed verification")
                        if target_dir.exists():
                            shutil.rmtree(target_dir)
                        return False
            
            self.logger.info(f"Model {model_name} not found in Hugging Face cache")
            return False
            
        except Exception as e:
            self.logger.error(f"Error copying from HF cache: {e}")
            return False
    
    def _find_model_in_cache(self, model_name: str, cache_dir: Path) -> List[Path]:
        """Find model directories in cache."""
        model_dirs = []
        
        try:
            self.logger.debug(f"Searching for {model_name} in {cache_dir}")
            
            # Different naming patterns in HF cache
            search_patterns = [
                f"*{model_name}*",
                f"*{model_name.replace('-', '_')}*",
                f"*{model_name.replace('_', '-')}*",
                f"*{model_name.replace('-', '--')}*",  # HF hub uses -- sometimes
            ]
            
            # For sentence transformers, also try the model name parts
            if model_name == 'all-MiniLM-L6-v2':
                search_patterns.extend([
                    "*MiniLM*",
                    "*all-MiniLM*",
                    "*sentence-transformers*all-MiniLM*"
                ])
            
            # Search in direct subdirectories first
            for pattern in search_patterns:
                for path in cache_dir.glob(pattern):
                    if path.is_dir():
                        # Check if it contains model files
                        if self._contains_model_files(path):
                            model_dirs.append(path)
                            self.logger.debug(f"Found model files in: {path}")
            
            # Search deeper in HF hub structure (models--org--name format)
            for subdir in cache_dir.iterdir():
                if subdir.is_dir():
                    # Check if this is a model directory itself
                    if self._contains_model_files(subdir):
                        # Check if the directory name matches our model
                        dir_name = subdir.name.lower()
                        model_name_lower = model_name.lower()
                        
                        if (model_name_lower in dir_name or 
                            model_name_lower.replace('-', '_') in dir_name or
                            model_name_lower.replace('-', '--') in dir_name):
                            model_dirs.append(subdir)
                            self.logger.debug(f"Found model in subdir: {subdir}")
                    
                    # Also search within subdirectories
                    for pattern in search_patterns:
                        try:
                            for path in subdir.glob(pattern):
                                if path.is_dir() and self._contains_model_files(path):
                                    model_dirs.append(path)
                                    self.logger.debug(f"Found model in nested dir: {path}")
                        except Exception:
                            continue
                    
                    # Search in snapshots directory (HF hub structure)
                    snapshots_dir = subdir / "snapshots"
                    if snapshots_dir.exists():
                        for snapshot_dir in snapshots_dir.iterdir():
                            if snapshot_dir.is_dir() and self._contains_model_files(snapshot_dir):
                                # Check if parent directory name matches model
                                parent_name = subdir.name.lower()
                                if (model_name.lower() in parent_name or
                                    model_name.lower().replace('-', '--') in parent_name):
                                    model_dirs.append(snapshot_dir)
                                    self.logger.debug(f"Found model in snapshot: {snapshot_dir}")
            
            self.logger.debug(f"Found {len(model_dirs)} potential directories for {model_name}")
            return model_dirs
            
        except Exception as e:
            self.logger.error(f"Error searching cache: {e}")
            return []
    
    def _contains_model_files(self, directory: Path) -> bool:
        """Check if directory contains model files."""
        try:
            files_in_dir = [f.name.lower() for f in directory.iterdir() if f.is_file()]
            
            # Must have config.json
            has_config = 'config.json' in files_in_dir
            
            # Check for model files (various formats)
            model_file_patterns = [
                'pytorch_model.bin',
                'model.safetensors', 
                'tf_model.h5',
                'model.onnx',
                'model.bin'
            ]
            has_model = any(pattern in files_in_dir for pattern in model_file_patterns)
            
            # For sentence transformers, also check for sentence_bert_config.json
            has_sentence_config = 'sentence_bert_config.json' in files_in_dir
            
            # Check for tokenizer files (not required for sentence transformers)
            tokenizer_files = [
                'tokenizer.json',
                'vocab.txt',
                'tokenizer_config.json',
                'special_tokens_map.json'
            ]
            has_tokenizer = any(f in files_in_dir for f in tokenizer_files)
            
            # Different requirements for different model types
            if has_sentence_config:
                # Sentence transformer - needs config and model
                return has_config and has_model
            else:
                # Regular transformer - needs config, model, and usually tokenizer
                return has_config and has_model
            
        except Exception as e:
            self.logger.debug(f"Error checking model files in {directory}: {e}")
            return False
    
    def _copy_model_files(self, source_dir: Path, target_dir: Path, model_name: str):
        """Copy model files from source to target."""
        try:
            self.logger.info(f"Copying model files from {source_dir} to {target_dir}")
            
            # Copy all files
            for item in source_dir.iterdir():
                if item.is_file():
                    target_file = target_dir / item.name
                    shutil.copy2(item, target_file)
                elif item.is_dir():
                    target_subdir = target_dir / item.name
                    shutil.copytree(item, target_subdir, dirs_exist_ok=True)
            
            # Create model_info.json
            config = self.model_configs[model_name]
            model_info = {
                'model_name': model_name,
                'model_type': config['type'],
                'copied_from_cache': True,
                'source_path': str(source_dir),
                'copied_at': datetime.now().isoformat()
            }
            
            if config['type'] == 'transformer':
                model_info.update({
                    'tokenizer_class': 'AutoTokenizer',
                    'model_class': 'AutoModel'
                })
            elif config['type'] == 'sentence_transformer':
                model_info.update({
                    'model_class': 'SentenceTransformer'
                })
            
            with open(target_dir / 'model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
                
        except Exception as e:
            raise Exception(f"Failed to copy model files: {e}")
    
    def _verify_copied_model(self, model_name: str, model_dir: Path) -> bool:
        """Verify that copied model is valid."""
        try:
            # Use the existing verification method
            return self._verify_model_integrity(model_name)
        except Exception as e:
            self.logger.error(f"Error verifying copied model: {e}")
            return False
    
    def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate total size of directory in MB."""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    def load_local_model(self, model_name: str):
        """Load model from local storage."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers not available")
            return None, None
        
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            self.logger.error(f"Model {model_name} not found locally. Download it first.")
            return None, None
        
        try:
            # Load model info
            model_info_file = model_dir / 'model_info.json'
            with open(model_info_file, 'r') as f:
                model_info = json.load(f)
            
            model_type = model_info.get('model_type')
            
            if model_type == 'transformer':
                # Load transformer model
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                model = AutoModel.from_pretrained(str(model_dir))
                
                self.logger.info(f"SUCCESS: Loaded transformer model: {model_name}")
                return model, tokenizer
            
            elif model_type == 'sentence_transformer':
                # Load sentence transformer
                model = SentenceTransformer(str(model_dir))
                
                self.logger.info(f"SUCCESS: Loaded sentence transformer: {model_name}")
                return model, None
            
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return None, None
    
    def get_embeddings(self, text: str, model_name: str):
        """Get text embeddings from specified model."""
        try:
            model, tokenizer = self.load_local_model(model_name)
            
            if model is None:
                self.logger.warning(f"Model {model_name} not available")
                return None
            
            if 'all-MiniLM-L6-v2' in model_name or model_name in ['sentence-transformers']:
                # Sentence transformer embeddings
                embeddings = model.encode([text])
                return embeddings[0]  # Return single embedding
            
            elif 'bert' in model_name:
                # BERT embeddings
                if tokenizer is None:
                    self.logger.error("BERT tokenizer not available")
                    return None
                
                # Tokenize and get embeddings
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use [CLS] token embedding or mean pooling
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                return embeddings
            
            else:
                self.logger.error(f"Embedding extraction not implemented for {model_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting embeddings from {model_name}: {e}")
            return None
    
    def load_model(self, model_name: str):
        """Load and return model for direct use."""
        model, tokenizer = self.load_local_model(model_name)
        return model
    
    def scan_and_import_from_cache(self) -> Dict[str, bool]:
        """Scan Hugging Face cache and import available models."""
        results = {}
        
        self.logger.info("Scanning Hugging Face cache for available models...")
        
        for model_name in self.model_configs.keys():
            try:
                model_dir = self.models_dir / model_name
                
                # Skip if already exists and is valid
                if model_dir.exists() and self._verify_model_integrity(model_name):
                    self.logger.info(f"Model {model_name} already exists locally")
                    results[model_name] = True
                    continue
                
                # Try to copy from cache
                if self._copy_from_hf_cache(model_name, model_dir):
                    results[model_name] = True
                    self.logger.info(f"SUCCESS: Imported {model_name} from cache")
                else:
                    results[model_name] = False
                    self.logger.info(f"FAILED: {model_name} not found in cache")
                    
            except Exception as e:
                self.logger.error(f"Error importing {model_name}: {e}")
                results[model_name] = False
        
        imported_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        self.logger.info(f"Cache import completed: {imported_count}/{total_count} models available")
        
        return results
    
    def download_all_required_models(self, try_cache_first: bool = True) -> bool:
        """Download all models required for the pipeline."""
        required_models = ['bert-base-uncased', 'xlm-roberta-base', 'all-MiniLM-L6-v2']
        
        self.logger.info("Setting up all required models for the pipeline...")
        
        # First, try to import from cache if requested
        if try_cache_first:
            self.logger.info("Step 1: Importing from Hugging Face cache...")
            cache_results = self.scan_and_import_from_cache()
            
            # Check which models still need downloading
            missing_models = [model for model in required_models if not cache_results.get(model, False)]
            
            if not missing_models:
                self.logger.info("SUCCESS: All required models imported from cache!")
                return True
            else:
                self.logger.info(f"Step 2: Downloading missing models: {missing_models}")
        else:
            missing_models = required_models
        
        # Download missing models
        success_count = len(required_models) - len(missing_models)  # Already imported
        
        for model_name in missing_models:
            if self.download_model(model_name):
                success_count += 1
            else:
                self.logger.error(f"Failed to download {model_name}")
        
        if success_count == len(required_models):
            self.logger.info("SUCCESS: All required models are now available")
            return True
        else:
            self.logger.warning(f"Available models: {success_count}/{len(required_models)}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about Hugging Face cache."""
        cache_info = {
            'cache_locations': [],
            'total_cache_size_mb': 0,
            'available_in_cache': {}
        }
        
        try:
            # Check common cache locations
            hf_cache_locations = [
                Path.home() / ".cache" / "huggingface" / "transformers",
                Path.home() / ".cache" / "huggingface" / "hub",
                Path.home() / ".cache" / "torch" / "sentence_transformers"
            ]
            
            for cache_dir in hf_cache_locations:
                if cache_dir.exists():
                    cache_size = self._calculate_directory_size(cache_dir)
                    cache_info['cache_locations'].append({
                        'path': str(cache_dir),
                        'size_mb': cache_size,
                        'exists': True
                    })
                    cache_info['total_cache_size_mb'] += cache_size
                    
                    # Check for our required models
                    for model_name in self.model_configs.keys():
                        if model_name not in cache_info['available_in_cache']:
                            model_dirs = self._find_model_in_cache(model_name, cache_dir)
                            cache_info['available_in_cache'][model_name] = len(model_dirs) > 0
                else:
                    cache_info['cache_locations'].append({
                        'path': str(cache_dir),
                        'size_mb': 0,
                        'exists': False
                    })
            
            return cache_info
            
        except Exception as e:
            self.logger.error(f"Error getting cache info: {e}")
            return cache_info
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {
            'available_models': {},
            'total_size_mb': 0,
            'models_directory': str(self.models_dir),
            'cache_info': self.get_cache_info()
        }
        
        for model_name, config in self.model_configs.items():
            model_dir = self.models_dir / model_name
            
            if model_dir.exists():
                is_valid = self._verify_model_integrity(model_name)
                actual_size = self._calculate_directory_size(model_dir)
                
                # Get source info from registry
                source = 'unknown'
                if model_name in self.model_registry.get('models', {}):
                    source = self.model_registry['models'][model_name].get('source', 'unknown')
                
                status['available_models'][model_name] = {
                    'status': 'available' if is_valid else 'corrupted',
                    'size_mb': actual_size,
                    'expected_size_mb': config['size_mb'],
                    'description': config['description'],
                    'use_case': config['use_case'],
                    'source': source
                }
                
                if is_valid:
                    status['total_size_mb'] += actual_size
            else:
                # Check if available in cache
                available_in_cache = status['cache_info']['available_in_cache'].get(model_name, False)
                
                status['available_models'][model_name] = {
                    'status': 'available_in_cache' if available_in_cache else 'not_downloaded',
                    'size_mb': 0,
                    'expected_size_mb': config['size_mb'],
                    'description': config['description'],
                    'use_case': config['use_case'],
                    'available_in_cache': available_in_cache
                }
        
        return status
    
    def create_portable_package(self, output_dir: str = "portable_models") -> bool:
        """Create a portable package of all models."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            self.logger.info(f"Creating portable model package at: {output_path}")
            
            # Copy models directory
            portable_models_dir = output_path / "local_models"
            if portable_models_dir.exists():
                shutil.rmtree(portable_models_dir)
            
            shutil.copytree(self.models_dir, portable_models_dir)
            
            # Create setup script
            setup_script = output_path / "setup_models.py"
            setup_content = '''"""
Setup script for portable models
Run this script in your new environment to set up the models
"""

import shutil
from pathlib import Path

def setup_models():
    """Setup models in new environment."""
    current_dir = Path(__file__).parent
    source_dir = current_dir / "local_models"
    target_dir = Path("local_models")
    
    if source_dir.exists():
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        print("SUCCESS: Models set up successfully!")
        print(f"Models available at: {target_dir.absolute()}")
    else:
        print("ERROR: Source models directory not found")

if __name__ == "__main__":
    setup_models()
'''
            
            with open(setup_script, 'w', encoding='utf-8') as f:
                f.write(setup_content)
            
            # Create README
            readme_file = output_path / "README.md"
            readme_content = f'''# Portable Transformer Models

This package contains pre-downloaded transformer models for the Kenyan Twitter Misinformation Detection System.

## Contents
- local_models/: All transformer models
- setup_models.py: Setup script for new environments

## Setup in New Environment

1. Copy this entire folder to your new environment
2. Run the setup script:
   ```bash
   python setup_models.py
   ```

## Models Included
'''
            
            status = self.get_model_status()
            for model_name, info in status['available_models'].items():
                if info['status'] == 'available':
                    readme_content += f"- {model_name}: {info['description']} ({info['size_mb']:.1f} MB)\n"
            
            readme_content += f"\nTotal Size: {status['total_size_mb']:.1f} MB\n"
            readme_content += f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
            total_size_mb = total_size / (1024 * 1024)
            
            self.logger.info(f"SUCCESS: Portable package created successfully!")
            self.logger.info(f"   Location: {output_path.absolute()}")
            self.logger.info(f"   Size: {total_size_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating portable package: {e}")
            return False
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from local storage."""
        try:
            model_dir = self.models_dir / model_name
            
            if model_dir.exists():
                shutil.rmtree(model_dir)
                
                # Update registry
                if model_name in self.model_registry['models']:
                    del self.model_registry['models'][model_name]
                    self._save_model_registry()
                
                self.logger.info(f"SUCCESS: Removed model: {model_name}")
                return True
            else:
                self.logger.warning(f"Model {model_name} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing model {model_name}: {e}")
            return False

# Global model manager instance
_model_manager = None

def get_model_manager() -> LocalModelManager:
    """Get global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = LocalModelManager()
    return _model_manager