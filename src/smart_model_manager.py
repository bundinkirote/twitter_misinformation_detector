#!/usr/bin/env python3
"""
Smart Model Manager Module

This module provides intelligent model management capabilities including automatic
model downloading, caching, local storage management, and version control for
transformer models. It implements efficient model lifecycle management with
fallback strategies and resource optimization.
"""

import os
import shutil
import sys
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union

# Configure logging with safe encoding
def setup_safe_logging():
    """Setup logging that won't fail on Unicode characters."""
    try:
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    except:
        # Fallback to basic logging if there are encoding issues
        logging.basicConfig(level=logging.INFO)

setup_safe_logging()

class SmartModelManager:
    """
    Smart Model Management Class
    
    Implements intelligent transformer model management with automatic caching,
    local storage optimization, and efficient model lifecycle handling. Provides
    systematic model downloading, version control, and resource management for
    optimal performance and storage efficiency.
    """
    
    def __init__(self, local_models_dir: str = "local_models"):
        self.logger = logging.getLogger(__name__)
        self.local_models_dir = Path(local_models_dir)
        self.local_models_dir.mkdir(exist_ok=True)
        
        # Find Hugging Face cache directory
        self.cache_dir = self._find_hf_cache()
        
        self.logger.info(f"Smart Model Manager initialized")
        self.logger.info(f"Local models: {self.local_models_dir}")
        self.logger.info(f"HF Cache: {self.cache_dir}")
    
    def _find_hf_cache(self) -> Optional[Path]:
        """Find the Hugging Face cache directory."""
        possible_locations = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "huggingface" / "transformers",
            Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
            Path("C:/Users") / os.environ.get("USERNAME", "") / ".cache" / "huggingface" / "hub",
        ]
        
        for location in [loc for loc in possible_locations if loc is not None]:
            if location.exists():
                return location
        
        return None
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Convert model name to local directory name (replace / with --)."""
        return model_name.replace("/", "--")
    
    def _get_local_model_path(self, model_name: str) -> Path:
        """Get the local path for a model."""
        normalized_name = self._normalize_model_name(model_name)
        return self.local_models_dir / normalized_name
    
    def _find_model_in_cache(self, model_name: str) -> Optional[Path]:
        """Find a model in the Hugging Face cache."""
        if not self.cache_dir:
            return None
        
        normalized_name = self._normalize_model_name(model_name)
        cache_model_dir = self.cache_dir / f"models--{normalized_name}"
        
        if not cache_model_dir.exists():
            return None
        
        # Look for snapshots directory
        snapshots_dir = cache_model_dir / "snapshots"
        if not snapshots_dir.exists():
            return None
        
        # Find the latest snapshot
        snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshots:
            return None
        
        latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
        
        # Verify it has required files
        required_files = ["config.json"]
        model_weight_files = ["pytorch_model.bin", "model.safetensors"]
        
        # Check basic files
        if not all((latest_snapshot / f).exists() for f in required_files):
            return None
        
        # For zero-shot models, also check for model weights
        if any((latest_snapshot / f).exists() for f in model_weight_files):
            return latest_snapshot
        else:
            # Cache exists but is incomplete (only tokenizer files)
            self.logger.warning(f"Cache for {model_name} is incomplete (missing model weights)")
            return None
    
    def _copy_model_from_cache(self, cache_path: Path, local_path: Path, model_name: str) -> bool:
        """Copy model from cache to local models directory."""
        try:
            self.logger.info(f"Copying {model_name} from cache to local storage...")
            
            # Create local directory
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Copy all files
            total_size = 0
            copied_files = 0
            
            for file_path in cache_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(cache_path)
                    target_file = local_path / relative_path
                    
                    # Create parent directories if needed
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(file_path, target_file)
                    
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    copied_files += 1
            
            # Create model info file
            model_info = {
                "model_name": model_name,
                "local_path": str(local_path),
                "copied_from_cache": str(cache_path),
                "copied_date": datetime.now().isoformat(),
                "size_mb": total_size / (1024*1024),
                "files_count": copied_files
            }
            
            with open(local_path / "model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)
            
            self.logger.info(f"Copied {model_name}: {copied_files} files, {total_size/(1024*1024):.1f} MB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to copy {model_name} from cache: {e}")
            return False
    
    def _download_model_to_cache(self, model_name: str, model_type: str = "auto") -> bool:
        """Download model to Hugging Face cache."""
        try:
            self.logger.info(f"Downloading {model_name} to cache (first time)...")
            
            if model_type == "sentence_transformer":
                from sentence_transformers import SentenceTransformer
                # This will download to cache
                model = SentenceTransformer(model_name)
                del model  # Free memory
                
            elif model_type == "zero_shot":
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                # Download both tokenizer and model to ensure complete cache
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                del tokenizer, model  # Free memory
                
            else:  # auto/default
                from transformers import AutoTokenizer, AutoModel
                # This will download to cache
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                del tokenizer, model  # Free memory
            
            self.logger.info(f"Downloaded {model_name} to cache")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {model_name}: {e}")
            return False
    
    def get_model_path(self, model_name: str, model_type: str = "auto") -> str:
        """
        Get the path to use for loading a model.
        
        Process:
        1. Check if model exists locally -> use local
        2. Check if model exists in cache -> copy to local, use local
        3. Download to cache -> copy to local, use local
        
        Args:
            model_name: Name of the model (e.g., 'bert-base-uncased', 'facebook/bart-large-mnli')
            model_type: Type of model ('auto', 'sentence_transformer', 'zero_shot')
            
        Returns:
            Path to use for loading the model
        """
        local_path = self._get_local_model_path(model_name)
        
        # Step 1: Check if model exists locally
        if local_path.exists() and (local_path / "config.json").exists():
            self.logger.info(f"Using local model: {model_name}")
            return str(local_path)
        
        # Step 2: Check if model exists in cache
        cache_path = self._find_model_in_cache(model_name)
        if cache_path:
            self.logger.info(f"Found {model_name} in cache, copying to local...")
            if self._copy_model_from_cache(cache_path, local_path, model_name):
                return str(local_path)
            else:
                self.logger.warning(f"Failed to copy {model_name}, using cache directly")
                return model_name
        
        # Step 3: Download to cache, then copy to local
        self.logger.info(f"Model {model_name} not found locally or in cache, downloading...")
        if self._download_model_to_cache(model_name, model_type):
            # Now try to find it in cache and copy
            cache_path = self._find_model_in_cache(model_name)
            if cache_path and self._copy_model_from_cache(cache_path, local_path, model_name):
                return str(local_path)
            else:
                self.logger.warning(f"Downloaded {model_name} but couldn't copy to local, using remote")
                return model_name
        else:
            self.logger.error(f"Failed to download {model_name}, using remote as fallback")
            return model_name
    
    def load_sentence_transformer(self, model_name: str):
        """Load a sentence transformer model."""
        from sentence_transformers import SentenceTransformer
        model_path = self.get_model_path(model_name, "sentence_transformer")
        return SentenceTransformer(model_path)
    
    def load_transformers_model(self, model_name: str):
        """Load a transformers model and tokenizer."""
        from transformers import AutoTokenizer, AutoModel
        model_path = self.get_model_path(model_name, "auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        return tokenizer, model
    
    def load_zero_shot_classifier(self, model_name: str):
        """Load a zero-shot classification pipeline."""
        from transformers import pipeline
        model_path = self.get_model_path(model_name, "zero_shot")
        return pipeline("zero-shot-classification", model=model_path)
    
    def get_local_models_info(self) -> Dict[str, Any]:
        """Get information about locally stored models."""
        models_info = {}
        total_size = 0
        
        if not self.local_models_dir.exists():
            return {"models": {}, "total_size_mb": 0, "count": 0}
        
        for model_dir in self.local_models_dir.iterdir():
            if model_dir.is_dir():
                # Calculate size
                size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                total_size += size_mb
                
                # Load model info if available
                info_file = model_dir / "model_info.json"
                model_info = {"size_mb": size_mb}
                
                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            saved_info = json.load(f)
                            model_info.update(saved_info)
                    except:
                        pass
                
                models_info[model_dir.name] = model_info
        
        return {
            "models": models_info,
            "total_size_mb": total_size,
            "count": len(models_info)
        }
    
    def cleanup_cache_after_copy(self, model_name: str) -> bool:
        """Optionally clean up cache after copying to local (to save space)."""
        if not self.cache_dir:
            return False
        
        try:
            normalized_name = self._normalize_model_name(model_name)
            cache_model_dir = self.cache_dir / f"models--{normalized_name}"
            
            if cache_model_dir.exists():
                shutil.rmtree(cache_model_dir)
                self.logger.info(f"Cleaned up cache for {model_name}")
                return True
        except Exception as e:
            self.logger.warning(f"Failed to cleanup cache for {model_name}: {e}")
        
        return False


# Global instance
_smart_model_manager = None

def get_smart_model_manager() -> SmartModelManager:
    """Get the global SmartModelManager instance."""
    global _smart_model_manager
    if _smart_model_manager is None:
        _smart_model_manager = SmartModelManager()
    return _smart_model_manager


# Convenience functions
def get_model_path(model_name: str, model_type: str = "auto") -> str:
    """Get the path for a model (downloads if needed)."""
    return get_smart_model_manager().get_model_path(model_name, model_type)

def load_sentence_transformer(model_name: str):
    """Load a sentence transformer model."""
    return get_smart_model_manager().load_sentence_transformer(model_name)

def load_transformers_model(model_name: str):
    """Load a transformers model and tokenizer."""
    return get_smart_model_manager().load_transformers_model(model_name)

def load_zero_shot_classifier(model_name: str):
    """Load a zero-shot classification pipeline."""
    return get_smart_model_manager().load_zero_shot_classifier(model_name)

def get_local_models_info() -> Dict[str, Any]:
    """Get information about locally stored models."""
    return get_smart_model_manager().get_local_models_info()


if __name__ == "__main__":
    # Test the smart model manager
    print("🧪 TESTING SMART MODEL MANAGER")
    print("=" * 60)
    
    manager = SmartModelManager()
    
    # Test with a small model
    test_models = [
        ("distilbert-base-uncased", "auto"),
        ("all-MiniLM-L6-v2", "sentence_transformer"),
        ("facebook/bart-large-mnli", "zero_shot")
    ]
    
    for model_name, model_type in test_models:
        print(f"\n🔍 Testing {model_name} ({model_type})")
        path = manager.get_model_path(model_name, model_type)
        print(f"Result: {path}")
    
    # Show local models info
    print(f"\n📊 LOCAL MODELS SUMMARY:")
    info = manager.get_local_models_info()
    print(f"Total models: {info['count']}")
    print(f"Total size: {info['total_size_mb']:.1f} MB")
    
    for model_name, model_info in info['models'].items():
        print(f"  📁 {model_name}: {model_info['size_mb']:.1f} MB")