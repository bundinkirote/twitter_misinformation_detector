"""
File Manager Module

This module provides comprehensive file management capabilities for dataset
organization, file operations, and data persistence. It implements robust
file handling, JSON serialization, and directory management for machine
learning pipeline data organization and storage.
"""

import os
import json
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
import logging
import numpy as np

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder Class
    
    Implements specialized JSON encoding for handling numpy types, datetime objects,
    and other non-serializable data types commonly encountered in machine learning
    workflows. Provides robust serialization for complex data structures.
    """
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        return super().default(obj)

class FileManager:
    """
    File Management Class
    
    Implements comprehensive file management operations including dataset organization,
    file persistence, directory management, and data serialization. Provides robust
    file handling capabilities for machine learning pipeline data management and
    storage operations with error handling and logging.
    """
    
    def __init__(self):
        self.base_dir = Path('.')
        self.datasets_dir = self.base_dir / 'datasets'
        self.logs_dir = self.base_dir / 'logs'
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def sanitize_dataset_name(dataset_name):
        """Sanitize dataset name to be filesystem-safe."""
        # Replace invalid characters with underscores
        # Invalid characters for Windows: < > : " | ? * and control characters
        sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', '_', dataset_name)
        
        # Replace multiple underscores with single underscore
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores and dots
        sanitized = sanitized.strip('_.')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'dataset'
        
        return sanitized
    
    def create_directories(self):
        """Create necessary directory structure."""
        directories = [
            'datasets',
            'logs',
            'static/images',
            'static/plots',
            'templates'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_dataset_directory(self, dataset_name):
        """Create directory structure for a specific dataset."""
        dataset_name = self.sanitize_dataset_name(dataset_name)
        dataset_path = self.datasets_dir / dataset_name
        
        subdirs = [
            'raw',
            'processed', 
            'features',
            'models',
            'results',
            'visualizations',
            'network_analysis'
        ]
        
        for subdir in subdirs:
            os.makedirs(dataset_path / subdir, exist_ok=True)
        
        # Create dataset info file
        info_file = dataset_path / 'dataset_info.json'
        if not info_file.exists():
            info = {
                'name': dataset_name,
                'created': datetime.now().isoformat(),
                'status': 'created'
            }
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)
        
        return str(dataset_path)
    
    def get_dataset_info(self, dataset_name):
        """Get dataset information."""
        try:
            dataset_name = self.sanitize_dataset_name(dataset_name)
            info_file = self.datasets_dir / dataset_name / 'dataset_info.json'
            
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                    # Ensure all values are JSON serializable
                    return self._make_json_serializable(info)
            
            return {'name': dataset_name, 'status': 'unknown'}
            
        except Exception as e:
            logging.error(f"Error getting dataset info for {dataset_name}: {e}")
            return {'name': dataset_name, 'status': 'error', 'error': str(e)}
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj
    
    def update_dataset_info(self, dataset_name, updates):
        """Update dataset information."""
        try:
            dataset_name = self.sanitize_dataset_name(dataset_name)
            dataset_dir = self.datasets_dir / dataset_name
            info_file = dataset_dir / 'dataset_info.json'
            
            # Ensure dataset directory exists
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
            else:
                info = {'name': dataset_name}
            
            # Convert any non-serializable values in updates
            serializable_updates = {}
            for key, value in updates.items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    serializable_updates[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    serializable_updates[key] = float(value)
                elif isinstance(value, (np.bool_, bool)):
                    serializable_updates[key] = bool(value)
                elif isinstance(value, tuple):
                    serializable_updates[key] = list(value)
                elif isinstance(value, np.ndarray):
                    serializable_updates[key] = value.tolist()
                else:
                    serializable_updates[key] = value
            
            info.update(serializable_updates)
            info['last_updated'] = datetime.now().isoformat()
            
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2, cls=CustomJSONEncoder)
                
        except Exception as e:
            logging.error(f"Error updating dataset info for {dataset_name}: {e}")
            # Create a minimal info file if update fails
            try:
                info_file = self.datasets_dir / dataset_name / 'dataset_info.json'
                minimal_info = {
                    'name': dataset_name,
                    'status': 'error',
                    'error': str(e),
                    'last_updated': datetime.now().isoformat()
                }
                with open(info_file, 'w') as f:
                    json.dump(minimal_info, f, indent=2)
            except:
                pass  # If even this fails, just continue
    
    def get_all_datasets(self):
        """Get list of all available datasets."""
        datasets = []
        
        if self.datasets_dir.exists():
            for dataset_dir in self.datasets_dir.iterdir():
                if dataset_dir.is_dir():
                    info = self.get_dataset_info(dataset_dir.name)
                    datasets.append(info)
        
        return datasets
    
    def list_datasets(self):
        """List all available datasets with their metadata."""
        datasets = []
        
        if not self.datasets_dir.exists():
            return datasets
        
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_info = self.get_dataset_info(dataset_dir.name)
                if dataset_info:
                    datasets.append({
                        'name': dataset_dir.name,
                        'samples': dataset_info.get('total_samples', 0),
                        'upload_time': dataset_info.get('upload_time', ''),
                        'has_labels': 'LABEL' in dataset_info.get('columns', []),
                        'status': dataset_info.get('status', 'uploaded')
                    })
        
        return datasets
    
    def load_processed_data(self, dataset_name):
        """Load processed data for a dataset."""
        try:
            dataset_name = self.sanitize_dataset_name(dataset_name)
            processed_file = self.datasets_dir / dataset_name / 'processed' / 'processed_data.csv'
            
            if not processed_file.exists():
                raise FileNotFoundError(f"Processed data file not found for dataset: {dataset_name}")
            
            df = pd.read_csv(processed_file)
            self.logger.info(f"Loaded processed data for {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading processed data for {dataset_name}: {e}")
            raise

    def get_dataset_stats(self, dataset_name):
        """Get basic statistics for a dataset."""
        try:
            processed_file = self.datasets_dir / dataset_name / 'processed' / 'processed_data.csv'
            
            if processed_file.exists():
                df = pd.read_csv(processed_file)
                
                stats = {
                    'total_samples': len(df),
                    'features': df.shape[1],
                    'columns': list(df.columns),
                    'missing_values': df.isnull().sum().sum(),
                    'data_types': df.dtypes.to_dict()
                }
                
                # Add label statistics if available
                if 'LABEL' in df.columns:
                    stats['misinformation_count'] = int(df['LABEL'].sum())
                    stats['misinformation_rate'] = float(df['LABEL'].mean() * 100)
                    stats['class_distribution'] = df['LABEL'].value_counts().to_dict()
                
                return stats
            
            return {'error': 'Processed data not found'}
        
        except Exception as e:
            self.logger.error(f"Error getting dataset stats: {e}")
            return {'error': str(e)}
    
    def save_results(self, dataset_name, results, result_type):
        """Save results to appropriate directory."""
        dataset_name = self.sanitize_dataset_name(dataset_name)
        results_dir = self.datasets_dir / dataset_name / 'results'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{result_type}_{timestamp}.json"
        
        filepath = results_dir / filename
        
        # Ensure directory exists
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return str(filepath)
    
    def load_results(self, dataset_name, result_type):
        """Load the latest results of a specific type."""
        results_dir = self.datasets_dir / dataset_name / 'results'
        
        if not results_dir.exists():
            return None
        
        # First try exact filename match
        exact_file = results_dir / f"{result_type}.json"
        if exact_file.exists():
            with open(exact_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Then try pattern match for timestamped files
        pattern = f"{result_type}_*.json"
        files = list(results_dir.glob(pattern))
        
        if not files:
            return None
        
        # Get the most recent file
        latest_file = max(files, key=os.path.getctime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_dataset_path(self, dataset_name):
        """Get the path to a dataset directory."""
        dataset_name = self.sanitize_dataset_name(dataset_name)
        return self.datasets_dir / dataset_name
    
    def get_results_path(self, dataset_name, result_type):
        """Get the path to the latest results file of a specific type."""
        results_dir = self.datasets_dir / dataset_name / 'results'
        
        if not results_dir.exists():
            return None
        
        # Find the latest file of the specified type
        pattern = f"{result_type}_*.json"
        files = list(results_dir.glob(pattern))
        
        if not files:
            return None
        
        # Get the most recent file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        return latest_file
    
    def get_model_path(self, dataset_name, model_name):
        """Get path for a specific model."""
        models_dir = self.datasets_dir / dataset_name / 'models'
        
        # Handle special case for best_model
        if model_name == 'best_model':
            # Try different patterns for best model
            possible_paths = [
                models_dir / 'best_model_complete.joblib',
                models_dir / 'best_model.joblib',
                models_dir / 'logistic_regression.joblib',  # fallback
                models_dir / 'naive_bayes.joblib'  # fallback
            ]
            
            for path in possible_paths:
                if path.exists():
                    return str(path)
        
        return str(models_dir / f"{model_name}.joblib")
    
    def get_visualization_path(self, dataset_name, viz_name):
        """Get path for a specific visualization."""
        viz_dir = self.datasets_dir / dataset_name / 'visualizations'
        return str(viz_dir / f"{viz_name}.png")
    
    def cleanup_old_files(self, dataset_name, days_old=30):
        """Clean up old files in dataset directory."""
        dataset_dir = self.datasets_dir / dataset_name
        
        if not dataset_dir.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                filepath = Path(root) / file
                
                # Skip important files
                if file in ['dataset_info.json', 'processed_data.csv']:
                    continue
                
                if filepath.stat().st_mtime < cutoff_time:
                    try:
                        filepath.unlink()
                        self.logger.info(f"Deleted old file: {filepath}")
                    except Exception as e:
                        self.logger.error(f"Error deleting file {filepath}: {e}")