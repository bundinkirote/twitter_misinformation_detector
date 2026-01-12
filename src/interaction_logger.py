"""
Interaction Logger Module

This module provides comprehensive interaction logging capabilities for prediction
system audit and reproducibility. It implements detailed logging of user interactions,
prediction results, and system behavior for compliance, analysis, and system
improvement purposes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import uuid
from src.utils.file_manager import FileManager


class InteractionLogger:
    """
    Interaction Logging Class
    
    Implements comprehensive logging of user interactions with the prediction system
    for audit, reproducibility, and compliance purposes. Provides detailed tracking
    of prediction requests, results, and system behavior with structured logging
    and data persistence capabilities.
    """
    
    def __init__(self, log_dir: str = "logs/interactions"):
        self.logger = logging.getLogger(__name__)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.file_manager = FileManager()
        
        # Initialize daily log file
        self.current_log_file = self._get_daily_log_file()
        
    def _get_daily_log_file(self) -> Path:
        """Get today's log file path."""
        today = datetime.now().strftime('%Y%m%d')
        return self.log_dir / f"interactions_{today}.jsonl"
    
    def _generate_session_id(self, user_ip: str = None) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().isoformat()
        unique_data = f"{timestamp}_{user_ip or 'unknown'}_{uuid.uuid4()}"
        return hashlib.md5(unique_data.encode()).hexdigest()[:12]
    
    def log_prediction_interaction(self, 
                                 input_text: str,
                                 model_name: str,
                                 dataset_name: str,
                                 prediction_result: Dict[str, Any],
                                 user_ip: str = None,
                                 session_id: str = None,
                                 transformer_used: str = None,
                                 feature_types: List[str] = None) -> str:
        """Log a prediction interaction."""
        
        interaction_id = str(uuid.uuid4())
        session_id = session_id or self._generate_session_id(user_ip)
        
        log_entry = {
            'interaction_id': interaction_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'interaction_type': 'prediction',
            'input_data': {
                'text': input_text,
                'text_length': len(input_text),
                'word_count': len(input_text.split())
            },
            'model_configuration': {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'transformer_used': transformer_used,
                'feature_types': feature_types or []
            },
            'prediction_result': {
                'prediction': prediction_result.get('prediction'),
                'prediction_label': prediction_result.get('prediction_label'),
                'confidence': prediction_result.get('confidence'),
                'probability': prediction_result.get('probability'),
                'has_explanation': 'explanation' in prediction_result
            },
            'system_info': {
                'user_ip': user_ip,
                'processing_time': prediction_result.get('processing_time'),
                'model_version': prediction_result.get('model_version')
            },
            'audit_info': {
                'reproducible': True,
                'input_hash': hashlib.md5(input_text.encode()).hexdigest()
            }
        }
        
        # Write to log file
        self._write_log_entry(log_entry)
        
        self.logger.info(f"Logged prediction interaction: {interaction_id}")
        return interaction_id
    
    def log_model_selection(self, 
                           available_models: List[str],
                           selected_model: str,
                           user_ip: str = None,
                           session_id: str = None) -> str:
        """Log model selection interaction."""
        
        interaction_id = str(uuid.uuid4())
        session_id = session_id or self._generate_session_id(user_ip)
        
        log_entry = {
            'interaction_id': interaction_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'interaction_type': 'model_selection',
            'selection_data': {
                'available_models': available_models,
                'selected_model': selected_model,
                'selection_index': available_models.index(selected_model) if selected_model in available_models else -1
            },
            'system_info': {
                'user_ip': user_ip
            }
        }
        
        self._write_log_entry(log_entry)
        return interaction_id
    
    def log_visualization_request(self,
                                 visualization_type: str,
                                 prediction_id: str = None,
                                 user_ip: str = None,
                                 session_id: str = None) -> str:
        """Log visualization request."""
        
        interaction_id = str(uuid.uuid4())
        
        log_entry = {
            'interaction_id': interaction_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'interaction_type': 'visualization_request',
            'visualization_data': {
                'type': visualization_type,
                'related_prediction': prediction_id
            },
            'system_info': {
                'user_ip': user_ip
            }
        }
        
        self._write_log_entry(log_entry)
        return interaction_id
    
    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Write log entry to file."""
        try:
            # Update log file if day changed
            current_log_file = self._get_daily_log_file()
            
            with open(current_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error writing log entry: {e}")
    
    def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get interaction history for a session."""
        try:
            history = []
            
            # Search through recent log files
            for i in range(7):  # Last 7 days
                date = datetime.now().replace(day=datetime.now().day - i)
                log_file = self.log_dir / f"interactions_{date.strftime('%Y%m%d')}.jsonl"
                
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                if entry.get('session_id') == session_id:
                                    history.append(entry)
                            except json.JSONDecodeError:
                                continue
                
                if len(history) >= limit:
                    break
            
            # Sort by timestamp and limit
            history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return history[:limit]
            
        except Exception as e:
            self.logger.error(f"Error retrieving session history: {e}")
            return []
    
    def get_daily_stats(self, date: str = None) -> Dict[str, Any]:
        """Get daily interaction statistics."""
        try:
            if not date:
                date = datetime.now().strftime('%Y%m%d')
            
            log_file = self.log_dir / f"interactions_{date}.jsonl"
            
            if not log_file.exists():
                return {'date': date, 'total_interactions': 0}
            
            stats = {
                'date': date,
                'total_interactions': 0,
                'predictions': 0,
                'model_selections': 0,
                'visualizations': 0,
                'unique_sessions': set(),
                'models_used': {},
                'prediction_labels': {'Misinformation': 0, 'Legitimate': 0}
            }
            
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        stats['total_interactions'] += 1
                        stats['unique_sessions'].add(entry.get('session_id', 'unknown'))
                        
                        interaction_type = entry.get('interaction_type', 'unknown')
                        if interaction_type == 'prediction':
                            stats['predictions'] += 1
                            
                            # Model usage
                            model = entry.get('model_configuration', {}).get('model_name', 'unknown')
                            stats['models_used'][model] = stats['models_used'].get(model, 0) + 1
                            
                            # Prediction labels
                            label = entry.get('prediction_result', {}).get('prediction_label', 'unknown')
                            if label in stats['prediction_labels']:
                                stats['prediction_labels'][label] += 1
                                
                        elif interaction_type == 'model_selection':
                            stats['model_selections'] += 1
                        elif interaction_type == 'visualization_request':
                            stats['visualizations'] += 1
                            
                    except json.JSONDecodeError:
                        continue
            
            stats['unique_sessions'] = len(stats['unique_sessions'])
            return stats
            
        except Exception as e:
            self.logger.error(f"Error generating daily stats: {e}")
            return {'date': date, 'total_interactions': 0, 'error': str(e)}
    
    def export_logs(self, date_range: tuple = None, format: str = 'json') -> str:
        """Export logs for a date range."""
        try:
            if not date_range:
                # Export today's logs
                start_date = end_date = datetime.now().strftime('%Y%m%d')
            else:
                start_date, end_date = date_range
            
            export_data = []
            current_date = datetime.strptime(start_date, '%Y%m%d')
            end_date_obj = datetime.strptime(end_date, '%Y%m%d')
            
            while current_date <= end_date_obj:
                date_str = current_date.strftime('%Y%m%d')
                log_file = self.log_dir / f"interactions_{date_str}.jsonl"
                
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                export_data.append(entry)
                            except json.JSONDecodeError:
                                continue
                
                current_date = current_date.replace(day=current_date.day + 1)
            
            # Save export
            export_file = self.log_dir / f"export_{start_date}_to_{end_date}.json"
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            return str(export_file)
            
        except Exception as e:
            self.logger.error(f"Error exporting logs: {e}")
            raise


# Global logger instance
_interaction_logger = None

def get_interaction_logger() -> InteractionLogger:
    """Get global interaction logger instance."""
    global _interaction_logger
    if _interaction_logger is None:
        _interaction_logger = InteractionLogger()
    return _interaction_logger