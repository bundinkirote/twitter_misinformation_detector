#!/usr/bin/env python3
"""
Prediction Service Module

This module provides comprehensive prediction capabilities using multiple analysis
methods and saved results. It integrates zero-shot classification, trained model
predictions, SHAP explanations, sentiment analysis, and behavioral analysis for
robust misinformation detection with detailed explanatory insights.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from datetime import datetime
import time

class PredictionService:
    """
    Comprehensive Prediction Service Class
    
    Implements comprehensive prediction capabilities combining multiple analysis
    methods including zero-shot classification, trained model predictions, SHAP
    explanations, sentiment analysis, behavioral analysis, and fact-checking.
    Provides integrated prediction results with detailed explanatory insights.
    """
    
    def __init__(self, model_evaluator=None, shap_explainer=None, file_manager=None):
        self.model_evaluator = model_evaluator
        self.shap_explainer = shap_explainer
        self.file_manager = file_manager
        self.logger = logging.getLogger(__name__)
        
    def create_comprehensive_prediction(self, text, dataset_name, model_name=None, 
                                      transformer_type='bert-base-uncased', 
                                      use_zero_shot=False, include_sentiment=True, 
                                      include_behavioral=True, include_shap=True, 
                                      include_fact_check=True):
        """
        Create a comprehensive prediction result using all available saved data for the dataset.
        This combines zero-shot results, trained model predictions, SHAP explanations, sentiment analysis, etc.
        """
        
        result = {
            'prediction': 'uncertain',
            'confidence': 0.0,
            'method': 'comprehensive_analysis',
            'dataset_name': dataset_name,
            'model_name': model_name or 'auto',
            'analyses': {}
        }
        
        dataset_dir = Path('datasets') / dataset_name
        
        # 1. Get zero-shot prediction from saved results
        zero_shot_result = self._get_zero_shot_prediction(text, dataset_dir)
        if zero_shot_result:
            result['analyses']['zero_shot'] = zero_shot_result
            
            # If using zero-shot mode, make this the primary prediction
            if use_zero_shot:
                result['prediction'] = zero_shot_result['prediction']
                result['confidence'] = zero_shot_result['confidence']
                result['method'] = 'zero_shot_similarity'
        
        # 2. Get trained model prediction if not using zero-shot
        if not use_zero_shot and model_name and self.model_evaluator:
            model_result = self._get_trained_model_prediction(text, dataset_name, model_name)
            if model_result:
                result['analyses']['trained_model'] = model_result
                
                # Extract prediction from nested structure
                if 'primary_prediction' in model_result:
                    primary = model_result['primary_prediction']
                    result['prediction'] = primary.get('prediction', 0)
                    result['confidence'] = primary.get('confidence', 0.0)
                    result['model_name'] = primary.get('model_name', model_name)
                    
                    # Map additional fields for frontend compatibility
                    if 'zero_shot_prediction' in model_result:
                        result['zero_shot_result'] = model_result['zero_shot_prediction']
                    if 'fact_check_validation' in model_result:
                        result['fact_check_scores'] = model_result['fact_check_validation']
                else:
                    # Fallback for flat structure
                    result['prediction'] = model_result.get('prediction', 0)
                    result['confidence'] = model_result.get('confidence', 0.0)
                    result['model_name'] = model_result.get('model_name', model_name)
                    
                result['method'] = f'trained_model_{model_name}'
                
                # Try to get SHAP explanation (if enabled)
                if include_shap:
                    shap_result = self._get_shap_explanation(text, dataset_name, model_name)
                    if shap_result:
                        result['shap_explanation'] = shap_result
                        result['analyses']['shap'] = shap_result
        
        # 3. Add sentiment analysis
        if include_sentiment:
            sentiment_result = self._get_sentiment_analysis(text)
            if sentiment_result:
                result['analyses']['sentiment'] = sentiment_result
                result['sentiment_analysis'] = sentiment_result  # Also add to main result for compatibility
        
        # 4. Add behavioral analysis
        if include_behavioral:
            behavioral_result = self._get_behavioral_analysis(text)
            if behavioral_result:
                result['analyses']['behavioral'] = behavioral_result
        
        # 5. Add fact-checking (if enabled)
        if include_fact_check:
            fact_check_result = self._get_fact_check_analysis(text)
            if fact_check_result:
                result['analyses']['fact_check'] = fact_check_result
        
        # 6. Combine all analyses for final confidence score
        if len(result['analyses']) > 1:
            result['confidence'] = self._calculate_weighted_confidence(result['analyses'])
            result['method'] = 'comprehensive_weighted'
        
        # Fallback if no prediction was made
        if result['confidence'] == 0.0 and zero_shot_result:
            result['prediction'] = zero_shot_result['prediction']
            result['confidence'] = zero_shot_result['confidence']
            result['method'] = 'zero_shot_fallback'
        elif result['confidence'] == 0.0:
            # Ultimate fallback using keyword analysis
            keyword_result = self._get_keyword_fallback_prediction(text)
            result.update(keyword_result)
        
        # Add prediction method for compatibility
        result['prediction_method'] = result['method']
        
        return result
    
    def _get_zero_shot_prediction(self, text, dataset_dir):
        """Get zero-shot prediction from saved results using similarity matching."""
        zero_shot_csv = dataset_dir / 'processed' / 'zero_shot_labeled.csv'
        
        if not zero_shot_csv.exists():
            return None
            
        try:
            df = pd.read_csv(zero_shot_csv)
            if not all(col in df.columns for col in ['CLEANED_TEXT', 'zero_shot_label', 'confidence_score']):
                return None
                
            # Find most similar text using TF-IDF
            texts = df['CLEANED_TEXT'].fillna('').astype(str).tolist()
            labels = df['zero_shot_label'].tolist()
            confidences = df['confidence_score'].tolist()
            
            if len(texts) == 0:
                return None
                
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            clean_input = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            
            all_texts = texts + [clean_input]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            input_vector = tfidf_matrix[-1]
            dataset_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(input_vector, dataset_vectors).flatten()
            
            max_idx = np.argmax(similarities)
            max_similarity = similarities[max_idx]
            
            if max_similarity > 0.1:  # Minimum similarity threshold
                prediction_label = labels[max_idx] if labels[max_idx] in ['misinformation', 'legitimate'] else 'uncertain'
                return {
                    'prediction': prediction_label,
                    'confidence': float(confidences[max_idx]),
                    'similarity': float(max_similarity),
                    'matched_text': texts[max_idx][:100] + '...' if len(texts[max_idx]) > 100 else texts[max_idx]
                }
                
        except Exception as e:
            self.logger.warning(f"Error loading zero-shot results: {e}")
            
        return None
    
    def _get_trained_model_prediction(self, text, dataset_name, model_name):
        """Get prediction from trained model."""
        if not self.model_evaluator:
            return None
            
        try:
            available_models = self.model_evaluator.get_available_models(dataset_name)
            if available_models and model_name in available_models:
                return self.model_evaluator.predict_with_model(dataset_name, text, model_name)
        except Exception as e:
            self.logger.warning(f"Trained model prediction failed: {e}")
            
        return None
    
    def _get_shap_explanation(self, text, dataset_name, model_name):
        """Get SHAP explanation for the prediction."""
        if not self.shap_explainer:
            self.logger.info("SHAP explainer not available, skipping SHAP explanation")
            return None
            
        try:
            # Add timeout and better error handling
            shap_result = self.shap_explainer.explain_prediction_for_text(dataset_name, model_name, text)
            if shap_result:
                self.logger.info("SHAP explanation generated successfully")
                return shap_result
            else:
                self.logger.warning("SHAP explanation returned None - model file may be missing")
                return None
        except Exception as e:
            self.logger.warning(f"SHAP explanation failed (non-critical): {e}")
            # Don't let SHAP failures break the entire prediction
            return None
    
    def _get_sentiment_analysis(self, text):
        """Get sentiment analysis results."""
        try:
            from .sentiment_analyzer import SentimentAnalyzer
            if self.file_manager:
                sentiment_analyzer = SentimentAnalyzer(self.file_manager)
                sentiment_result = sentiment_analyzer.analyze_sentiment(text)
                return {
                    'sentiment': sentiment_result.get('sentiment', 'neutral'),
                    'confidence': sentiment_result.get('confidence', 0.0),
                    'scores': sentiment_result.get('scores', {})
                }
            else:
                self.logger.warning("File manager not available for sentiment analysis")
                return None
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {e}")
            return None
    
    def _get_behavioral_analysis(self, text):
        """Get behavioral pattern analysis using lightweight text-only analysis."""
        try:
            # Use simple text-based behavioral indicators instead of complex framework analysis
            # This avoids the need for dataset columns during prediction
            
            text_lower = text.lower()
            
            # Urgency indicators
            urgency_keywords = ['breaking', 'urgent', 'immediately', 'now', 'quickly', 'asap', 'emergency']
            urgency_score = sum(1 for keyword in urgency_keywords if keyword in text_lower) / len(urgency_keywords)
            
            # Emotional manipulation indicators
            emotion_keywords = ['shocking', 'unbelievable', 'amazing', 'terrible', 'outrageous', 'incredible', 'devastating']
            emotional_manipulation = sum(1 for keyword in emotion_keywords if keyword in text_lower) / len(emotion_keywords)
            
            # Authority claims indicators
            authority_keywords = ['expert', 'official', 'confirmed', 'proven', 'study shows', 'research', 'according to']
            authority_claims = sum(1 for keyword in authority_keywords if keyword in text_lower) / len(authority_keywords)
            
            # Additional behavioral patterns
            exclamation_count = text.count('!')
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            
            return {
                'urgency_score': min(urgency_score + (exclamation_count * 0.1), 1.0),
                'emotional_manipulation': min(emotional_manipulation + (caps_ratio * 0.5), 1.0),
                'authority_claims': authority_claims,
                'text_length': len(text),
                'exclamation_count': exclamation_count,
                'caps_ratio': caps_ratio,
                'method': 'text_based_lightweight'
            }
            
        except Exception as e:
            self.logger.warning(f"Behavioral analysis failed: {e}")
            return None
    
    def _get_fact_check_analysis(self, text):
        """Get fact-checking analysis results with timeout and error handling."""
        try:
            from .fact_check_validator import FactCheckValidator
            if not self.file_manager:
                self.logger.warning("File manager not available for fact checking")
                return None
            
            # Add timeout handling for network operations
            import threading
            import time
            
            result_container = {'result': None, 'error': None}
            
            def fact_check_worker():
                try:
                    fact_checker = FactCheckValidator(self.file_manager)
                    result_container['result'] = fact_checker.validate_with_external_sources(text)
                except Exception as e:
                    result_container['error'] = str(e)
            
            # Start fact-checking in a separate thread with timeout
            thread = threading.Thread(target=fact_check_worker)
            thread.daemon = True
            thread.start()
            thread.join(timeout=10)  # 10 second timeout
            
            if thread.is_alive():
                self.logger.warning("Fact checking timed out (10s) - skipping")
                return None
            
            if result_container['error']:
                self.logger.warning(f"Fact checking failed (non-critical): {result_container['error']}")
                return None
            
            if result_container['result']:
                self.logger.info("Fact checking completed successfully")
                return result_container['result']
            else:
                self.logger.warning("Fact checking returned no results")
                return None
                
        except Exception as e:
            self.logger.warning(f"Fact checking setup failed (non-critical): {e}")
            # Don't let fact-checking failures break the entire prediction
            return None
    
    def _calculate_weighted_confidence(self, analyses):
        """Calculate weighted confidence score from multiple analyses."""
        weights = {
            'zero_shot': 0.4,
            'trained_model': 0.5,
            'sentiment': 0.05,
            'behavioral': 0.05
        }
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for analysis_type, analysis_result in analyses.items():
            if analysis_type in weights and isinstance(analysis_result, dict):
                confidence = analysis_result.get('confidence', 0.0)
                if confidence > 0:
                    weighted_confidence += confidence * weights[analysis_type]
                    total_weight += weights[analysis_type]
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _get_keyword_fallback_prediction(self, text):
        """Fallback prediction using keyword analysis."""
        text_lower = text.lower()
        misinformation_keywords = ['breaking', 'urgent', 'shocking', 'exposed', 'leaked', 'secret', 'hidden truth']
        legitimate_keywords = ['official', 'confirmed', 'according to', 'reported by', 'statement']
        
        misinformation_score = sum(1 for keyword in misinformation_keywords if keyword in text_lower)
        legitimate_score = sum(1 for keyword in legitimate_keywords if keyword in text_lower)
        
        if misinformation_score > legitimate_score:
            prediction = 1
            confidence = 0.6 + (misinformation_score * 0.1)
        else:
            prediction = 0
            confidence = 0.6 + (legitimate_score * 0.1)
        
        return {
            'prediction': prediction,
            'confidence': min(confidence, 0.9),  # Cap at 0.9
            'method': 'keyword_analysis_fallback'
        }
    
    def find_best_dataset_for_prediction(self, model_id=None):
        """Find the best dataset to use for prediction based on available data."""
        datasets_dir = Path('datasets')
        
        # If model_id specifies a dataset, extract it
        if model_id and model_id != 'zero_shot' and '_' in model_id:
            # Look for common model types at the end
            model_types = ['logistic_regression', 'naive_bayes', 'random_forest', 'svm', 'xgboost']
            
            for model_type in model_types:
                if model_id.endswith('_' + model_type):
                    dataset_name = model_id[:-len('_' + model_type)]
                    if (datasets_dir / dataset_name).exists():
                        return dataset_name, model_type
            
            # Fallback to original logic if no model type matches
            parts = model_id.rsplit('_', 1)
            if len(parts) == 2:
                dataset_name = parts[0]
                if (datasets_dir / dataset_name).exists():
                    return dataset_name, parts[1]
        
        # Otherwise, find the best available dataset with saved results
        # Prioritize datasets with zero-shot results for better predictions
        datasets_with_zero_shot = []
        datasets_with_models = []
        
        for dataset_dir in datasets_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                if dataset_name not in ['default', 'test', 'json_test', 'test_dataset']:
                    zero_shot_csv = dataset_dir / 'processed' / 'zero_shot_labeled.csv'
                    models_dir = dataset_dir / 'models'
                    
                    if zero_shot_csv.exists():
                        datasets_with_zero_shot.append(dataset_name)
                    elif models_dir.exists() and any(models_dir.glob('*.joblib')):
                        datasets_with_models.append(dataset_name)
        
        # Prefer datasets with zero-shot results
        if datasets_with_zero_shot:
            return datasets_with_zero_shot[0], None
        elif datasets_with_models:
            return datasets_with_models[0], None
        
        return None, None
    
    def get_available_models(self, dataset_name):
        """Get all available trained models for a dataset."""
        if not self.model_evaluator:
            return {}
            
        try:
            return self.model_evaluator.get_available_models(dataset_name)
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return {}


def create_prediction_service(model_evaluator=None, shap_explainer=None, file_manager=None):
    """Factory function to create a prediction service instance."""
    return PredictionService(model_evaluator, shap_explainer, file_manager)