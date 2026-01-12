"""
Sentiment Analysis Module

This module provides comprehensive sentiment analysis capabilities using transformer-based
models with language-adaptive routing. It implements multiple sentiment analysis strategies
including VADER, transformer models, and multilingual approaches for robust sentiment
detection in social media content across different languages.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
from pathlib import Path

# Sentiment analysis imports
try:
    import warnings
    # Suppress transformers warnings about unused weights
    warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
    
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import logging as transformers_logging
    # Set transformers logging to ERROR to suppress warnings
    transformers_logging.set_verbosity_error()
    
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    """
    Language-Adaptive Sentiment Analysis Class
    
    Implements comprehensive sentiment analysis with language-specific routing
    and multiple analysis strategies. Supports transformer-based models for
    English and multilingual content, with VADER sentiment analysis as fallback.
    Provides robust sentiment detection across different languages and contexts.
    """
    
    def __init__(self, file_manager):
        self.logger = logging.getLogger(__name__)
        self.file_manager = file_manager
        
        # Initialize VADER as fallback
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Sentiment model configurations
        self.sentiment_models = {
            'english': {
                'cardiffnlp/twitter-roberta-base-sentiment-latest': {
                    'type': 'transformer',
                    'language': 'en',
                    'labels': ['NEGATIVE', 'NEUTRAL', 'POSITIVE'],
                    'description': 'Twitter-trained RoBERTa for English sentiment'
                },
                'distilbert-base-uncased-finetuned-sst-2-english': {
                    'type': 'transformer',
                    'language': 'en',
                    'labels': ['NEGATIVE', 'POSITIVE'],
                    'description': 'DistilBERT for English sentiment (binary)'
                }
            },
            'multilingual': {
                'cardiffnlp/twitter-xlm-roberta-base-sentiment': {
                    'type': 'transformer',
                    'language': 'multilingual',
                    'labels': ['NEGATIVE', 'NEUTRAL', 'POSITIVE'],
                    'description': 'XLM-RoBERTa for multilingual sentiment'
                },
                'nlptown/bert-base-multilingual-uncased-sentiment': {
                    'type': 'transformer',
                    'language': 'multilingual',
                    'labels': ['NEGATIVE', 'POSITIVE'],
                    'description': 'Multilingual BERT for sentiment'
                }
            }
        }
        
        # Cache for loaded models
        self.loaded_sentiment_models = {}
        
        # Kenyan context sentiment indicators
        self.kenyan_sentiment_indicators = {
            'positive': [
                'poa', 'sawa', 'vizuri', 'nzuri', 'asante', 'hongera', 'furaha',
                'maendeleo', 'mafanikio', 'amani', 'upendo', 'heshima'
            ],
            'negative': [
                'mbaya', 'vibaya', 'hasira', 'uchungu', 'huzuni', 'wasiwasi',
                'uongozi mbaya', 'rushwa', 'udhalimu', 'ukosefu', 'tatizo'
            ],
            'political_positive': [
                'uongozi mzuri', 'sera nzuri', 'maendeleo', 'uongozi', 'amani',
                'umoja', 'matumaini', 'mabadiliko mazuri'
            ],
            'political_negative': [
                'uongozi mbaya', 'rushwa', 'udhalimu', 'ubaguzi', 'ghasia',
                'machafuko', 'uongozi duni', 'sera mbaya'
            ]
        }
        
        self.logger.info("Sentiment analyzer initialized")
    
    def analyze_sentiment(self, text: str, language: str = 'en', 
                         use_transformer: bool = True) -> Dict[str, Any]:
        """
        Analyze sentiment of text using language-appropriate model.
        
        Args:
            text: Text to analyze
            language: Detected language ('en', 'sw', 'mixed')
            use_transformer: Whether to use transformer models
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or len(text.strip()) < 3:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0},
                'method': 'insufficient_text'
            }
        
        # Clean text
        clean_text = self._clean_text_for_sentiment(text)
        
        # Select appropriate method
        if use_transformer and TRANSFORMERS_AVAILABLE:
            result = self._analyze_with_transformer(clean_text, language)
        else:
            result = self._analyze_with_vader(clean_text, language)
        
        # Add Kenyan context enhancement
        result = self._enhance_with_kenyan_context(clean_text, result)
        
        return result
    
    def _clean_text_for_sentiment(self, text: str) -> str:
        """Clean text for sentiment analysis."""
        import re
        
        # Remove URLs but keep mentions and hashtags (they carry sentiment)
        clean_text = re.sub(r'http\S+|www\S+', '', text)
        
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def _analyze_with_transformer(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze sentiment using transformer models."""
        try:
            # Select model based on language
            if language == 'en':
                model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
                model_category = 'english'
            else:  # sw, mixed, or unknown
                model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
                model_category = 'multilingual'
            
            # Load model if not cached
            if model_name not in self.loaded_sentiment_models:
                self.logger.info(f"Loading sentiment model: {model_name}")
                try:
                    sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        tokenizer=model_name,
                        return_all_scores=True
                    )
                    self.loaded_sentiment_models[model_name] = sentiment_pipeline
                except Exception as e:
                    self.logger.warning(f"Failed to load transformer model: {e}")
                    return self._analyze_with_vader(text, language)
            
            # Get sentiment pipeline
            sentiment_pipeline = self.loaded_sentiment_models[model_name]
            
            # Analyze sentiment
            results = sentiment_pipeline(text[:512])  # Limit text length
            
            # Process results
            if isinstance(results[0], list):
                scores = {result['label'].lower(): result['score'] for result in results[0]}
            else:
                scores = {results[0]['label'].lower(): results[0]['score']}
            
            # Normalize labels
            normalized_scores = self._normalize_sentiment_scores(scores)
            
            # Determine primary sentiment
            primary_sentiment = max(normalized_scores, key=normalized_scores.get)
            confidence = normalized_scores[primary_sentiment]
            
            return {
                'sentiment': primary_sentiment,
                'confidence': confidence,
                'scores': normalized_scores,
                'method': f'transformer_{model_category}',
                'model_used': model_name,
                'language_processed': language
            }
            
        except Exception as e:
            self.logger.error(f"Error in transformer sentiment analysis: {e}")
            return self._analyze_with_vader(text, language)
    
    def _analyze_with_vader(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER (fallback method)."""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Convert VADER scores to standard format
            normalized_scores = {
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg']
            }
            
            # Determine primary sentiment
            if scores['compound'] >= 0.05:
                primary_sentiment = 'positive'
                confidence = scores['pos']
            elif scores['compound'] <= -0.05:
                primary_sentiment = 'negative'
                confidence = scores['neg']
            else:
                primary_sentiment = 'neutral'
                confidence = scores['neu']
            
            return {
                'sentiment': primary_sentiment,
                'confidence': confidence,
                'scores': normalized_scores,
                'compound_score': scores['compound'],
                'method': 'vader',
                'language_processed': language
            }
            
        except Exception as e:
            self.logger.error(f"Error in VADER sentiment analysis: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'scores': {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0},
                'method': 'error_fallback'
            }
    
    def _normalize_sentiment_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize sentiment scores to standard format."""
        normalized = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        for label, score in scores.items():
            label_lower = label.lower()
            
            if 'pos' in label_lower or label_lower == 'positive':
                normalized['positive'] = score
            elif 'neg' in label_lower or label_lower == 'negative':
                normalized['negative'] = score
            elif 'neu' in label_lower or label_lower == 'neutral':
                normalized['neutral'] = score
            elif label_lower in ['label_0', '0']:  # Some models use numeric labels
                normalized['negative'] = score
            elif label_lower in ['label_1', '1']:
                normalized['positive'] = score
            elif label_lower in ['label_2', '2']:
                normalized['neutral'] = score
        
        # Ensure scores sum to 1
        total = sum(normalized.values())
        if total > 0:
            normalized = {k: v/total for k, v in normalized.items()}
        else:
            normalized = {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
        
        return normalized
    
    def _enhance_with_kenyan_context(self, text: str, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance sentiment analysis with Kenyan context indicators."""
        text_lower = text.lower()
        
        # Count Kenyan sentiment indicators
        kenyan_positive = sum(1 for word in self.kenyan_sentiment_indicators['positive'] 
                             if word in text_lower)
        kenyan_negative = sum(1 for word in self.kenyan_sentiment_indicators['negative'] 
                             if word in text_lower)
        
        # Political context
        political_positive = sum(1 for phrase in self.kenyan_sentiment_indicators['political_positive'] 
                               if phrase in text_lower)
        political_negative = sum(1 for phrase in self.kenyan_sentiment_indicators['political_negative'] 
                               if phrase in text_lower)
        
        # Calculate enhancement factor
        total_kenyan_indicators = kenyan_positive + kenyan_negative + political_positive + political_negative
        
        if total_kenyan_indicators > 0:
            # Adjust sentiment based on Kenyan indicators
            kenyan_sentiment_score = (
                (kenyan_positive + political_positive) - (kenyan_negative + political_negative)
            ) / total_kenyan_indicators
            
            # Blend with original sentiment
            original_compound = (
                base_result['scores']['positive'] - base_result['scores']['negative']
            )
            
            # Weight: 70% original, 30% Kenyan context
            blended_compound = 0.7 * original_compound + 0.3 * kenyan_sentiment_score
            
            # Update scores
            if blended_compound > 0.1:
                enhanced_sentiment = 'positive'
                enhanced_confidence = min(base_result['confidence'] + 0.1, 1.0)
            elif blended_compound < -0.1:
                enhanced_sentiment = 'negative'
                enhanced_confidence = min(base_result['confidence'] + 0.1, 1.0)
            else:
                enhanced_sentiment = base_result['sentiment']
                enhanced_confidence = base_result['confidence']
            
            # Add Kenyan context information
            base_result.update({
                'sentiment': enhanced_sentiment,
                'confidence': enhanced_confidence,
                'kenyan_context': {
                    'kenyan_positive_indicators': kenyan_positive,
                    'kenyan_negative_indicators': kenyan_negative,
                    'political_positive_indicators': political_positive,
                    'political_negative_indicators': political_negative,
                    'total_kenyan_indicators': total_kenyan_indicators,
                    'kenyan_sentiment_score': kenyan_sentiment_score,
                    'context_enhanced': True
                }
            })
        else:
            base_result['kenyan_context'] = {
                'context_enhanced': False,
                'total_kenyan_indicators': 0
            }
        
        return base_result
    
    def process_dataset_sentiment(self, dataset_name: str, language_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process entire dataset for sentiment analysis.
        
        Args:
            dataset_name: Name of the dataset
            language_results: Results from language detection
            
        Returns:
            Dictionary with sentiment analysis results
        """
        self.logger.info(f"Processing sentiment analysis for dataset: {dataset_name}")
        
        try:
            # Load processed data
            processed_file = Path('datasets') / dataset_name / 'processed' / 'processed_data.csv'
            if not processed_file.exists():
                raise FileNotFoundError('Processed dataset not found')
            
            df = pd.read_csv(processed_file)
            
            # Find text column
            text_column = None
            # Check for text columns in order of preference
            for col in ['COMBINED_TEXT', 'CLEANED_TEXT', 'TWEET_CONTENT', 'TEXT', 'CONTENT', 'MESSAGE']:
                if col in df.columns:
                    text_column = col
                    break
            
            if not text_column:
                raise ValueError('No text column found in dataset')
            
            # Process sentiment for each text
            results = []
            sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
            method_counts = {}
            
            # Use language detection results if available
            dominant_language = language_results.get('dominant_language', 'en')
            
            # Limit processing for performance
            sample_size = min(len(df), 1000)
            df_sample = df.head(sample_size)
            
            for idx, row in df_sample.iterrows():
                text = str(row[text_column]) if pd.notna(row[text_column]) else ""
                
                if len(text.strip()) < 3:
                    continue
                
                # Analyze sentiment
                sentiment_result = self.analyze_sentiment(
                    text, 
                    language=dominant_language,
                    use_transformer=True
                )
                
                # Store result
                result_record = {
                    'index': idx,
                    'text_preview': text[:100] + '...' if len(text) > 100 else text,
                    **sentiment_result
                }
                
                results.append(result_record)
                
                # Update counts
                sentiment_counts[sentiment_result['sentiment']] += 1
                method = sentiment_result['method']
                method_counts[method] = method_counts.get(method, 0) + 1
            
            # Calculate statistics
            total_processed = len(results)
            sentiment_distribution = {
                sentiment: count / total_processed 
                for sentiment, count in sentiment_counts.items()
                if count > 0
            }
            
            # Calculate average confidence by sentiment
            avg_confidence = {}
            for sentiment in sentiment_counts:
                sentiment_results = [r for r in results if r['sentiment'] == sentiment]
                if sentiment_results:
                    avg_confidence[sentiment] = np.mean([r['confidence'] for r in sentiment_results])
                else:
                    avg_confidence[sentiment] = 0.0
            
            # Identify high-emotion content (for misinformation susceptibility)
            high_emotion_threshold = 0.8
            high_emotion_content = [
                r for r in results 
                if r['confidence'] > high_emotion_threshold and r['sentiment'] != 'neutral'
            ]
            
            # Create summary
            summary = {
                'dataset_name': dataset_name,
                'analysis_date': datetime.now().isoformat(),
                'total_samples_analyzed': total_processed,
                'total_samples_in_dataset': len(df),
                'dominant_language_used': dominant_language,
                'sentiment_distribution': sentiment_distribution,
                'sentiment_counts': sentiment_counts,
                'method_counts': method_counts,
                'average_confidence_by_sentiment': avg_confidence,
                'high_emotion_content_count': len(high_emotion_content),
                'high_emotion_percentage': len(high_emotion_content) / total_processed * 100,
                'detailed_results': results[:50],  # Store first 50 for display
                'sample_sentiments': {
                    'positive': [r['text_preview'] for r in results if r['sentiment'] == 'positive'][:5],
                    'negative': [r['text_preview'] for r in results if r['sentiment'] == 'negative'][:5],
                    'neutral': [r['text_preview'] for r in results if r['sentiment'] == 'neutral'][:5]
                },
                'kenyan_context_enhanced': sum(1 for r in results if r.get('kenyan_context', {}).get('context_enhanced', False)),
                'theoretical_insights': {
                    'rct_emotional_benefit_score': sentiment_distribution.get('positive', 0) + sentiment_distribution.get('negative', 0),
                    'rat_suitable_target_score': len(high_emotion_content) / total_processed,
                    'emotional_manipulation_risk': sentiment_distribution.get('negative', 0) * 1.2 + sentiment_distribution.get('positive', 0) * 0.8
                }
            }
            
            # Save results
            self.file_manager.save_results(dataset_name, summary, 'sentiment_analysis')
            
            # Update dataset info
            self.file_manager.update_dataset_info(dataset_name, {
                'sentiment_analysis_completed': True,
                'sentiment_distribution': sentiment_distribution,
                'high_emotion_percentage': len(high_emotion_content) / total_processed * 100
            })
            
            self.logger.info(f"Sentiment analysis completed. Distribution: {sentiment_distribution}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            raise
    
    def analyze_dataset_sentiment(self, dataset_name: str, language_results: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process_dataset_sentiment for backward compatibility."""
        return self.process_dataset_sentiment(dataset_name, language_results)
    
    def extract_sentiment_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Extract sentiment features for machine learning.
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            
        Returns:
            DataFrame with sentiment features
        """
        self.logger.info("Extracting sentiment features for ML")
        
        sentiment_features = pd.DataFrame(index=df.index)
        
        # Process each text
        for idx, row in df.iterrows():
            text = str(row[text_column]) if pd.notna(row[text_column]) else ""
            
            if len(text.strip()) < 3:
                # Default values for empty text
                sentiment_features.loc[idx, 'sentiment_positive'] = 0.0
                sentiment_features.loc[idx, 'sentiment_neutral'] = 1.0
                sentiment_features.loc[idx, 'sentiment_negative'] = 0.0
                sentiment_features.loc[idx, 'sentiment_confidence'] = 0.0
                sentiment_features.loc[idx, 'sentiment_compound'] = 0.0
                continue
            
            # Analyze sentiment
            result = self.analyze_sentiment(text, use_transformer=False)  # Use VADER for speed
            
            # Extract features
            sentiment_features.loc[idx, 'sentiment_positive'] = result['scores']['positive']
            sentiment_features.loc[idx, 'sentiment_neutral'] = result['scores']['neutral']
            sentiment_features.loc[idx, 'sentiment_negative'] = result['scores']['negative']
            sentiment_features.loc[idx, 'sentiment_confidence'] = result['confidence']
            sentiment_features.loc[idx, 'sentiment_compound'] = result.get('compound_score', 0.0)
            
            # Kenyan context features
            kenyan_context = result.get('kenyan_context', {})
            sentiment_features.loc[idx, 'kenyan_positive_indicators'] = kenyan_context.get('kenyan_positive_indicators', 0)
            sentiment_features.loc[idx, 'kenyan_negative_indicators'] = kenyan_context.get('kenyan_negative_indicators', 0)
            sentiment_features.loc[idx, 'political_sentiment_indicators'] = (
                kenyan_context.get('political_positive_indicators', 0) + 
                kenyan_context.get('political_negative_indicators', 0)
            )
        
        # Additional derived features
        sentiment_features['sentiment_polarity'] = (
            sentiment_features['sentiment_positive'] - sentiment_features['sentiment_negative']
        )
        
        sentiment_features['sentiment_intensity'] = (
            sentiment_features['sentiment_positive'] + sentiment_features['sentiment_negative']
        )
        
        sentiment_features['is_high_emotion'] = (
            sentiment_features['sentiment_intensity'] > 0.7
        ).astype(int)
        
        self.logger.info(f"Extracted {sentiment_features.shape[1]} sentiment features")
        return sentiment_features

def main():
    """Test sentiment analysis functionality."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test samples
    test_texts = [
        "I love the new government policies, they are amazing!",
        "This is terrible news about corruption in the government.",
        "Rais Ruto ana sera nzuri za maendeleo.",
        "Hii ni habari mbaya kuhusu rushwa serikalini.",
        "This fake news ina spread misinformation but watu wanaamini.",
        "The election results are neutral and fair."
    ]
    
    from src.utils.file_manager import FileManager
    file_manager = FileManager()
    analyzer = SentimentAnalyzer(file_manager)
    
    print("💭 SENTIMENT ANALYSIS TEST")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze_sentiment(text, language='mixed', use_transformer=False)
        
        print(f"\n{i}. Text: {text}")
        print(f"   Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
        print(f"   Scores: P={result['scores']['positive']:.2f}, "
              f"N={result['scores']['neutral']:.2f}, "
              f"Neg={result['scores']['negative']:.2f}")
        print(f"   Method: {result['method']}")
        
        if 'kenyan_context' in result and result['kenyan_context']['context_enhanced']:
            kenyan = result['kenyan_context']
            print(f"   Kenyan Context: +{kenyan['kenyan_positive_indicators']} "
                  f"-{kenyan['kenyan_negative_indicators']} "
                  f"(Political: +{kenyan['political_positive_indicators']} "
                  f"-{kenyan['political_negative_indicators']})")

if __name__ == "__main__":
    main()