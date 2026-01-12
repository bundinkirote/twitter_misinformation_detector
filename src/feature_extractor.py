"""
Feature Extraction Module

This module provides comprehensive feature extraction capabilities for machine learning
models. It implements multiple feature extraction strategies including textual features,
behavioral patterns, network analysis, and theoretical framework-based features for
misinformation detection in social media content.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
from src.utils.file_manager import FileManager
from src.language_detector import LanguageDetector
from src.sentiment_analyzer import SentimentAnalyzer
from src.theoretical_frameworks import TheoreticalFrameworks
from src.local_model_manager import get_model_manager
from src.model_compatibility import get_compatibility_manager
import re
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import textstat
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """
    Feature Extraction Class
    
    Implements comprehensive feature extraction for misinformation detection including
    textual analysis, sentiment analysis, linguistic patterns, behavioral features,
    and theoretical framework-based indicators. Supports multiple feature combination
    strategies for optimal model performance.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_manager = FileManager()
        
        # Initialize new components
        self.language_detector = LanguageDetector(self.file_manager)
        self.sentiment_analyzer = SentimentAnalyzer(self.file_manager)
        self.theoretical_frameworks = TheoreticalFrameworks()
        self.model_manager = get_model_manager()
        self.compatibility_manager = get_compatibility_manager()
        
        # Initialize enhanced text analysis components
        self.sentiment_intensity_analyzer = SentimentIntensityAnalyzer()
        self._download_nltk_data()
        
        # Initialize stop words
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            
        # Behavioral patterns (compiled regex for efficiency)
        self._compile_behavioral_patterns()
        
        # LDA model storage
        self.lda_models = {}
        self.topic_vectorizers = {}
    
    def _load_previous_analysis_results(self, dataset_name, exclude_zero_shot=False):
        """Load results from previous analysis steps (sentiment, behavioral, etc.)."""
        self.logger.info("Loading previous analysis results for feature integration...")
        
        results = {}
        
        try:
            # Load content analysis results (includes sentiment analysis)
            content_results = self.file_manager.load_results(dataset_name, 'content_analysis')
            if content_results:
                results['content_analysis'] = content_results
                self.logger.info("✅ Content analysis results loaded")
            
            # Load behavioral profiling results
            behavioral_results = self.file_manager.load_results(dataset_name, 'behavioral_profiling')
            if behavioral_results:
                results['behavioral_profiling'] = behavioral_results
                self.logger.info("✅ Behavioral profiling results loaded")
            
            # Load zero-shot classification results (unless excluded)
            if not exclude_zero_shot:
                zero_shot_results = self.file_manager.load_results(dataset_name, 'zero_shot_classification')
                if zero_shot_results:
                    results['zero_shot_classification'] = zero_shot_results
                    self.logger.info("✅ Zero-shot classification results loaded")
            
            # Load language detection results
            language_results = self.file_manager.load_results(dataset_name, 'language_detection')
            if language_results:
                results['language_detection'] = language_results
                self.logger.info("✅ Language detection results loaded")
                
            return results
            
        except Exception as e:
            self.logger.warning(f"Error loading previous analysis results: {e}")
            return {}
    
    def _extract_features_from_previous_results(self, previous_results, df_length):
        """Extract features from previous analysis results instead of recalculating."""
        self.logger.info("Extracting features from previous analysis results...")
        
        all_features = []
        feature_names = []
        
        # Extract sentiment features from content analysis results
        if 'content_analysis' in previous_results:
            content_results = previous_results['content_analysis']
            
            # Extract sentiment scores if available
            if 'sentiment_analysis' in content_results:
                sentiment_data = content_results['sentiment_analysis']
                
                # Try to extract individual sentiment scores
                sentiment_features = []
                sentiment_names = []
                
                if 'average_scores' in sentiment_data:
                    avg_scores = sentiment_data['average_scores']
                    for score_type in ['positive', 'negative', 'neutral', 'compound']:
                        if score_type in avg_scores:
                            # Create feature array with the average score for all samples
                            feature_array = np.full(df_length, avg_scores[score_type])
                            sentiment_features.append(feature_array.reshape(-1, 1))
                            sentiment_names.append(f'sentiment_{score_type}_avg')
                
                if sentiment_features:
                    sentiment_feature_matrix = np.hstack(sentiment_features)
                    all_features.append(sentiment_feature_matrix)
                    feature_names.extend(sentiment_names)
                    self.logger.info(f"✅ Extracted {len(sentiment_names)} sentiment features from previous results")
        
        # Extract behavioral features from behavioral profiling results
        if 'behavioral_profiling' in previous_results:
            behavioral_results = previous_results['behavioral_profiling']
            
            behavioral_features = []
            behavioral_names = []
            
            # Extract behavioral metrics
            if 'behavioral_metrics' in behavioral_results:
                metrics = behavioral_results['behavioral_metrics']
                
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        # Create feature array with the metric value for all samples
                        feature_array = np.full(df_length, metric_value)
                        behavioral_features.append(feature_array.reshape(-1, 1))
                        behavioral_names.append(f'behavioral_{metric_name}')
            
            # Extract gratification features
            if 'gratification_analysis' in behavioral_results:
                gratification_data = behavioral_results['gratification_analysis']
                
                for grat_type, grat_value in gratification_data.items():
                    if isinstance(grat_value, (int, float)):
                        feature_array = np.full(df_length, grat_value)
                        behavioral_features.append(feature_array.reshape(-1, 1))
                        behavioral_names.append(f'gratification_{grat_type}')
            
            if behavioral_features:
                behavioral_feature_matrix = np.hstack(behavioral_features)
                all_features.append(behavioral_feature_matrix)
                feature_names.extend(behavioral_names)
                self.logger.info(f"✅ Extracted {len(behavioral_names)} behavioral features from previous results")
        
        # Extract classification features from zero-shot results (dataset-level metrics)
        if 'zero_shot_classification' in previous_results:
            zero_shot_results = previous_results['zero_shot_classification']
            
            classification_features = []
            classification_names = []
            
            # Extract dataset-level classification metrics as features
            if 'average_confidence' in zero_shot_results:
                feature_array = np.full(df_length, zero_shot_results['average_confidence'])
                classification_features.append(feature_array.reshape(-1, 1))
                classification_names.append('zeroshot_avg_confidence')
            
            if 'classification_distribution' in zero_shot_results:
                dist = zero_shot_results['classification_distribution']
                for class_type, percentage in dist.items():
                    feature_array = np.full(df_length, percentage)
                    classification_features.append(feature_array.reshape(-1, 1))
                    classification_names.append(f'zeroshot_{class_type}_ratio')
            
            if 'confidence_distribution' in zero_shot_results:
                conf_dist = zero_shot_results['confidence_distribution']
                for conf_level, percentage in conf_dist.items():
                    feature_array = np.full(df_length, percentage)
                    classification_features.append(feature_array.reshape(-1, 1))
                    classification_names.append(f'zeroshot_{conf_level}_ratio')
            
            if classification_features:
                classification_feature_matrix = np.hstack(classification_features)
                all_features.append(classification_feature_matrix)
                feature_names.extend(classification_names)
                self.logger.info(f"✅ Extracted {len(classification_names)} zero-shot dataset metrics as features")
        
        # Extract language features from language detection results
        if 'language_detection' in previous_results:
            language_results = previous_results['language_detection']
            
            language_features = []
            language_names = []
            
            if 'language_distribution' in language_results:
                lang_dist = language_results['language_distribution']
                
                for lang, percentage in lang_dist.items():
                    feature_array = np.full(df_length, percentage / 100.0)  # Convert to 0-1 scale
                    language_features.append(feature_array.reshape(-1, 1))
                    language_names.append(f'language_{lang}_percentage')
            
            if language_features:
                language_feature_matrix = np.hstack(language_features)
                all_features.append(language_feature_matrix)
                feature_names.extend(language_names)
                self.logger.info(f"✅ Extracted {len(language_names)} language features from previous results")
        
        # Categorize features for better analysis
        feature_categories = self.categorize_features(feature_names)
        
        return all_features, feature_names, feature_categories
    
    def _run_and_extract_zero_shot_features(self, df, dataset_name, zero_shot_config):
        """Run zero-shot classification and extract features from the results."""
        try:
            self.logger.info("Running zero-shot classification as part of feature engineering...")
            
            # Initialize zero-shot classifier
            from .zero_shot_labeling import ZeroShotLabeler
            zero_shot_classifier = ZeroShotLabeler(self.file_manager)
            
            # Run zero-shot classification
            zero_shot_results = zero_shot_classifier.classify_dataset(dataset_name, zero_shot_config)
            
            # Save results for later use
            self.file_manager.save_results(dataset_name, zero_shot_results, 'zero_shot_classification')
            
            # Extract features from the results
            features = []
            feature_names = []
            
            # Dataset-level classification metrics as features
            if 'average_confidence' in zero_shot_results:
                feature_array = np.full(len(df), zero_shot_results['average_confidence'])
                features.append(feature_array.reshape(-1, 1))
                feature_names.append('zeroshot_avg_confidence')
            
            if 'classification_distribution' in zero_shot_results:
                dist = zero_shot_results['classification_distribution']
                for class_type, percentage in dist.items():
                    feature_array = np.full(len(df), percentage)
                    features.append(feature_array.reshape(-1, 1))
                    feature_names.append(f'zeroshot_{class_type}_ratio')
            
            if 'confidence_distribution' in zero_shot_results:
                conf_dist = zero_shot_results['confidence_distribution']
                for conf_level, percentage in conf_dist.items():
                    feature_array = np.full(len(df), percentage)
                    features.append(feature_array.reshape(-1, 1))
                    feature_names.append(f'zeroshot_{conf_level}_ratio')
            
            # Individual sample features (if available)
            if 'predictions' in zero_shot_results and len(zero_shot_results['predictions']) == len(df):
                # Convert predictions to numeric features
                predictions = zero_shot_results['predictions']
                prediction_numeric = []
                for pred in predictions:
                    if pred == 'misinformation':
                        prediction_numeric.append(1)
                    elif pred == 'legitimate':
                        prediction_numeric.append(0)
                    else:  # uncertain
                        prediction_numeric.append(0.5)
                
                features.append(np.array(prediction_numeric).reshape(-1, 1))
                feature_names.append('zeroshot_prediction')
            
            if 'confidence_scores' in zero_shot_results and len(zero_shot_results['confidence_scores']) == len(df):
                confidence_scores = np.array(zero_shot_results['confidence_scores'])
                features.append(confidence_scores.reshape(-1, 1))
                feature_names.append('zeroshot_confidence')
            
            # Combine all zero-shot features
            if features:
                zero_shot_feature_matrix = np.hstack(features)
            else:
                # Fallback if no features extracted
                zero_shot_feature_matrix = np.zeros((len(df), 1))
                feature_names = ['zeroshot_dummy']
            
            self.logger.info(f"✅ Extracted {len(feature_names)} zero-shot features from classification results")
            return zero_shot_feature_matrix, feature_names
            
        except Exception as e:
            self.logger.error(f"Error running zero-shot classification: {e}")
            # Return dummy features on error
            dummy_features = np.zeros((len(df), 1))
            dummy_names = ['zeroshot_error']
            return dummy_features, dummy_names
    
    def _safe_column_access(self, df: pd.DataFrame, column_name: str, default_value=0):
        """Safely access a column with exact column name matching, returning default value if column doesn't exist."""
        # Define column name variations (based on exact dataset column names)
        column_variations = {
            # User profile columns (from labeled-data.csv)
            'FOLLOWERS': ['Followers', 'FOLLOWERS_COUNT', 'FOLLOWERS', 'followers_count'],
            'FOLLOWED': ['Followed', 'FOLLOWING_COUNT', 'FOLLOWED', 'following_count'],
            'TWEETS': ['Tweets', 'TWEET_COUNT', 'TWEETS_COUNT', 'tweet_count'],
            'VERIFIED': ['Verified', 'IS_VERIFIED', 'VERIFIED', 'Is Blue Verified', 'is_verified'],
            'DESCRIPTION': ['Description', 'BIO', 'PROFILE_DESCRIPTION', 'USER_DESCRIPTION', 'DESCRIPTION'],
            'LOCATION': ['Location', 'USER_LOCATION', 'PROFILE_LOCATION', 'LOCATION'],
            'URL': ['URL', 'URLS', 'LINKS'],
            
            # Tweet-level columns (from RutoMustGo.csv and RejectFinanceBill2024.csv)
            'MENTIONS_IN_TWEET': ['Mentions in Tweet', 'MENTIONS IN TWEET', 'MENTIONS_IN_TWEET', 'mentions_in_tweet'],
            'RETWEET_COUNT': ['Retweet Count', 'RETWEET COUNT', 'RETWEET_COUNT', 'retweet_count'],
            'FAVORITE_COUNT': ['Favorite Count', 'FAVORITE COUNT', 'FAVORITE_COUNT', 'favorite_count'],
            'REPLY_COUNT': ['Reply Count', 'REPLY COUNT', 'REPLY_COUNT', 'reply_count'],
            'QUOTE_COUNT': ['Quote Count', 'QUOTE COUNT', 'QUOTE_COUNT', 'quote_count'],
            'HASHTAGS': ['Hashtags in Tweet', 'HASHTAGS IN TWEET', 'HASHTAGS', 'hashtags'],
            'URLS_IN_TWEET': ['URLs in Tweet', 'URLS IN TWEET', 'URLS_IN_TWEET', 'urls_in_tweet'],
            'MEDIA_IN_TWEET': ['Media in Tweet', 'MEDIA IN TWEET', 'MEDIA_IN_TWEET', 'media_in_tweet'],
            'AUTHOR_ID': ['Author ID', 'AUTHOR_ID', 'author_id', 'User ID'],
            'TWEET_DATE': ['Tweet Date (UTC)', 'TWEET_DATE', 'tweet_date'],
            'SOURCE': ['Source', 'SOURCE', 'source'],
            'LANGUAGE': ['Language', 'LANGUAGE', 'language'],
            
            # Account/User features
            'ACCOUNT_CREATED': ['Account Created', 'ACCOUNT_CREATED', 'account_created'],
            'FOLLOWERS_COUNT': ['Followers Count', 'FOLLOWERS_COUNT', 'followers_count', 'Followers'],
            'FOLLOWING_COUNT': ['Following Count', 'FOLLOWING_COUNT', 'following_count', 'Followed'],
            'FOLLOWERS_FOLLOWING_RATIO': ['FOLLOWERS_FOLLOWING_RATIO', 'followers_following_ratio'],
            'TWEET_COUNT': ['Tweet Count', 'TWEET_COUNT', 'tweet_count', 'Tweets'],
            'LISTED_COUNT': ['Listed Count', 'LISTED_COUNT', 'listed_count'],
            
            # Text features
            'TEXT_LENGTH': ['TEXT_LENGTH', 'text_length'],
            'WORD_COUNT': ['WORD_COUNT', 'word_count'],
            'READABILITY_SCORE': ['READABILITY_SCORE', 'readability_score'],
            'TWEET_CONTENT': ['TWEET', 'Tweet', 'TWEET_CONTENT', 'tweet_content', 'COMBINED_TEXT', 'CLEANED_TEXT'],
            
            # Sentiment features
            'SENTIMENT_POSITIVE': ['SENTIMENT_POSITIVE', 'sentiment_positive'],
            'SENTIMENT_NEGATIVE': ['SENTIMENT_NEGATIVE', 'sentiment_negative'],
            'SENTIMENT_NEUTRAL': ['SENTIMENT_NEUTRAL', 'sentiment_neutral'],
            'SENTIMENT_COMPOUND': ['SENTIMENT_COMPOUND', 'sentiment_compound']
        }
        
        # Find the actual column to use
        actual_column = None
        
        # Check exact match first
        if column_name in df.columns:
            actual_column = column_name
        # Check variations
        elif column_name in column_variations:
            for variation in column_variations[column_name]:
                if variation in df.columns:
                    actual_column = variation
                    break
        
        if actual_column:
            col_data = df[actual_column].copy()
            
            # Handle boolean columns
            if isinstance(default_value, bool):
                if col_data.dtype == 'object':
                    # Map string boolean values to integers first, then convert to bool
                    col_data = col_data.map({
                        'TRUE': 1, 'True': 1, 'true': 1, True: 1, 1: 1, '1': 1,
                        'FALSE': 0, 'False': 0, 'false': 0, False: 0, 0: 0, '0': 0
                    }).fillna(1 if default_value else 0)
                    col_data = col_data.astype(bool)
                else:
                    col_data = pd.to_numeric(col_data, errors='coerce').fillna(1 if default_value else 0).astype(bool)
            
            # Handle numeric columns
            elif isinstance(default_value, (int, float)):
                col_data = pd.to_numeric(col_data, errors='coerce').fillna(default_value)
            
            # Handle string columns
            else:
                col_data = col_data.fillna(default_value).astype(str)
            
            return col_data
        else:
            # Column not found - return default values
            if isinstance(default_value, bool):
                return pd.Series([default_value] * len(df), dtype=bool)
            elif isinstance(default_value, (int, float)):
                return pd.Series([default_value] * len(df), dtype=type(default_value))
            else:
                return pd.Series([default_value] * len(df), dtype=str)
    
    def extract_features(self, dataset_name, feature_types):
        """Extract specified types of features, integrating previous analysis results."""
        self.logger.info(f"Extracting features for dataset: {dataset_name}")
        
        try:
            # Load processed data
            processed_path = Path('datasets') / dataset_name / 'processed' / 'processed_data.csv'
            df = pd.read_csv(processed_path)
            
            features_info = {
                'dataset_name': dataset_name,
                'extraction_date': datetime.now().isoformat(),
                'feature_types': feature_types,
                'total_samples': len(df)
            }
            
            # Load previous analysis results
            previous_results = self._load_previous_analysis_results(dataset_name)
            
            # Extract different types of features
            all_features = []
            feature_names = []
            
            # First, integrate features from previous analysis results
            if previous_results:
                self.logger.info("🔄 Integrating features from previous analysis steps...")
                prev_features, prev_names, prev_categories = self._extract_features_from_previous_results(previous_results, len(df))
                all_features.extend(prev_features)
                feature_names.extend(prev_names)
                features_info['previous_analysis_features'] = len(prev_names)
            
            if 'text' in feature_types:
                text_features, text_names = self._extract_text_features(df, dataset_name)
                all_features.append(text_features)
                feature_names.extend(text_names)
                features_info['text_features'] = len(text_names)
            
            if 'behavioral' in feature_types:
                behavioral_features, behavioral_names = self._extract_behavioral_features(df)
                all_features.append(behavioral_features)
                feature_names.extend(behavioral_names)
                features_info['behavioral_features'] = len(behavioral_names)
            
            if 'network' in feature_types:
                network_features, network_names = self._extract_network_features(df)
                all_features.append(network_features)
                feature_names.extend(network_names)
                features_info['network_features'] = len(network_names)
            
            if 'sentiment' in feature_types:
                sentiment_features, sentiment_names = self._extract_sentiment_features(df)
                all_features.append(sentiment_features)
                feature_names.extend(sentiment_names)
                features_info['sentiment_features'] = len(sentiment_names)
            
            if 'theoretical' in feature_types:
                theoretical_features, theoretical_names = self._extract_theoretical_features(df)
                all_features.append(theoretical_features)
                feature_names.extend(theoretical_names)
                features_info['theoretical_features'] = len(theoretical_names)
            
            if 'language' in feature_types:
                language_features, language_names = self._extract_language_features(df, dataset_name)
                all_features.append(language_features)
                feature_names.extend(language_names)
                features_info['language_features'] = len(language_names)
            
            if 'embeddings' in feature_types:
                embedding_features, embedding_names = self._extract_embedding_features(df, dataset_name)
                all_features.append(embedding_features)
                feature_names.extend(embedding_names)
                features_info['embedding_features'] = len(embedding_names)
            
            if 'dataprocessor' in feature_types:
                processor_features, processor_names = self._extract_data_processor_features(df)
                all_features.append(processor_features)
                feature_names.extend(processor_names)
                features_info['data_processor_features'] = len(processor_names)
            
            if 'zeroshot' in feature_types:
                # Zero-shot features will be handled by the comprehensive extraction
                # This is a placeholder for custom feature selection
                self.logger.info("Zero-shot features requested - will be handled by comprehensive extraction")
                features_info['zeroshot_features'] = 0
            
            # Combine all features
            if all_features:
                X = np.hstack(all_features)
            else:
                # Use basic features if no specific types selected
                X, feature_names = self._extract_basic_features(df)
            
            y = df['LABEL'].values if 'LABEL' in df.columns else np.zeros(len(df))
            
            # Ensure dataset directory structure exists
            self.file_manager.create_dataset_directory(dataset_name)
            
            # Save features
            features_dir = Path('datasets') / dataset_name / 'features'
            
            # Save combined features (backward compatibility)
            np.save(features_dir / 'X_features.npy', X)
            np.save(features_dir / 'y_labels.npy', y)
            
            # Save individual feature components for modular loading
            self._save_individual_feature_components(df, dataset_name, feature_types, features_dir)
            
            # Save feature names and info
            with open(features_dir / 'feature_names.txt', 'w', encoding='utf-8') as f:
                for name in feature_names:
                    f.write(f"{name}\n")
            
            features_info['total_features'] = X.shape[1]
            features_info['feature_names'] = feature_names
            
            # Save features info
            self.file_manager.save_results(dataset_name, features_info, 'features_info')
            
            # Update dataset info
            self.file_manager.update_dataset_info(dataset_name, {
                'status': 'features_extracted',
                'features_extracted': True,
                'total_features': X.shape[1]
            })
            
            self.logger.info(f"Features extracted successfully. Shape: {X.shape}")
            return features_info
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            raise
    
    def _extract_text_features(self, df, dataset_name):
        """Extract TF-IDF text features using exact column name matching."""
        self.logger.info("Extracting text features with exact column matching...")
        
        # Use exact column matching to find text content
        text_content = self._safe_column_access(df, 'TWEET_CONTENT', '')
        texts = text_content.fillna('')
        
        # Filter out empty texts
        non_empty_texts = [text for text in texts if text.strip()]
        
        if not non_empty_texts:
            self.logger.warning("No non-empty texts found, creating dummy text features")
            # Create dummy features
            tfidf_features = np.zeros((len(texts), 10))
            feature_names = [f"tfidf_dummy_{i}" for i in range(10)]
            return tfidf_features, feature_names
        
        # Create TF-IDF vectorizer with more lenient parameters
        vectorizer = TfidfVectorizer(
            max_features=min(1000, len(non_empty_texts)),
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # More lenient
            max_df=0.95,
            token_pattern=r'\b\w+\b'  # More inclusive token pattern
        )
        
        try:
            # Fit and transform
            tfidf_features = vectorizer.fit_transform(texts).toarray()
            
            # Save vectorizer
            vectorizer_path = Path('datasets') / dataset_name / 'features' / 'tfidf_vectorizer.joblib'
            joblib.dump(vectorizer, vectorizer_path)
            
            # Get feature names
            feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
            
            return tfidf_features, feature_names
            
        except ValueError as e:
            if "empty vocabulary" in str(e):
                self.logger.warning("Empty vocabulary detected, creating basic text features")
                # Create basic text features instead
                text_lengths = np.array([len(text) for text in texts]).reshape(-1, 1)
                word_counts = np.array([len(text.split()) for text in texts]).reshape(-1, 1)
                char_counts = np.array([len(text.replace(' ', '')) for text in texts]).reshape(-1, 1)
                
                basic_features = np.hstack([text_lengths, word_counts, char_counts])
                feature_names = ['text_length', 'word_count', 'char_count']
                
                return basic_features, feature_names
            else:
                raise
    
    def _extract_behavioral_features(self, df):
        """Extract behavioral features using exact column name matching."""
        self.logger.info("Extracting behavioral features with exact column matching...")
        self.logger.info(f"Available columns in dataset: {list(df.columns)}")
        
        features = []
        feature_names = []
        
        # Account age and activity features
        if 'ACCOUNT_AGE_DAYS' in df.columns:
            account_age = self._safe_column_access(df, 'ACCOUNT_AGE_DAYS', 0)
            features.append(account_age.values.reshape(-1, 1))
            feature_names.append('account_age_days')
        
        if 'TWEETS_PER_DAY' in df.columns:
            tweets_per_day = self._safe_column_access(df, 'TWEETS_PER_DAY', 0)
            features.append(tweets_per_day.values.reshape(-1, 1))
            feature_names.append('tweets_per_day')
        
        # Follower-related features using exact column matching
        followers_count = self._safe_column_access(df, 'FOLLOWERS_COUNT', 0)
        features.append(followers_count.values.reshape(-1, 1))
        feature_names.append('followers_count')
        
        following_count = self._safe_column_access(df, 'FOLLOWING_COUNT', 0)
        features.append(following_count.values.reshape(-1, 1))
        feature_names.append('following_count')
        
        # Calculate followers/following ratio
        followers_following_ratio = self._safe_column_access(df, 'FOLLOWERS_FOLLOWING_RATIO', 0)
        if followers_following_ratio.sum() == 0:  # If ratio column doesn't exist, calculate it
            # Avoid division by zero
            ratio = followers_count / (following_count + 1)
            features.append(ratio.values.reshape(-1, 1))
        else:
            features.append(followers_following_ratio.values.reshape(-1, 1))
        feature_names.append('followers_following_ratio')
        
        # Tweet-related features
        tweet_count = self._safe_column_access(df, 'TWEET_COUNT', 0)
        features.append(tweet_count.values.reshape(-1, 1))
        feature_names.append('tweet_count')
        
        listed_count = self._safe_column_access(df, 'LISTED_COUNT', 0)
        features.append(listed_count.values.reshape(-1, 1))
        feature_names.append('listed_count')
        
        # User engagement metrics
        favourites_count = self._safe_column_access(df, 'FAVOURITES_COUNT', 0)
        features.append(favourites_count.values.reshape(-1, 1))
        feature_names.append('favourites_count')
        

        media_count = self._safe_column_access(df, 'MEDIA_COUNT', 0)
        features.append(media_count.values.reshape(-1, 1))
        feature_names.append('media_count')
        
        # Verification status indicators
        is_verified = self._safe_column_access(df, 'VERIFIED', False)
        features.append(is_verified.astype(int).values.reshape(-1, 1))
        feature_names.append('is_verified')
        
        is_blue_verified = self._safe_column_access(df, 'IS_BLUE_VERIFIED', False)
        features.append(is_blue_verified.astype(int).values.reshape(-1, 1))
        feature_names.append('is_blue_verified')
        
        # Tweet engagement metrics
        retweet_count = self._safe_column_access(df, 'RETWEET_COUNT', 0)
        features.append(retweet_count.values.reshape(-1, 1))
        feature_names.append('retweet_count')
        
        favorite_count = self._safe_column_access(df, 'FAVORITE_COUNT', 0)
        features.append(favorite_count.values.reshape(-1, 1))
        feature_names.append('favorite_count')
        
        reply_count = self._safe_column_access(df, 'REPLY_COUNT', 0)
        features.append(reply_count.values.reshape(-1, 1))
        feature_names.append('reply_count')
        
        quote_count = self._safe_column_access(df, 'QUOTE_COUNT', 0)
        features.append(quote_count.values.reshape(-1, 1))
        feature_names.append('quote_count')
        
        impression_count = self._safe_column_access(df, 'IMPRESSION_COUNT', 0)
        features.append(impression_count.values.reshape(-1, 1))
        feature_names.append('impression_count')
        
        # Derived engagement metrics
        if 'TOTAL_ENGAGEMENT' in df.columns:
            total_engagement = self._safe_column_access(df, 'TOTAL_ENGAGEMENT', 0)
            features.append(total_engagement.values.reshape(-1, 1))
            feature_names.append('total_engagement')
        
        if 'ENGAGEMENT_RATE' in df.columns:
            engagement_rate = self._safe_column_access(df, 'ENGAGEMENT_RATE', 0)
            features.append(engagement_rate.values.reshape(-1, 1))
            feature_names.append('engagement_rate')
        
        # Pre-computed influence and engagement scores
        tweet_engagement_score = self._safe_column_access(df, 'TWEET_ENGAGEMENT_SCORE', 0)
        features.append(tweet_engagement_score.values.reshape(-1, 1))
        feature_names.append('tweet_engagement_score')
        
        user_influence_score = self._safe_column_access(df, 'USER_INFLUENCE_SCORE', 0)
        features.append(user_influence_score.values.reshape(-1, 1))
        feature_names.append('user_influence_score')
        
        verification_score = self._safe_column_access(df, 'VERIFICATION_SCORE', 0)
        features.append(verification_score.values.reshape(-1, 1))
        feature_names.append('verification_score')
        
        # Text-based behavioral features
        text_length = self._safe_column_access(df, 'TEXT_LENGTH', 0)
        features.append(text_length.values.reshape(-1, 1))
        feature_names.append('text_length')
        
        word_count = self._safe_column_access(df, 'WORD_COUNT', 0)
        features.append(word_count.values.reshape(-1, 1))
        feature_names.append('word_count')
        
        readability_score = self._safe_column_access(df, 'READABILITY_SCORE', 0)
        features.append(readability_score.values.reshape(-1, 1))
        feature_names.append('readability_score')
        
        # Content type classification features
        relationship_features = ['IS_RETWEET', 'IS_QUOTE', 'IS_REPLY', 'IS_MENTIONSINRETWEET']
        for rel_feature in relationship_features:
            if rel_feature in df.columns:
                rel_value = self._safe_column_access(df, rel_feature, 0)
                features.append(rel_value.values.reshape(-1, 1))
                feature_names.append(rel_feature.lower())
        
        # Content metadata features
        if 'HAS_MEDIA' in df.columns:
            has_media = self._safe_column_access(df, 'HAS_MEDIA', 0)
            features.append(has_media.values.reshape(-1, 1))
            feature_names.append('has_media')
        
        if 'HAS_URLS' in df.columns:
            has_urls = self._safe_column_access(df, 'HAS_URLS', 0)
            features.append(has_urls.values.reshape(-1, 1))
            feature_names.append('has_urls')
        
        if 'HASHTAG_COUNT' in df.columns:
            hashtag_count = self._safe_column_access(df, 'HASHTAG_COUNT', 0)
            features.append(hashtag_count.values.reshape(-1, 1))
            feature_names.append('hashtag_count')
        
        if 'MENTION_COUNT' in df.columns:
            mention_count = self._safe_column_access(df, 'MENTION_COUNT', 0)
            features.append(mention_count.values.reshape(-1, 1))
            feature_names.append('mention_count')
        
        # Combine all features - ensure consistent dtypes
        if features:
            # Convert all features to float32 to avoid object arrays
            processed_features = []
            for feature in features:
                processed_features.append(feature.astype(np.float32))
            behavioral_features = np.hstack(processed_features)
        else:
            # Fallback if no features
            behavioral_features = np.zeros((len(df), 1), dtype=np.float32)
            feature_names = ['behavioral_dummy']
        
        self.logger.info(f"Extracted {len(feature_names)} behavioral features using exact column matching")
        self.logger.info(f"Behavioral feature names: {feature_names}")
        self.logger.info(f"Behavioral features shape: {behavioral_features.shape}")
        return behavioral_features, feature_names
    
    def _extract_network_features(self, df):
        """Extract network-based features using exact column name matching."""
        self.logger.info("Extracting network features with exact column matching...")
        
        features = []
        feature_names = []
        
        # Network centrality measures
        network_centrality_features = [
            'DEGREE', 'IN_DEGREE', 'OUT_DEGREE', 'BETWEENNESS_CENTRALITY', 
            'CLOSENESS_CENTRALITY', 'EIGENVECTOR_CENTRALITY', 'PAGERANK', 'CLUSTERING_COEFFICIENT'
        ]
        
        for feature_name in network_centrality_features:
            if feature_name in df.columns:
                feature_values = self._safe_column_access(df, feature_name, 0)
                features.append(feature_values.values.reshape(-1, 1))
                feature_names.append(feature_name.lower())
                
                # Include normalized versions if available
                normalized_name = f'{feature_name}_NORMALIZED'
                if normalized_name in df.columns:
                    normalized_values = self._safe_column_access(df, normalized_name, 0)
                    features.append(normalized_values.values.reshape(-1, 1))
                    feature_names.append(f'{feature_name.lower()}_normalized')
        
        # Extract tweet content for analysis
        tweet_content = self._safe_column_access(df, 'TWEET', '')
        if tweet_content.fillna('').str.len().sum() == 0:
            # Fallback to legacy column name
            tweet_content = self._safe_column_access(df, 'TWEET_CONTENT', '')
        
        # Social interaction indicators
        # Mention count analysis
        if 'MENTION_COUNT' in df.columns:
            mention_counts = self._safe_column_access(df, 'MENTION_COUNT', 0)
        else:
            mention_counts = tweet_content.fillna('').str.count('@')
        features.append(mention_counts.values.reshape(-1, 1))
        feature_names.append('mention_count')
        
        # Hashtag usage analysis
        if 'HASHTAG_COUNT' in df.columns:
            hashtag_counts = self._safe_column_access(df, 'HASHTAG_COUNT', 0)
        else:
            hashtag_counts = tweet_content.fillna('').str.count('#')
        features.append(hashtag_counts.values.reshape(-1, 1))
        feature_names.append('hashtag_count')
        
        # URL sharing analysis
        if 'HAS_URLS' in df.columns:
            url_counts = self._safe_column_access(df, 'HAS_URLS', 0)
        else:
            url_counts = tweet_content.fillna('').str.count('http')
        features.append(url_counts.values.reshape(-1, 1))
        feature_names.append('url_count')
        
        # Media presence
        if 'HAS_MEDIA' in df.columns:
            media_presence = self._safe_column_access(df, 'HAS_MEDIA', 0)
        else:
            # Check if MEDIA column has content
            media_col = self._safe_column_access(df, 'MEDIA', '')
            media_presence = (media_col.fillna('').str.len() > 0).astype(int)
        features.append(media_presence.values.reshape(-1, 1))
        feature_names.append('has_media')
        
        # Network interaction patterns
        if 'RELATIONSHIP' in df.columns:
            # Create binary features for different relationship types
            relationship_types = df['RELATIONSHIP'].unique()
            for rel_type in relationship_types:
                if pd.notna(rel_type):
                    rel_feature = (df['RELATIONSHIP'] == rel_type).astype(int)
                    features.append(rel_feature.values.reshape(-1, 1))
                    feature_names.append(f'relationship_{str(rel_type).lower()}')
        
        # Self-reference detection
        if 'AUTHOR_VERTEX' in df.columns and 'TARGET_VERTEX' in df.columns:
            # Self-reference indicator (author mentions themselves)
            self_reference = (df['AUTHOR_VERTEX'] == df['TARGET_VERTEX']).astype(int)
            features.append(self_reference.values.reshape(-1, 1))
            feature_names.append('is_self_reference')
        
        # Network amplification metrics
        engagement_features = ['RETWEET_COUNT', 'FAVORITE_COUNT', 'REPLY_COUNT', 'QUOTE_COUNT', 'IMPRESSION_COUNT']
        for eng_feature in engagement_features:
            if eng_feature in df.columns:
                eng_values = self._safe_column_access(df, eng_feature, 0)
                features.append(eng_values.values.reshape(-1, 1))
                feature_names.append(eng_feature.lower())
        
        # Composite engagement metrics
        if 'TOTAL_ENGAGEMENT' in df.columns:
            total_engagement = self._safe_column_access(df, 'TOTAL_ENGAGEMENT', 0)
            features.append(total_engagement.values.reshape(-1, 1))
            feature_names.append('total_engagement')
        
        if 'ENGAGEMENT_RATE' in df.columns:
            engagement_rate = self._safe_column_access(df, 'ENGAGEMENT_RATE', 0)
            features.append(engagement_rate.values.reshape(-1, 1))
            feature_names.append('engagement_rate')
        
        # User influence metrics
        if 'USER_INFLUENCE_SCORE' in df.columns:
            influence_score = self._safe_column_access(df, 'USER_INFLUENCE_SCORE', 0)
            features.append(influence_score.values.reshape(-1, 1))
            feature_names.append('user_influence_score')
        
        # Content engagement scoring
        if 'TWEET_ENGAGEMENT_SCORE' in df.columns:
            tweet_engagement = self._safe_column_access(df, 'TWEET_ENGAGEMENT_SCORE', 0)
            features.append(tweet_engagement.values.reshape(-1, 1))
            feature_names.append('tweet_engagement_score')
        
        # Combine all network features - ensure consistent dtypes
        if features:
            # Convert all features to float32 to avoid object arrays
            processed_features = []
            for feature in features:
                processed_features.append(feature.astype(np.float32))
            network_features = np.hstack(processed_features)
        else:
            # Fallback if no features
            network_features = np.zeros((len(df), 1), dtype=np.float32)
            feature_names = ['network_dummy']
        
        self.logger.info(f"Extracted {len(feature_names)} network features using exact column matching")
        return network_features, feature_names
    
    def _extract_sentiment_features(self, df):
        """Extract sentiment features using exact column name matching."""
        self.logger.info("Extracting sentiment features with exact column matching...")
        
        features = []
        feature_names = []
        
        sentiment_cols = ['SENTIMENT_POSITIVE', 'SENTIMENT_NEGATIVE', 'SENTIMENT_NEUTRAL', 'SENTIMENT_COMPOUND']
        
        for col in sentiment_cols:
            sentiment_data = self._safe_column_access(df, col, 0.0)
            features.append(sentiment_data.values.reshape(-1, 1))
            feature_names.append(col.lower())
        
        # Combine all sentiment features
        sentiment_features = np.hstack(features)
        
        self.logger.info(f"Extracted {len(feature_names)} sentiment features using exact column matching")
        return sentiment_features, feature_names
    
    def _extract_data_processor_features(self, df):
        """Extract features generated by the data processor during data processing."""
        self.logger.info("Extracting data processor features...")
        
        features = []
        feature_names = []
        
        # Data processor features (generated during data processing)
        data_processor_features = [
            'TEXT_LENGTH',
            'WORD_COUNT', 
            'READABILITY_SCORE',
            'SENTIMENT_POSITIVE',
            'SENTIMENT_NEGATIVE',
            'SENTIMENT_NEUTRAL',
            'SENTIMENT_COMPOUND',
            'FOLLOWERS_FOLLOWING_RATIO',
            'IS_VERIFIED'
        ]
        
        for feature_name in data_processor_features:
            if feature_name in df.columns:
                feature_data = pd.to_numeric(df[feature_name], errors='coerce').fillna(0)
                features.append(feature_data.values.reshape(-1, 1))
                feature_names.append(f'dp_{feature_name.lower()}')
        
        if features:
            processor_features = np.hstack(features)
            self.logger.info(f"Extracted {len(feature_names)} data processor features")
        else:
            # Create minimal features if none found
            processor_features = np.zeros((len(df), 1))
            feature_names = ['dp_no_features']
            self.logger.warning("No data processor features found, using placeholder")
        
        return processor_features, feature_names
    
    def _extract_basic_features(self, df):
        """Extract basic features when no specific types are selected."""
        self.logger.info("Extracting basic features...")
        
        features = []
        feature_names = []
        
        # Include all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['LABEL']  # Don't include target variable
        
        for col in numeric_cols:
            if col not in exclude_cols:
                features.append(df[col].values.reshape(-1, 1))
                feature_names.append(col.lower())
        
        if features:
            basic_features = np.hstack(features)
        else:
            # Create minimal features
            basic_features = np.ones((len(df), 1))
            feature_names = ['constant_feature']
        
        return basic_features, feature_names
    
    def load_features(self, dataset_name, feature_combo=None):
        """Load previously extracted features, optionally filtered by feature combination."""
        try:
            if feature_combo is None:
                # Load all features (backward compatibility)
                return self._load_all_features(dataset_name)
            else:
                # Load specific feature combination
                return self._load_feature_combination(dataset_name, feature_combo)
                
        except Exception as e:
            self.logger.error(f"Error loading features: {e}")
            return None, None, None
    
    def _load_all_features(self, dataset_name):
        """Load all previously extracted features."""
        try:
            features_dir = Path('datasets') / dataset_name / 'features'
            
            # Try loading without pickle first (safer)
            try:
                X = np.load(features_dir / 'X_features.npy')
                y = np.load(features_dir / 'y_labels.npy')
            except ValueError as ve:
                if "allow_pickle" in str(ve):
                    # Fallback to allow_pickle=True for object arrays
                    self.logger.warning(f"Loading features with allow_pickle=True due to object arrays")
                    X = np.load(features_dir / 'X_features.npy', allow_pickle=True)
                    y = np.load(features_dir / 'y_labels.npy', allow_pickle=True)
                    
                    # Convert to numeric if possible
                    if X.dtype == object:
                        try:
                            X = X.astype(np.float32)
                        except (ValueError, TypeError):
                            self.logger.error("Cannot convert X features to numeric")
                            return None, None, None
                    
                    if y.dtype == object:
                        try:
                            y = y.astype(np.int32)
                        except (ValueError, TypeError):
                            self.logger.error("Cannot convert y labels to numeric")
                            return None, None, None
                else:
                    raise ve
            
            # Load feature names
            feature_names = []
            names_file = features_dir / 'feature_names.txt'
            if names_file.exists():
                with open(names_file, 'r', encoding='utf-8') as f:
                    feature_names = [line.strip() for line in f.readlines()]
            
            # Validate for NaN values
            if np.isnan(X).any():
                self.logger.warning("NaN values detected in features X. Cleaning...")
                # Remove rows with NaN values
                nan_mask = ~np.isnan(X).any(axis=1)
                X = X[nan_mask]
                y = y[nan_mask]
                removed_samples = (~nan_mask).sum()
                self.logger.info(f"Removed {removed_samples} samples with NaN values")
            
            if np.isnan(y).any():
                self.logger.warning("NaN values detected in labels y. Cleaning...")
                # Remove samples with NaN labels
                nan_mask = ~np.isnan(y)
                X = X[nan_mask]
                y = y[nan_mask]
                removed_samples = (~nan_mask).sum()
                self.logger.info(f"Removed {removed_samples} samples with NaN labels")
            
            # Final validation
            if np.isnan(X).any() or np.isnan(y).any():
                self.logger.error("NaN values still present after cleaning. Cannot proceed with training.")
                return None, None, None
            
            self.logger.info(f"Features loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, feature_names
            
        except Exception as e:
            self.logger.error(f"Error loading all features: {e}")
            return None, None, None
    
    def _load_feature_combination(self, dataset_name, feature_combo):
        """Load specific feature combination by combining individual feature types."""
        try:
            self.logger.info(f"Loading feature combination: {feature_combo}")
            features_dir = Path('datasets') / dataset_name / 'features'
            
            # Load labels (same for all combinations)
            y = np.load(features_dir / 'y_labels.npy')
            if y.dtype == object:
                y = y.astype(np.int32)
            
            # Check if individual components exist, if not, try to use combined features
            if not self._individual_components_exist(features_dir):
                self.logger.info("Individual components not found, trying to use combined features...")
                return self._load_combined_features_as_fallback(dataset_name, feature_combo, y)
            
            # Define feature combination mappings
            feature_mappings = {
                'base_model': ['text', 'lda'],  # TF-IDF + LDA baseline
                'rat_framework': ['text', 'lda', 'rat'],  # RAT features only
                'rct_framework': ['text', 'lda', 'rct'],  # RCT features only
                'ugt_framework': ['text', 'lda', 'ugt'],  # UGT features only
                'framework_embeddings_all': ['text', 'lda', 'rat', 'rct', 'ugt'],  # All frameworks
                'transformer_embeddings': ['text', 'lda', 'transformer'],  # BERT/RoBERTa
                'behavioral_features': ['text', 'lda', 'behavioral'],  # User behavior patterns
                'complete_model': ['text', 'lda', 'rat', 'rct', 'ugt', 'transformer', 'behavioral', 'sentiment']  # Everything
            }
            
            if feature_combo not in feature_mappings:
                self.logger.error(f"Unknown feature combination: {feature_combo}")
                return None, None, None
            
            # Load individual feature components
            feature_components = []
            feature_names = []
            required_features = feature_mappings[feature_combo]
            
            for feature_type in required_features:
                component_features, component_names = self._load_feature_component(features_dir, feature_type)
                if component_features is not None:
                    feature_components.append(component_features)
                    feature_names.extend(component_names)
                else:
                    self.logger.warning(f"Could not load {feature_type} features, skipping...")
            
            if not feature_components:
                self.logger.warning(f"No individual components loaded for {feature_combo}, using combined features as fallback")
                return self._load_combined_features_as_fallback(dataset_name, feature_combo, y)
            
            # Combine all feature components
            X = np.hstack(feature_components)
            
            # Ensure same number of samples
            if len(y) != X.shape[0]:
                min_samples = min(len(y), X.shape[0])
                self.logger.warning(f"Sample count mismatch. Truncating to {min_samples} samples")
                X = X[:min_samples]
                y = y[:min_samples]
            
            # Validate for NaN values
            if np.isnan(X).any():
                self.logger.warning("NaN values detected in combined features. Cleaning...")
                nan_mask = ~np.isnan(X).any(axis=1)
                X = X[nan_mask]
                y = y[nan_mask]
                removed_samples = (~nan_mask).sum()
                self.logger.info(f"Removed {removed_samples} samples with NaN values")
            
            if np.isnan(y).any():
                self.logger.warning("NaN values detected in labels. Cleaning...")
                nan_mask = ~np.isnan(y)
                X = X[nan_mask]
                y = y[nan_mask]
                removed_samples = (~nan_mask).sum()
                self.logger.info(f"Removed {removed_samples} samples with NaN labels")
            
            # Final validation
            if np.isnan(X).any() or np.isnan(y).any():
                self.logger.error("NaN values still present after cleaning. Cannot proceed.")
                return None, None, None
            
            self.logger.info(f"Feature combination '{feature_combo}' loaded: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, feature_names
            
        except Exception as e:
            self.logger.error(f"Error loading feature combination {feature_combo}: {e}")
            return None, None, None
    
    def _load_feature_component(self, features_dir, feature_type):
        """Load individual feature component."""
        try:
            # Check unified directory first (preferred)
            unified_dir = features_dir / 'unified'
            
            if feature_type == 'text':
                # Check unified directory first
                if unified_dir.exists():
                    unified_file = unified_dir / 'text_features.npy'
                    if unified_file.exists():
                        features = np.load(unified_file, allow_pickle=True)
                        # Handle potential shape issues by flattening and reshaping if needed
                        if features.ndim > 2:
                            self.logger.warning(f"Text features have {features.ndim} dimensions, flattening...")
                            features = features.reshape(features.shape[0], -1)
                        feature_names = [f'text_{i}' for i in range(features.shape[1])]
                        return features, feature_names
                
                # Fallback to individual file
                tfidf_file = features_dir / 'tfidf_features.npy'
                if tfidf_file.exists():
                    features = np.load(tfidf_file)
                    feature_names = [f'tfidf_{i}' for i in range(features.shape[1])]
                    return features, feature_names
                    
            elif feature_type == 'lda':
                # Check unified directory first
                if unified_dir.exists():
                    unified_file = unified_dir / 'data_processor_features.npy'
                    if unified_file.exists():
                        features = np.load(unified_file, allow_pickle=True)
                        if features.ndim > 2:
                            features = features.reshape(features.shape[0], -1)
                        feature_names = [f'lda_{i}' for i in range(features.shape[1])]
                        return features, feature_names
                
                # Fallback to individual file
                lda_file = features_dir / 'lda_features.npy'
                if lda_file.exists():
                    features = np.load(lda_file)
                    feature_names = [f'lda_topic_{i}' for i in range(features.shape[1])]
                    return features, feature_names
                    
            elif feature_type == 'rat':
                # Check unified directory first
                if unified_dir.exists():
                    unified_file = unified_dir / 'theoretical_features.npy'
                    if unified_file.exists():
                        features = np.load(unified_file, allow_pickle=True)
                        if features.ndim > 2:
                            features = features.reshape(features.shape[0], -1)
                        # Use only RAT portion (first third)
                        rat_size = features.shape[1] // 3
                        rat_features = features[:, :rat_size]
                        feature_names = [f'rat_{i}' for i in range(rat_features.shape[1])]
                        return rat_features, feature_names
                
                # Fallback to individual file
                rat_file = features_dir / 'rat_features.npy'
                if rat_file.exists():
                    features = np.load(rat_file)
                    feature_names = [f'rat_{i}' for i in range(features.shape[1])]
                    return features, feature_names
                    
            elif feature_type == 'rct':
                # Check unified directory first
                if unified_dir.exists():
                    unified_file = unified_dir / 'theoretical_features.npy'
                    if unified_file.exists():
                        features = np.load(unified_file, allow_pickle=True)
                        if features.ndim > 2:
                            features = features.reshape(features.shape[0], -1)
                        # Use only RCT portion (middle third)
                        rct_size = features.shape[1] // 3
                        rct_features = features[:, rct_size:2*rct_size]
                        feature_names = [f'rct_{i}' for i in range(rct_features.shape[1])]
                        return rct_features, feature_names
                
                # Fallback to individual file
                rct_file = features_dir / 'rct_features.npy'
                if rct_file.exists():
                    features = np.load(rct_file)
                    feature_names = [f'rct_{i}' for i in range(features.shape[1])]
                    return features, feature_names
                    
            elif feature_type == 'ugt':
                # Check unified directory first
                if unified_dir.exists():
                    unified_file = unified_dir / 'theoretical_features.npy'
                    if unified_file.exists():
                        features = np.load(unified_file, allow_pickle=True)
                        if features.ndim > 2:
                            features = features.reshape(features.shape[0], -1)
                        # Use only UGT portion (last third)
                        ugt_size = features.shape[1] // 3
                        ugt_features = features[:, 2*ugt_size:]
                        feature_names = [f'ugt_{i}' for i in range(ugt_features.shape[1])]
                        return ugt_features, feature_names
                
                # Fallback to individual file
                ugt_file = features_dir / 'ugt_features.npy'
                if ugt_file.exists():
                    features = np.load(ugt_file)
                    feature_names = [f'ugt_{i}' for i in range(features.shape[1])]
                    return features, feature_names
                    
            elif feature_type == 'transformer':
                # Check unified directory first
                if unified_dir.exists():
                    unified_file = unified_dir / 'transformer_embeddings_features.npy'
                    if unified_file.exists():
                        features = np.load(unified_file, allow_pickle=True)
                        if features.ndim > 2:
                            features = features.reshape(features.shape[0], -1)
                        feature_names = [f'transformer_{i}' for i in range(features.shape[1])]
                        return features, feature_names
                
                # Fallback to individual file or existing embeddings
                transformer_file = features_dir / 'transformer_embeddings.npy'
                if transformer_file.exists():
                    features = np.load(transformer_file)
                    feature_names = [f'transformer_{i}' for i in range(features.shape[1])]
                    return features, feature_names
                
                # Check for BERT embeddings
                bert_file = features_dir / 'bert_embeddings.npy'
                if bert_file.exists():
                    features = np.load(bert_file)
                    feature_names = [f'bert_{i}' for i in range(features.shape[1])]
                    return features, feature_names
                    
            elif feature_type == 'behavioral':
                # Check unified directory first
                if unified_dir.exists():
                    unified_file = unified_dir / 'behavioral_features.npy'
                    if unified_file.exists():
                        features = np.load(unified_file, allow_pickle=True)
                        if features.ndim > 2:
                            features = features.reshape(features.shape[0], -1)
                        feature_names = [f'behavioral_{i}' for i in range(features.shape[1])]
                        return features, feature_names
                
                # Fallback to individual file
                behavioral_file = features_dir / 'behavioral_features.npy'
                if behavioral_file.exists():
                    features = np.load(behavioral_file)
                    feature_names = [f'behavioral_{i}' for i in range(features.shape[1])]
                    return features, feature_names
                    
            elif feature_type == 'sentiment':
                # Check unified directory first
                if unified_dir.exists():
                    unified_file = unified_dir / 'sentiment_features.npy'
                    if unified_file.exists():
                        features = np.load(unified_file, allow_pickle=True)
                        if features.ndim > 2:
                            features = features.reshape(features.shape[0], -1)
                        feature_names = [f'sentiment_{i}' for i in range(features.shape[1])]
                        return features, feature_names
                
                # Fallback to individual file
                sentiment_file = features_dir / 'sentiment_features.npy'
                if sentiment_file.exists():
                    features = np.load(sentiment_file)
                    feature_names = [f'sentiment_{i}' for i in range(features.shape[1])]
                    return features, feature_names
            
            # If we get here, the feature type wasn't found
            self.logger.warning(f"Feature component '{feature_type}' not found")
            return None, []
            
        except Exception as e:
            self.logger.error(f"Error loading feature component {feature_type}: {e}")
            return None, []
    
    def _individual_components_exist(self, features_dir):
        """Check if individual feature components exist."""
        # Check for unified directory first (preferred)
        unified_dir = features_dir / 'unified'
        if unified_dir.exists():
            unified_files = ['text_features.npy', 'data_processor_features.npy']
            if all((unified_dir / file).exists() for file in unified_files):
                return True
        
        # Check for individual component files
        required_files = ['tfidf_features.npy', 'lda_features.npy']
        return all((features_dir / file).exists() for file in required_files)
    
    def _create_individual_components_from_combined(self, dataset_name, features_dir):
        """Create individual feature components from combined features for backward compatibility."""
        try:
            self.logger.info("Creating individual components from combined features...")
            
            # Load combined features
            X_combined = np.load(features_dir / 'X_features.npy')
            
            # Load processed data to recreate individual components
            processed_file = Path('datasets') / dataset_name / 'processed' / 'processed_data.csv'
            if not processed_file.exists():
                self.logger.error("Processed data not found, cannot recreate individual components")
                return
            
            df = pd.read_csv(processed_file)
            
            # Recreate TF-IDF features
            try:
                text_content = self._safe_column_access(df, 'TWEET_CONTENT', '')
                texts = text_content.fillna('').astype(str).tolist()
                tfidf_features, _ = self._extract_text_features(df, dataset_name)
                np.save(features_dir / 'tfidf_features.npy', tfidf_features)
                self.logger.info("✅ Created TF-IDF features from combined data")
            except Exception as e:
                self.logger.warning(f"Could not recreate TF-IDF features: {e}")
                # Create dummy TF-IDF features (first portion of combined features)
                tfidf_size = min(1000, X_combined.shape[1] // 2)
                dummy_tfidf = X_combined[:, :tfidf_size]
                np.save(features_dir / 'tfidf_features.npy', dummy_tfidf)
            
            # Recreate LDA features
            try:
                text_content = self._safe_column_access(df, 'TWEET_CONTENT', '')
                texts = text_content.fillna('').astype(str).tolist()
                lda_features, _ = self.extract_lda_features(texts, dataset_name, n_topics=10)
                np.save(features_dir / 'lda_features.npy', lda_features)
                self.logger.info("✅ Created LDA features from combined data")
            except Exception as e:
                self.logger.warning(f"Could not recreate LDA features: {e}")
                # Create dummy LDA features
                dummy_lda = np.zeros((X_combined.shape[0], 10))
                np.save(features_dir / 'lda_features.npy', dummy_lda)
            
            # Create dummy framework features (if they don't exist)
            framework_features = ['rat_features.npy', 'rct_features.npy', 'ugt_features.npy']
            for framework_file in framework_features:
                if not (features_dir / framework_file).exists():
                    dummy_framework = np.zeros((X_combined.shape[0], 10))
                    np.save(features_dir / framework_file, dummy_framework)
                    self.logger.info(f"✅ Created dummy {framework_file}")
            
            # Create dummy transformer embeddings (if they don't exist)
            if not (features_dir / 'transformer_embeddings.npy').exists():
                # Check if BERT embeddings exist (different naming)
                if (features_dir / 'bert_embeddings.npy').exists():
                    bert_embeddings = np.load(features_dir / 'bert_embeddings.npy')
                    np.save(features_dir / 'transformer_embeddings.npy', bert_embeddings)
                    self.logger.info("✅ Used existing BERT embeddings as transformer embeddings")
                elif (features_dir / 'sentence_embeddings.npy').exists():
                    sentence_embeddings = np.load(features_dir / 'sentence_embeddings.npy')
                    np.save(features_dir / 'transformer_embeddings.npy', sentence_embeddings)
                    self.logger.info("✅ Used existing sentence embeddings as transformer embeddings")
                else:
                    dummy_transformer = np.zeros((X_combined.shape[0], 768))  # Standard BERT size
                    np.save(features_dir / 'transformer_embeddings.npy', dummy_transformer)
                    self.logger.info("✅ Created dummy transformer embeddings")
            
            # Create dummy behavioral and sentiment features (if they don't exist)
            other_features = ['behavioral_features.npy', 'sentiment_features.npy']
            for feature_file in other_features:
                if not (features_dir / feature_file).exists():
                    dummy_features = np.zeros((X_combined.shape[0], 5))
                    np.save(features_dir / feature_file, dummy_features)
                    self.logger.info(f"✅ Created dummy {feature_file}")
            
            self.logger.info("Individual components created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating individual components: {e}")
    
    def _load_combined_features_as_fallback(self, dataset_name, feature_combo, y):
        """Load combined features as fallback when individual components are not available."""
        try:
            features_dir = Path('datasets') / dataset_name / 'features'
            
            # Load combined features directly without any reshaping
            X_combined_file = features_dir / 'X_features.npy'
            if not X_combined_file.exists():
                self.logger.error("Combined features file not found")
                return None, None, None
                
            X_combined = np.load(X_combined_file, allow_pickle=True)
            self.logger.info(f"Loaded combined features with shape: {X_combined.shape}")
            
            # Load feature names if available
            feature_names = []
            feature_names_file = features_dir / 'feature_names.txt'
            if feature_names_file.exists():
                try:
                    with open(feature_names_file, 'r', encoding='utf-8') as f:
                        feature_names = [line.strip() for line in f.readlines() if line.strip()]
                except Exception as e:
                    self.logger.warning(f"Could not load feature names: {e}")
                    feature_names = [f'feature_{i}' for i in range(X_combined.shape[1])]
            else:
                # Generate generic feature names
                feature_names = [f'feature_{i}' for i in range(X_combined.shape[1])]
            
            # For different feature combinations, we'll use the full combined features
            # This is a fallback approach - ideally individual components should be available
            self.logger.info(f"Using combined features as fallback for {feature_combo}")
            
            # Ensure same number of samples
            if len(y) != X_combined.shape[0]:
                min_samples = min(len(y), X_combined.shape[0])
                self.logger.warning(f"Sample count mismatch. Truncating to {min_samples} samples")
                X_combined = X_combined[:min_samples]
                y = y[:min_samples]
            
            # Basic validation - check for obvious issues
            if X_combined.size == 0:
                self.logger.error("Combined features array is empty")
                return None, None, None
                
            if len(y) == 0:
                self.logger.error("Labels array is empty")
                return None, None, None
            
            # Convert to float32 to ensure compatibility
            if X_combined.dtype != np.float32:
                X_combined = X_combined.astype(np.float32)
            
            if y.dtype not in [np.int32, np.int64]:
                y = y.astype(np.int32)
            
            self.logger.info(f"Fallback features ready: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
            return X_combined, y, feature_names
            
        except Exception as e:
            self.logger.error(f"Error loading combined features as fallback: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None, None
    
    def _save_individual_feature_components(self, df, dataset_name, feature_types, features_dir):
        """Save individual feature components for modular loading."""
        try:
            self.logger.info("Saving individual feature components...")
            
            # Always save TF-IDF features (core component)
            try:
                text_features, _ = self._extract_text_features(df, dataset_name)
                np.save(features_dir / 'tfidf_features.npy', text_features)
                self.logger.info("✅ Saved TF-IDF features")
            except Exception as e:
                self.logger.warning(f"Could not save TF-IDF features: {e}")
                # Create dummy TF-IDF features
                dummy_tfidf = np.zeros((len(df), 100))
                np.save(features_dir / 'tfidf_features.npy', dummy_tfidf)
            
            # Always save LDA features (core component)
            try:
                # Extract text content for LDA
                text_content = self._safe_column_access(df, 'TWEET_CONTENT', '')
                texts = text_content.fillna('').astype(str).tolist()
                lda_features, _ = self.extract_lda_features(texts, dataset_name, n_topics=10)
                np.save(features_dir / 'lda_features.npy', lda_features)
                self.logger.info("✅ Saved LDA features")
            except Exception as e:
                self.logger.warning(f"Could not save LDA features: {e}")
                # Create dummy LDA features
                dummy_lda = np.zeros((len(df), 10))
                np.save(features_dir / 'lda_features.npy', dummy_lda)
            
            # Save theoretical framework features (always save, even if dummy)
            try:
                if 'theoretical' in feature_types:
                    # RAT features - pass DataFrame directly
                    rat_features_df = self.theoretical_frameworks.extract_rat_features(df)
                    if rat_features_df is not None and not rat_features_df.empty:
                        rat_features = rat_features_df.values
                        np.save(features_dir / 'rat_features.npy', rat_features)
                        self.logger.info(f"✅ Saved RAT features: {rat_features.shape}")
                    else:
                        dummy_rat = np.zeros((len(df), 6))  # 6 RAT features
                        np.save(features_dir / 'rat_features.npy', dummy_rat)
                        self.logger.info("✅ Saved dummy RAT features")
                    
                    # RCT features - pass DataFrame directly
                    rct_features_df = self.theoretical_frameworks.extract_rct_features(df)
                    if rct_features_df is not None and not rct_features_df.empty:
                        rct_features = rct_features_df.values
                        np.save(features_dir / 'rct_features.npy', rct_features)
                        self.logger.info(f"✅ Saved RCT features: {rct_features.shape}")
                    else:
                        dummy_rct = np.zeros((len(df), 7))  # 7 RCT features
                        np.save(features_dir / 'rct_features.npy', dummy_rct)
                        self.logger.info("✅ Saved dummy RCT features")
                    
                    # UGT features - check if method exists
                    try:
                        ugt_features_df = self.theoretical_frameworks.extract_content_gratification_features(df)
                        if ugt_features_df is not None and not ugt_features_df.empty:
                            ugt_features = ugt_features_df.values
                            np.save(features_dir / 'ugt_features.npy', ugt_features)
                            self.logger.info(f"✅ Saved UGT features: {ugt_features.shape}")
                        else:
                            dummy_ugt = np.zeros((len(df), 10))  # 10 UGT features
                            np.save(features_dir / 'ugt_features.npy', dummy_ugt)
                            self.logger.info("✅ Saved dummy UGT features")
                    except AttributeError:
                        # Fallback if extract_ugt_features doesn't exist
                        dummy_ugt = np.zeros((len(df), 10))
                        np.save(features_dir / 'ugt_features.npy', dummy_ugt)
                        self.logger.info("✅ Saved dummy UGT features (method not found)")
                else:
                    # Create dummy framework features if theoretical not in feature_types
                    dummy_features = np.zeros((len(df), 10))
                    np.save(features_dir / 'rat_features.npy', dummy_features)
                    np.save(features_dir / 'rct_features.npy', dummy_features)
                    np.save(features_dir / 'ugt_features.npy', dummy_features)
                    self.logger.info("✅ Saved dummy framework features")
                        
            except Exception as e:
                self.logger.warning(f"Could not save framework features: {e}")
                # Create dummy framework features
                dummy_features = np.zeros((len(df), 10))
                np.save(features_dir / 'rat_features.npy', dummy_features)
                np.save(features_dir / 'rct_features.npy', dummy_features)
                np.save(features_dir / 'ugt_features.npy', dummy_features)
            
            # Save transformer embeddings (always save, even if dummy)
            try:
                if 'embeddings' in feature_types or 'language' in feature_types:
                    embedding_features, _ = self._extract_embedding_features(df, dataset_name)
                    np.save(features_dir / 'transformer_embeddings.npy', embedding_features)
                    self.logger.info("✅ Saved transformer embeddings")
                else:
                    # Create dummy transformer features
                    dummy_transformer = np.zeros((len(df), 768))  # Standard BERT size
                    np.save(features_dir / 'transformer_embeddings.npy', dummy_transformer)
                    self.logger.info("✅ Saved dummy transformer embeddings")
            except Exception as e:
                self.logger.warning(f"Could not save transformer embeddings: {e}")
                # Create dummy transformer features
                dummy_transformer = np.zeros((len(df), 768))  # Standard BERT size
                np.save(features_dir / 'transformer_embeddings.npy', dummy_transformer)
            
            # Save behavioral features (always save, even if dummy)
            try:
                if 'behavioral' in feature_types:
                    behavioral_features, _ = self._extract_behavioral_features(df)
                    np.save(features_dir / 'behavioral_features.npy', behavioral_features)
                    self.logger.info("✅ Saved behavioral features")
                else:
                    dummy_behavioral = np.zeros((len(df), 5))
                    np.save(features_dir / 'behavioral_features.npy', dummy_behavioral)
                    self.logger.info("✅ Saved dummy behavioral features")
            except Exception as e:
                self.logger.warning(f"Could not save behavioral features: {e}")
                dummy_behavioral = np.zeros((len(df), 5))
                np.save(features_dir / 'behavioral_features.npy', dummy_behavioral)
            
            # Save sentiment features (always save, even if dummy)
            try:
                if 'sentiment' in feature_types:
                    sentiment_features, _ = self._extract_sentiment_features(df)
                    np.save(features_dir / 'sentiment_features.npy', sentiment_features)
                    self.logger.info("✅ Saved sentiment features")
                else:
                    dummy_sentiment = np.zeros((len(df), 5))
                    np.save(features_dir / 'sentiment_features.npy', dummy_sentiment)
                    self.logger.info("✅ Saved dummy sentiment features")
            except Exception as e:
                self.logger.warning(f"Could not save sentiment features: {e}")
                dummy_sentiment = np.zeros((len(df), 5))
                np.save(features_dir / 'sentiment_features.npy', dummy_sentiment)
            
            self.logger.info("Individual feature components saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving individual feature components: {e}")
            # Continue without failing the entire process
    
    def scale_features(self, dataset_name, X_train, X_test=None):
        """Scale features using StandardScaler with NaN validation."""
        # Check for NaN values before scaling
        if np.isnan(X_train).any():
            self.logger.error("NaN values detected in X_train before scaling. Cannot proceed.")
            raise ValueError("X_train contains NaN values. Please clean the data first.")
        
        if X_test is not None and np.isnan(X_test).any():
            self.logger.error("NaN values detected in X_test before scaling. Cannot proceed.")
            raise ValueError("X_test contains NaN values. Please clean the data first.")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Validate scaled data
        if np.isnan(X_train_scaled).any():
            self.logger.error("NaN values introduced during scaling of X_train")
            raise ValueError("Scaling introduced NaN values in X_train")
        
        # Save scaler
        scaler_path = Path('datasets') / dataset_name / 'features' / 'feature_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            
            # Validate scaled test data
            if np.isnan(X_test_scaled).any():
                self.logger.error("NaN values introduced during scaling of X_test")
                raise ValueError("Scaling introduced NaN values in X_test")
            
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def _extract_theoretical_features(self, df):
        """Extract theoretical framework features (RAT, RCT, Content Gratification)."""
        self.logger.info("Extracting theoretical framework features...")
        
        try:
            # Extract all theoretical features
            theoretical_features_df = self.theoretical_frameworks.extract_all_theoretical_features(df)
            
            # Convert to numpy array
            theoretical_features = theoretical_features_df.values
            feature_names = list(theoretical_features_df.columns)
            
            self.logger.info(f"Extracted {len(feature_names)} theoretical features")
            return theoretical_features, feature_names
            
        except Exception as e:
            self.logger.error(f"Error extracting theoretical features: {e}")
            # Return dummy features
            dummy_features = np.zeros((len(df), 10))
            dummy_names = [f'theoretical_dummy_{i}' for i in range(10)]
            return dummy_features, dummy_names
    
    def _extract_language_features(self, df, dataset_name):
        """Extract language detection and transformer-based features using exact column matching."""
        self.logger.info("Extracting language and transformer features with exact column matching...")
        
        try:
            # Use exact column matching to find text content
            text_content = self._safe_column_access(df, 'TWEET_CONTENT', '')
            
            if text_content.isna().all() or (text_content == '').all():
                self.logger.warning("No text content found for language features")
                return np.zeros((len(df), 5)), ['lang_dummy_' + str(i) for i in range(5)]
            
            features = []
            feature_names = []
            
            # Process language detection for each text
            language_scores = []
            confidence_scores = []
            is_mixed_flags = []
            kiswahili_scores = []
            
            for idx, text in enumerate(text_content):
                text = str(text) if pd.notna(text) else ""
                
                if len(text.strip()) < 3:
                    # Default values for empty text
                    language_scores.append(0)  # 0 for English, 1 for Kiswahili, 2 for mixed
                    confidence_scores.append(0.0)
                    is_mixed_flags.append(0)
                    kiswahili_scores.append(0.0)
                    continue
                
                # Detect language
                lang_result = self.language_detector.detect_language(text)
                
                # Convert language to numeric
                lang_numeric = 0  # Default to English
                if lang_result['primary_language'] == 'sw':
                    lang_numeric = 1
                elif lang_result['primary_language'] == 'mixed':
                    lang_numeric = 2
                
                language_scores.append(lang_numeric)
                confidence_scores.append(lang_result['confidence'])
                is_mixed_flags.append(1 if lang_result['is_mixed'] else 0)
                kiswahili_scores.append(lang_result.get('kiswahili_score', 0.0))
            
            # Add language features
            features.extend([
                np.array(language_scores).reshape(-1, 1),
                np.array(confidence_scores).reshape(-1, 1),
                np.array(is_mixed_flags).reshape(-1, 1),
                np.array(kiswahili_scores).reshape(-1, 1)
            ])
            
            feature_names.extend([
                'language_detected',
                'language_confidence',
                'is_mixed_language',
                'kiswahili_score'
            ])
            
            # Add transformer selection features
            transformer_selections = []
            for idx, text in enumerate(text_content):
                text = str(text) if pd.notna(text) else ""
                
                if len(text.strip()) < 3:
                    transformer_selections.append(0)  # Default to monolingual
                    continue
                
                lang_result = self.language_detector.detect_language(text)
                transformer_result = self.language_detector.select_transformer(lang_result)
                
                # Convert transformer type to numeric (0 for monolingual, 1 for multilingual)
                transformer_numeric = 1 if transformer_result['model_type'] == 'multilingual' else 0
                transformer_selections.append(transformer_numeric)
            
            features.append(np.array(transformer_selections).reshape(-1, 1))
            feature_names.append('transformer_type')
            
            # Combine all language features
            if features:
                language_features = np.hstack(features)
            else:
                language_features = np.zeros((len(df), 1))
                feature_names = ['language_dummy']
            
            self.logger.info(f"Extracted {len(feature_names)} language features")
            return language_features, feature_names
            
        except Exception as e:
            self.logger.error(f"Error extracting language features: {e}")
            # Return dummy features
            dummy_features = np.zeros((len(df), 5))
            dummy_names = [f'language_dummy_{i}' for i in range(5)]
            return dummy_features, dummy_names
    
    def _extract_enhanced_sentiment_features(self, df):
        """Extract enhanced sentiment features using exact column matching."""
        self.logger.info("Extracting enhanced sentiment features with exact column matching...")
        
        try:
            # Use exact column matching to find text content
            text_content = self._safe_column_access(df, 'TWEET_CONTENT', '')
            
            if text_content.isna().all() or (text_content == '').all():
                self.logger.warning("No text content found for enhanced sentiment features")
                return np.zeros((len(df), 8)), ['sentiment_dummy_' + str(i) for i in range(8)]
            
            # Create a temporary dataframe with the text content for the sentiment analyzer
            temp_df = df.copy()
            temp_df['TEXT_CONTENT'] = text_content
            
            # Extract sentiment features using the new analyzer
            sentiment_features_df = self.sentiment_analyzer.extract_sentiment_features(temp_df, 'TEXT_CONTENT')
            
            # Convert to numpy array
            sentiment_features = sentiment_features_df.values
            feature_names = list(sentiment_features_df.columns)
            
            self.logger.info(f"Extracted {len(feature_names)} enhanced sentiment features")
            return sentiment_features, feature_names
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced sentiment features: {e}")
            # Return dummy features
            dummy_features = np.zeros((len(df), 8))
            dummy_names = [f'sentiment_dummy_{i}' for i in range(8)]
            return dummy_features, dummy_names
    
    def extract_behavioral_features(self, df):
        """Extract behavioral features from dataset."""
        return self._extract_behavioral_features(df)
    
    def extract_sentiment_features(self, df):
        """Extract sentiment features from dataset."""
        return self._extract_sentiment_features(df)
    
    def extract_comprehensive_features(self, dataset_name, embedding_config=None, zero_shot_config=None):
        """
        Extract comprehensive features including all new components.
        
        Args:
            dataset_name: Name of the dataset
            embedding_config: Configuration for embeddings (model, strategy, etc.)
            zero_shot_config: Configuration for zero-shot classification
            
        Returns:
            Dictionary with comprehensive feature information
        """
        self.logger.info(f"Extracting comprehensive features for dataset: {dataset_name}")
        
        try:
            # Load processed data
            processed_path = Path('datasets') / dataset_name / 'processed' / 'processed_data.csv'
            df = pd.read_csv(processed_path)
            
            features_info = {
                'dataset_name': dataset_name,
                'extraction_date': datetime.now().isoformat(),
                'total_samples': len(df),
                'comprehensive_extraction': True
            }
            
            # Extract all feature types
            all_features = []
            feature_names = []
            
            # 1. Data processor features (basic features generated during data processing)
            processor_features, processor_names = self._extract_data_processor_features(df)
            all_features.append(processor_features)
            feature_names.extend(processor_names)
            features_info['data_processor_features'] = len(processor_names)
            
            # 2. Text features (TF-IDF)
            text_features, text_names = self._extract_text_features(df, dataset_name)
            all_features.append(text_features)
            feature_names.extend(text_names)
            features_info['text_features'] = len(text_names)
            
            # 3. Behavioral features
            behavioral_features, behavioral_names = self._extract_behavioral_features(df)
            all_features.append(behavioral_features)
            feature_names.extend(behavioral_names)
            features_info['behavioral_features'] = len(behavioral_names)
            
            # 4. Enhanced sentiment features
            sentiment_features, sentiment_names = self._extract_enhanced_sentiment_features(df)
            all_features.append(sentiment_features)
            feature_names.extend(sentiment_names)
            features_info['sentiment_features'] = len(sentiment_names)
            
            # 5. Language features
            language_features, language_names = self._extract_language_features(df, dataset_name)
            all_features.append(language_features)
            feature_names.extend(language_names)
            features_info['language_features'] = len(language_names)
            
            # 6. Theoretical framework features
            theoretical_features, theoretical_names = self._extract_theoretical_features(df)
            all_features.append(theoretical_features)
            feature_names.extend(theoretical_names)
            features_info['theoretical_features'] = len(theoretical_names)
            
            # Generate theoretical insights from the extracted features
            try:
                from src.insights_generator import InsightsGenerator
                insights_gen = InsightsGenerator()
                
                # Convert theoretical features back to DataFrame for insights generation
                theoretical_df = pd.DataFrame(theoretical_features, columns=theoretical_names)
                theoretical_insights = insights_gen.generate_theoretical_insights(df, theoretical_df)
                features_info['theoretical_insights'] = theoretical_insights
                
                self.logger.info(f"Generated theoretical insights: {theoretical_insights}")
            except Exception as e:
                self.logger.warning(f"Could not generate theoretical insights: {e}")
                features_info['theoretical_insights'] = {
                    'rat_risk_perception': 0.0,
                    'rat_benefit_perception': 0.0,
                    'rct_coping_appraisal': 0.0,
                    'rct_threat_appraisal': 0.0
                }
            
            # 7. Network features
            network_features, network_names = self._extract_network_features(df)
            all_features.append(network_features)
            feature_names.extend(network_names)
            features_info['network_features'] = len(network_names)
            
            # 8. Embedding features (if requested)
            if embedding_config and embedding_config.get('use_embeddings', False):
                self.logger.info(f"🧠 Extracting embeddings using {embedding_config.get('embedding_model', 'default')} model...")
                embedding_features, embedding_names = self._extract_embedding_features(df, dataset_name, embedding_config)
                all_features.append(embedding_features)
                feature_names.extend(embedding_names)
                features_info['embedding_features'] = len(embedding_names)
                features_info['embedding_model'] = embedding_config.get('embedding_model')
                features_info['embedding_strategy'] = embedding_config.get('embedding_strategy')
            else:
                features_info['embedding_features'] = 0
            
            # 9. Enhanced text features (behavioral analysis + LDA)
            self.logger.info("🔍 Extracting enhanced text features with behavioral analysis and LDA...")
            enhanced_text_features, enhanced_text_names = self._extract_enhanced_text_features(df, dataset_name)
            all_features.append(enhanced_text_features)
            feature_names.extend(enhanced_text_names)
            features_info['enhanced_text_features'] = len(enhanced_text_names)
            
            # 10. Run zero-shot classification (if requested) and extract features
            if zero_shot_config and zero_shot_config.get('run_zero_shot', False):
                self.logger.info(f"🤖 Running zero-shot classification with {zero_shot_config.get('zero_shot_model', 'default')} model...")
                zero_shot_features, zero_shot_names = self._run_and_extract_zero_shot_features(df, dataset_name, zero_shot_config)
                all_features.append(zero_shot_features)
                feature_names.extend(zero_shot_names)
                features_info['zero_shot_features'] = len(zero_shot_names)
                features_info['zero_shot_model'] = zero_shot_config.get('zero_shot_model')
                features_info['zero_shot_confidence_threshold'] = zero_shot_config.get('confidence_threshold')
            else:
                features_info['zero_shot_features'] = 0
            
            # 11. Load and integrate previous analysis results (excluding zero-shot since we just ran it)
            previous_results = self._load_previous_analysis_results(dataset_name, exclude_zero_shot=True)
            if previous_results:
                prev_features, prev_names, prev_categories = self._extract_features_from_previous_results(previous_results, len(df))
                all_features.extend(prev_features)
                feature_names.extend(prev_names)
                features_info['previous_analysis_features'] = len(prev_names)
            else:
                features_info['previous_analysis_features'] = 0
            
            # Combine all features - ensure consistent dtypes to avoid object arrays
            processed_features = []
            for feature_array in all_features:
                if feature_array.size > 0:  # Only add non-empty arrays
                    # Convert to float32 to ensure consistent dtype
                    if feature_array.dtype == object:
                        # Try to convert object arrays to numeric
                        try:
                            feature_array = feature_array.astype(np.float32)
                        except (ValueError, TypeError):
                            self.logger.warning(f"Skipping non-numeric feature array with shape {feature_array.shape}")
                            continue
                    else:
                        feature_array = feature_array.astype(np.float32)
                    processed_features.append(feature_array)
            
            if processed_features:
                X = np.hstack(processed_features)
            else:
                # Fallback: create dummy features if no valid features
                self.logger.warning("No valid features found, creating dummy features")
                X = np.zeros((len(df), 1), dtype=np.float32)
                feature_names = ['dummy_feature']
            
            y = df['LABEL'].values if 'LABEL' in df.columns else np.zeros(len(df))
            
            # Ensure y labels are proper integer type
            y = y.astype(np.int32)
            
            # Save features
            features_dir = Path('datasets') / dataset_name / 'features'
            features_dir.mkdir(parents=True, exist_ok=True)
            
            # X should already be float32 from the processing above
            # y should already be int32 from the conversion above
            np.save(features_dir / 'X_features.npy', X)
            np.save(features_dir / 'y_labels.npy', y)
            
            # Also save individual feature type arrays for model trainer compatibility
            unified_dir = features_dir / 'unified'
            unified_dir.mkdir(exist_ok=True)
            
            feature_start = 0
            for i, (feature_array, feature_type) in enumerate(zip(processed_features, ['data_processor', 'text', 'behavioral', 'sentiment', 'language', 'theoretical', 'network', 'transformer_embeddings'])):
                if feature_array.size > 0:
                    feature_end = feature_start + feature_array.shape[1]
                    individual_features = X[:, feature_start:feature_end]
                    np.save(unified_dir / f'{feature_type}_features.npy', individual_features)
                    feature_start = feature_end
            
            # Save feature names and info
            with open(features_dir / 'feature_names.txt', 'w', encoding='utf-8') as f:
                for name in feature_names:
                    f.write(f"{name}\n")
            
            features_info['total_features'] = X.shape[1]
            features_info['feature_names'] = feature_names
            
            # Categorize features for better analysis
            feature_categories = self.categorize_features(feature_names)
            features_info['feature_categories'] = feature_categories
            
            features_info['feature_breakdown'] = {
                'text': features_info.get('text_features', 0),
                'behavioral': features_info.get('behavioral_features', 0),
                'sentiment': features_info.get('sentiment_features', 0),
                'language': features_info.get('language_features', 0),
                'theoretical': features_info.get('theoretical_features', 0),
                'network': features_info.get('network_features', 0),
                'transformer_embeddings': features_info.get('embedding_features', 0)
            }
            
            # Debug logging for feature breakdown
            self.logger.info(f"Feature breakdown: {features_info['feature_breakdown']}")
            self.logger.info(f"Individual feature counts: text={features_info.get('text_features', 0)}, "
                           f"behavioral={features_info.get('behavioral_features', 0)}, "
                           f"sentiment={features_info.get('sentiment_features', 0)}, "
                           f"theoretical={features_info.get('theoretical_features', 0)}")
            
            # Save comprehensive features info
            self.file_manager.save_results(dataset_name, features_info, 'comprehensive_features_info')
            
            # Update dataset info
            self.file_manager.update_dataset_info(dataset_name, {
                'status': 'comprehensive_features_extracted',
                'comprehensive_features_extracted': True,
                'total_comprehensive_features': X.shape[1],
                'feature_breakdown': features_info['feature_breakdown']
            })
            
            self.logger.info(f"Comprehensive features extracted successfully. Shape: {X.shape}")
            return features_info
            
        except Exception as e:
            self.logger.error(f"Error extracting comprehensive features: {e}")
            raise
    
    def _extract_embedding_features(self, df, dataset_name, embedding_config=None):
        """Extract transformer-based embedding features using exact column matching."""
        try:
            self.logger.info("Extracting transformer embedding features with exact column matching...")
            
            # Use exact column matching to find text content
            text_content = self._safe_column_access(df, 'TWEET_CONTENT', '')
            texts = text_content.fillna('').astype(str).tolist()
            
            # Get embedding configuration
            if embedding_config is None:
                embedding_config = {
                    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'embedding_strategy': 'mean_pooling',
                    'reduce_embeddings': False
                }
            
            model_name = embedding_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            strategy = embedding_config.get('embedding_strategy', 'mean_pooling')
            reduce_dims = embedding_config.get('reduce_embeddings', False)
            
            self.logger.info(f"Using embedding model: {model_name} with {strategy} strategy")
            
            # Get BERT embeddings
            bert_embeddings = []
            sentence_embeddings = []
            
            # Load models from local model manager
            bert_model = self.model_manager.load_model('bert-base-uncased')
            sentence_model = self.model_manager.load_model('all-MiniLM-L6-v2')
            
            for text in texts:
                try:
                    # BERT embeddings (768 dimensions)
                    if bert_model:
                        bert_emb = self.model_manager.get_embeddings(text, 'bert-base-uncased')
                        if bert_emb is not None:
                            bert_embeddings.append(bert_emb)
                        else:
                            bert_embeddings.append(np.zeros(768))  # Fallback
                    else:
                        bert_embeddings.append(np.zeros(768))
                    
                    # Sentence transformer embeddings (384 dimensions)
                    if sentence_model:
                        sent_emb = self.model_manager.get_embeddings(text, 'all-MiniLM-L6-v2')
                        if sent_emb is not None:
                            sentence_embeddings.append(sent_emb)
                        else:
                            sentence_embeddings.append(np.zeros(384))  # Fallback
                    else:
                        sentence_embeddings.append(np.zeros(384))
                        
                except Exception as e:
                    self.logger.warning(f"Error processing text embedding: {e}")
                    bert_embeddings.append(np.zeros(768))
                    sentence_embeddings.append(np.zeros(384))
            
            # Convert to numpy arrays
            bert_features = np.array(bert_embeddings)
            sentence_features = np.array(sentence_embeddings)
            
            # Combine embeddings
            embedding_features = np.hstack([bert_features, sentence_features])
            
            # Generate feature names
            bert_names = [f'bert_dim_{i}' for i in range(768)]
            sentence_names = [f'sentence_transformer_dim_{i}' for i in range(384)]
            embedding_names = bert_names + sentence_names
            
            # Save embeddings separately for future use
            features_dir = Path('datasets') / dataset_name / 'features'
            features_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(features_dir / 'bert_embeddings.npy', bert_features)
            np.save(features_dir / 'sentence_embeddings.npy', sentence_features)
            
            self.logger.info(f"Extracted {embedding_features.shape[1]} embedding features")
            return embedding_features, embedding_names
            
        except Exception as e:
            self.logger.error(f"Error extracting embedding features: {e}")
            # Return zeros if extraction fails
            n_samples = len(df)
            fallback_features = np.zeros((n_samples, 1152))  # 768 + 384
            fallback_names = [f'embedding_fallback_{i}' for i in range(1152)]
            return fallback_features, fallback_names
    
    def get_feature_names(self, dataset_name):
        """Get feature names for a dataset."""
        try:
            features_dir = Path('datasets') / dataset_name / 'features'
            names_file = features_dir / 'feature_names.txt'
            
            if names_file.exists():
                with open(names_file, 'r') as f:
                    return [line.strip() for line in f.readlines()]
            else:
                # Generate default names
                X, _, _ = self.load_features(dataset_name)
                if X is not None:
                    return [f'feature_{i}' for i in range(X.shape[1])]
                return []
        except:
            return []
    
    def get_feature_types(self, dataset_name):
        """Get feature type breakdown for a dataset."""
        try:
            results = self.file_manager.load_results(dataset_name, 'comprehensive_features_info')
            if results:
                return results.get('feature_breakdown', {})
            return {'text': 0, 'behavioral': 0, 'sentiment': 0, 'embeddings': 0}
        except:
            return {'text': 0, 'behavioral': 0, 'sentiment': 0, 'embeddings': 0}
    
    def get_scaler(self, dataset_name):
        """Get the trained scaler for a dataset."""
        try:
            scaler_path = Path('datasets') / dataset_name / 'features' / 'feature_scaler.joblib'
            if scaler_path.exists():
                # Use compatibility manager for safe loading
                scaler_dict = self.compatibility_manager.safe_load_model(scaler_path)
                if scaler_dict and 'model' in scaler_dict:
                    return scaler_dict['model']
                elif scaler_dict:
                    return scaler_dict
                else:
                    # Fallback to direct joblib loading
                    return joblib.load(scaler_path)
            return None
        except Exception as e:
            self.logger.warning(f"Could not load scaler: {e}")
            return None
    
    def extract_features_for_text(self, text, dataset_name):
        """Extract features for a single text using saved components."""
        try:
            # Use the existing extract_text_features method for a single text
            features = self.extract_text_features(dataset_name, [text])
            if features is not None and len(features) > 0:
                return features[0]  # Return the first (and only) row
            return None
        except Exception as e:
            self.logger.error(f"Error extracting features for single text: {e}")
            return None
    
    def extract_text_features(self, dataset_name, texts):
        """Extract features from text list using saved components."""
        try:
            features_dir = Path('datasets') / dataset_name / 'features'
            
            # Load vectorizer
            vectorizer_path = features_dir / 'tfidf_vectorizer.joblib'
            if not vectorizer_path.exists():
                self.logger.error(f"TF-IDF vectorizer not found for {dataset_name}")
                return None
            
            # Use compatibility manager for safe loading
            try:
                vectorizer_dict = self.compatibility_manager.safe_load_model(vectorizer_path)
                if vectorizer_dict and 'model' in vectorizer_dict:
                    vectorizer = vectorizer_dict['model']
                elif vectorizer_dict:
                    vectorizer = vectorizer_dict
                else:
                    vectorizer = joblib.load(vectorizer_path)
            except Exception as e:
                self.logger.warning(f"Compatibility loading failed, trying direct joblib: {e}")
                vectorizer = joblib.load(vectorizer_path)
            tfidf_features = vectorizer.transform(texts).toarray()
            
            # Try to add embeddings if available
            try:
                from src.model_manager import ModelManager
                model_manager = ModelManager()
                
                all_features = []
                for text in texts:
                    # Get BERT embeddings
                    bert_emb = model_manager.get_embeddings(text, 'bert-base-uncased')
                    if bert_emb is not None:
                        bert_features = bert_emb.reshape(1, -1)
                    else:
                        bert_features = np.zeros((1, 768))
                    
                    # Get sentence transformer embeddings
                    sent_emb = model_manager.get_embeddings(text, 'all-MiniLM-L6-v2')
                    if sent_emb is not None:
                        sentence_features = sent_emb.reshape(1, -1)
                    else:
                        sentence_features = np.zeros((1, 384))
                    
                    # Combine features
                    combined = np.hstack([
                        tfidf_features[len(all_features):len(all_features)+1],
                        bert_features,
                        sentence_features
                    ])
                    all_features.append(combined)
                
                return np.vstack(all_features)
                
            except Exception as e:
                self.logger.warning(f"Could not add embeddings, using TF-IDF only: {e}")
                return tfidf_features
                
        except Exception as e:
            self.logger.error(f"Error extracting text features: {e}")
            return None
    
    def categorize_features(self, feature_names):
        """Categorize features by type for better analysis and visualization."""
        categories = {
            'text_features': [],
            'behavioral_features': [],
            'sentiment_features': [],
            'network_features': [],
            'theoretical_features': [],
            'language_features': [],
            'zero_shot_features': [],
            'embedding_features': []
        }
        
        for name in feature_names:
            # Text/TF-IDF features
            if name.startswith('tfidf_'):
                categories['text_features'].append(name)
            
            # Behavioral features
            elif any(keyword in name.lower() for keyword in [
                'behavioral', 'engagement', 'activity', 'high_activity', 
                'follower', 'following', 'verified', 'profile'
            ]):
                categories['behavioral_features'].append(name)
            
            # Sentiment features
            elif any(keyword in name.lower() for keyword in [
                'sentiment', 'positive', 'negative', 'compound', 'neutral'
            ]):
                categories['sentiment_features'].append(name)
            
            # Network features
            elif any(keyword in name.lower() for keyword in [
                'network', 'mention', 'hashtag', 'url', 'retweet', 'reply'
            ]):
                categories['network_features'].append(name)
            
            # Theoretical framework features (RAT, RCT, UGT)
            elif any(keyword in name.lower() for keyword in [
                'rat_', 'rct_', 'cg_', 'perceived_risk', 'perceived_benefit',
                'coping_appraisal', 'threat_appraisal', 'gratification'
            ]):
                categories['theoretical_features'].append(name)
            
            # Language features
            elif any(keyword in name.lower() for keyword in [
                'lang_', 'language', 'english', 'swahili', 'mixed'
            ]):
                categories['language_features'].append(name)
            
            # Zero-shot features
            elif name.startswith('zeroshot_'):
                categories['zero_shot_features'].append(name)
            
            # Embedding features (BERT, XLM-RoBERTa, etc.)
            elif any(keyword in name.lower() for keyword in [
                'bert_', 'xlm_', 'roberta_', 'embedding_', 'transformer_'
            ]):
                categories['embedding_features'].append(name)
        
        # Calculate category statistics
        category_stats = {}
        total_features = len(feature_names)
        
        for category, features in categories.items():
            category_stats[category] = {
                'count': len(features),
                'percentage': (len(features) / total_features * 100) if total_features > 0 else 0,
                'features': features[:10]  # Store first 10 feature names as examples
            }
        
        return category_stats
    
    def _extract_enhanced_text_features(self, df, dataset_name):
        """Extract enhanced text-based behavioral features and LDA topics."""
        self.logger.info("Extracting enhanced text features with behavioral analysis and LDA...")
        
        # Use exact column matching to find text content
        text_content = self._safe_column_access(df, 'TWEET_CONTENT', '')
        texts = text_content.fillna('').astype(str).tolist()
        
        if not any(text.strip() for text in texts):
            self.logger.warning("No valid text content found, creating dummy enhanced text features")
            dummy_features = np.zeros((len(df), 10))
            feature_names = [f"enhanced_text_dummy_{i}" for i in range(10)]
            return dummy_features, feature_names
        
        # Extract comprehensive text features (behavioral + LDA)
        enhanced_features, enhanced_names = self.extract_comprehensive_text_features(
            texts, dataset_name, include_lda=True, n_topics=10
        )
        
        # Prefix feature names to avoid conflicts
        prefixed_names = [f"enhanced_{name}" for name in enhanced_names]
        
        self.logger.info(f"Extracted {len(prefixed_names)} enhanced text features")
        return enhanced_features, prefixed_names
    
    def _find_text_column(self, df):
        """Find the main text column in the dataframe."""
        possible_names = ['TWEET_CONTENT', 'text', 'content', 'tweet', 'message', 'post']
        
        for col in df.columns:
            if col.upper() in [name.upper() for name in possible_names]:
                return col
        
        # If not found, return the first string column with substantial content
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 10:  # Assume text columns have average length > 10
                    return col
        
        return None
    
    # Enhanced Text Analysis Methods (moved from enhanced_text_analyzer.py)
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        required_data = [
            'punkt', 'stopwords', 'vader_lexicon', 
            'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'
        ]
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except:
                    self.logger.warning(f"Could not download NLTK data: {data}")
    
    def _compile_behavioral_patterns(self):
        """Compile regex patterns for behavioral analysis."""
        self.patterns = {
            # Urgency indicators
            'urgency': re.compile(r'\b(urgent|immediately|asap|now|quick|fast|hurry|rush)\b', re.IGNORECASE),
            
            # Authority claims
            'authority': re.compile(r'\b(expert|official|confirmed|verified|proven|fact|truth|research shows|studies show)\b', re.IGNORECASE),
            
            # Emotional manipulation
            'fear': re.compile(r'\b(danger|threat|risk|scary|afraid|fear|panic|terror|disaster|crisis)\b', re.IGNORECASE),
            'anger': re.compile(r'\b(outrage|angry|furious|mad|hate|disgusting|terrible|awful|horrible)\b', re.IGNORECASE),
            'excitement': re.compile(r'\b(amazing|incredible|unbelievable|shocking|wow|omg|fantastic|awesome)\b', re.IGNORECASE),
            
            # Social proof
            'social_proof': re.compile(r'\b(everyone|everybody|most people|many people|thousands|millions|trending|viral)\b', re.IGNORECASE),
            
            # Conspiracy indicators
            'conspiracy': re.compile(r'\b(cover.?up|conspiracy|hidden|secret|they don\'t want|mainstream media|wake up|sheeple)\b', re.IGNORECASE),
            
            # Certainty/uncertainty
            'certainty': re.compile(r'\b(definitely|certainly|absolutely|guaranteed|sure|proven|fact|truth)\b', re.IGNORECASE),
            'uncertainty': re.compile(r'\b(maybe|perhaps|possibly|might|could|allegedly|reportedly|supposedly)\b', re.IGNORECASE),
            
            # Call to action
            'call_to_action': re.compile(r'\b(share|retweet|spread|tell everyone|pass it on|don\'t let them|fight|resist|act now)\b', re.IGNORECASE),
            
            # Personal anecdotes
            'personal': re.compile(r'\b(i saw|i heard|my friend|someone told me|i know someone|personal experience)\b', re.IGNORECASE),
            
            # Sensationalism
            'sensational': re.compile(r'\b(breaking|exclusive|shocking|revealed|exposed|leaked|bombshell|scandal)\b', re.IGNORECASE),
        }
    
    def extract_advanced_behavioral_features(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract advanced behavioral features from text content only.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        self.logger.info("Extracting advanced text-based behavioral features...")
        
        features = []
        feature_names = []
        
        # Process each text
        for text in texts:
            text_features = self._extract_single_text_behavioral_features(text)
            features.append(text_features)
        
        # Convert to numpy array
        feature_matrix = np.array(features)
        
        # Generate feature names
        feature_names = self._get_behavioral_feature_names()
        
        self.logger.info(f"Extracted {len(feature_names)} advanced behavioral features")
        return feature_matrix, feature_names
    
    def _extract_single_text_behavioral_features(self, text: str) -> List[float]:
        """Extract behavioral features from a single text."""
        if not isinstance(text, str) or not text.strip():
            return [0.0] * 50  # Return zeros for empty text
        
        features = []
        text_lower = text.lower()
        
        # Basic text statistics
        features.extend(self._extract_basic_text_stats(text))
        
        # Linguistic features
        features.extend(self._extract_linguistic_features(text))
        
        # Behavioral pattern matching
        features.extend(self._extract_pattern_features(text))
        
        # Sentiment and emotion features
        features.extend(self._extract_emotion_features(text))
        
        # Readability and complexity
        features.extend(self._extract_readability_features(text))
        
        # Social and psychological indicators
        features.extend(self._extract_psychological_features(text))
        
        return features
    
    def _extract_basic_text_stats(self, text: str) -> List[float]:
        """Extract basic text statistics."""
        features = []
        
        # Length features
        features.append(len(text))  # Character count
        features.append(len(text.split()))  # Word count
        features.append(len(sent_tokenize(text)))  # Sentence count
        
        # Average lengths
        words = text.split()
        if words:
            features.append(np.mean([len(word) for word in words]))  # Avg word length
            features.append(len(text) / len(words))  # Avg chars per word
        else:
            features.extend([0.0, 0.0])
        
        sentences = sent_tokenize(text)
        if sentences:
            features.append(len(words) / len(sentences))  # Avg words per sentence
        else:
            features.append(0.0)
        
        # Punctuation features
        features.append(text.count('!'))  # Exclamation marks
        features.append(text.count('?'))  # Question marks
        features.append(text.count('.'))  # Periods
        features.append(text.count(','))  # Commas
        
        # Capitalization features
        features.append(sum(1 for c in text if c.isupper()) / len(text) if text else 0)  # Uppercase ratio
        features.append(len([word for word in words if word.isupper()]) / len(words) if words else 0)  # All caps words ratio
        
        return features
    
    def _extract_linguistic_features(self, text: str) -> List[float]:
        """Extract linguistic features."""
        features = []
        
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # POS tag ratios
            total_tokens = len(pos_tags) if pos_tags else 1
            
            # Count different POS types
            pos_counts = Counter([tag for _, tag in pos_tags])
            
            # Noun ratio
            noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
            noun_count = sum(pos_counts.get(tag, 0) for tag in noun_tags)
            features.append(noun_count / total_tokens)
            
            # Verb ratio
            verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
            verb_count = sum(pos_counts.get(tag, 0) for tag in verb_tags)
            features.append(verb_count / total_tokens)
            
            # Adjective ratio
            adj_tags = ['JJ', 'JJR', 'JJS']
            adj_count = sum(pos_counts.get(tag, 0) for tag in adj_tags)
            features.append(adj_count / total_tokens)
            
            # Adverb ratio
            adv_tags = ['RB', 'RBR', 'RBS']
            adv_count = sum(pos_counts.get(tag, 0) for tag in adv_tags)
            features.append(adv_count / total_tokens)
            
            # Pronoun ratio
            pronoun_tags = ['PRP', 'PRP$']
            pronoun_count = sum(pos_counts.get(tag, 0) for tag in pronoun_tags)
            features.append(pronoun_count / total_tokens)
            
        except Exception as e:
            self.logger.warning(f"Error in linguistic analysis: {e}")
            features.extend([0.0] * 5)
        
        return features
    
    def _extract_pattern_features(self, text: str) -> List[float]:
        """Extract behavioral pattern features."""
        features = []
        
        # Count pattern matches
        for pattern_name, pattern in self.patterns.items():
            matches = len(pattern.findall(text))
            # Normalize by text length
            normalized_count = matches / len(text.split()) if text.split() else 0
            features.append(normalized_count)
        
        return features
    
    def _extract_emotion_features(self, text: str) -> List[float]:
        """Extract emotion and sentiment features."""
        features = []
        
        # VADER sentiment
        sentiment_scores = self.sentiment_intensity_analyzer.polarity_scores(text)
        features.extend([
            sentiment_scores['pos'],
            sentiment_scores['neu'], 
            sentiment_scores['neg'],
            sentiment_scores['compound']
        ])
        
        # Emotional intensity
        features.append(abs(sentiment_scores['compound']))  # Emotional intensity
        
        # Sentiment consistency (variance across sentences)
        sentences = sent_tokenize(text)
        if len(sentences) > 1:
            sent_scores = [self.sentiment_intensity_analyzer.polarity_scores(sent)['compound'] for sent in sentences]
            features.append(np.var(sent_scores))  # Sentiment variance
        else:
            features.append(0.0)
        
        return features
    
    def _extract_readability_features(self, text: str) -> List[float]:
        """Extract readability and complexity features."""
        features = []
        
        try:
            # Readability scores
            features.append(textstat.flesch_reading_ease(text))
            features.append(textstat.flesch_kincaid_grade(text))
            features.append(textstat.automated_readability_index(text))
            features.append(textstat.coleman_liau_index(text))
            
        except Exception as e:
            self.logger.warning(f"Error in readability analysis: {e}")
            features.extend([0.0] * 4)
        
        return features
    
    def _extract_psychological_features(self, text: str) -> List[float]:
        """Extract psychological and social indicators."""
        features = []
        
        # First person usage (self-reference)
        first_person = len(re.findall(r'\b(i|me|my|mine|myself)\b', text.lower()))
        features.append(first_person / len(text.split()) if text.split() else 0)
        
        # Second person usage (direct address)
        second_person = len(re.findall(r'\b(you|your|yours|yourself)\b', text.lower()))
        features.append(second_person / len(text.split()) if text.split() else 0)
        
        # Third person usage
        third_person = len(re.findall(r'\b(he|she|they|them|their|his|her|him)\b', text.lower()))
        features.append(third_person / len(text.split()) if text.split() else 0)
        
        # Cognitive complexity (use of complex conjunctions)
        complex_conjunctions = len(re.findall(r'\b(however|therefore|nevertheless|furthermore|moreover|consequently)\b', text.lower()))
        features.append(complex_conjunctions / len(text.split()) if text.split() else 0)
        
        # Temporal references
        temporal_refs = len(re.findall(r'\b(now|today|yesterday|tomorrow|recently|soon|later|before|after)\b', text.lower()))
        features.append(temporal_refs / len(text.split()) if text.split() else 0)
        
        return features
    
    def _get_behavioral_feature_names(self) -> List[str]:
        """Get names for all behavioral features."""
        names = []
        
        # Basic text stats (12 features)
        names.extend([
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'avg_chars_per_word', 'avg_words_per_sentence', 'exclamation_count',
            'question_count', 'period_count', 'comma_count', 'uppercase_ratio',
            'allcaps_words_ratio'
        ])
        
        # Linguistic features (5 features)
        names.extend(['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'pronoun_ratio'])
        
        # Pattern features (12 features - one for each pattern)
        names.extend([f'pattern_{name}' for name in self.patterns.keys()])
        
        # Emotion features (6 features)
        names.extend(['sentiment_pos', 'sentiment_neu', 'sentiment_neg', 'sentiment_compound', 'emotional_intensity', 'sentiment_variance'])
        
        # Readability features (4 features)
        names.extend(['flesch_reading_ease', 'flesch_kincaid_grade', 'automated_readability', 'coleman_liau'])
        
        # Psychological features (5 features)
        names.extend(['first_person_ratio', 'second_person_ratio', 'third_person_ratio', 'complex_conjunctions_ratio', 'temporal_refs_ratio'])
        
        return names
    
    def extract_lda_features(self, texts: List[str], dataset_name: str, n_topics: int = 10) -> Tuple[np.ndarray, List[str]]:
        """
        Extract LDA topic modeling features.
        
        Args:
            texts: List of text strings
            dataset_name: Name of the dataset for model saving
            n_topics: Number of topics for LDA
            
        Returns:
            Tuple of (topic_features, feature_names)
        """
        self.logger.info(f"Extracting LDA features with {n_topics} topics...")
        
        # Clean texts
        cleaned_texts = [self._clean_text_for_lda(text) for text in texts]
        non_empty_texts = [text for text in cleaned_texts if text.strip()]
        
        if len(non_empty_texts) < 2:
            self.logger.warning("Not enough texts for LDA, returning dummy features")
            dummy_features = np.zeros((len(texts), n_topics))
            feature_names = [f'topic_{i}' for i in range(n_topics)]
            return dummy_features, feature_names
        
        # Create count vectorizer for LDA
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        try:
            # Fit vectorizer on non-empty texts
            doc_term_matrix = vectorizer.fit_transform(non_empty_texts)
            
            # Train LDA model
            lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100,
                learning_method='batch'
            )
            
            lda_model.fit(doc_term_matrix)
            
            # Transform all texts (including empty ones)
            all_doc_term_matrix = vectorizer.transform(cleaned_texts)
            topic_features = lda_model.transform(all_doc_term_matrix)
            
            # Save models
            self._save_lda_models(dataset_name, lda_model, vectorizer)
            
            # Generate feature names
            feature_names = [f'topic_{i}' for i in range(n_topics)]
            
            self.logger.info(f"Successfully extracted {n_topics} LDA topic features")
            return topic_features, feature_names
            
        except Exception as e:
            self.logger.error(f"Error in LDA feature extraction: {e}")
            # Return dummy features
            dummy_features = np.zeros((len(texts), n_topics))
            feature_names = [f'topic_{i}' for i in range(n_topics)]
            return dummy_features, feature_names
    
    def _clean_text_for_lda(self, text: str) -> str:
        """Clean text for LDA processing."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _save_lda_models(self, dataset_name: str, lda_model, vectorizer):
        """Save LDA model and vectorizer."""
        try:
            # Create directory
            model_dir = Path('datasets') / dataset_name / 'features'
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save models
            lda_path = model_dir / 'lda_model.joblib'
            vectorizer_path = model_dir / 'lda_vectorizer.joblib'
            
            joblib.dump(lda_model, lda_path)
            joblib.dump(vectorizer, vectorizer_path)
            
            # Store in memory for reuse
            self.lda_models[dataset_name] = lda_model
            self.topic_vectorizers[dataset_name] = vectorizer
            
            self.logger.info(f"LDA models saved for dataset: {dataset_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving LDA models: {e}")
    
    def load_lda_models(self, dataset_name: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Load saved LDA models."""
        try:
            model_dir = Path('datasets') / dataset_name / 'features'
            lda_path = model_dir / 'lda_model.joblib'
            vectorizer_path = model_dir / 'lda_vectorizer.joblib'
            
            if lda_path.exists() and vectorizer_path.exists():
                lda_model = joblib.load(lda_path)
                vectorizer = joblib.load(vectorizer_path)
                
                # Store in memory
                self.lda_models[dataset_name] = lda_model
                self.topic_vectorizers[dataset_name] = vectorizer
                
                return lda_model, vectorizer
            else:
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error loading LDA models: {e}")
            return None, None
    
    def get_topic_words(self, dataset_name: str, n_words: int = 10) -> Dict[int, List[str]]:
        """Get top words for each topic."""
        if dataset_name not in self.lda_models or dataset_name not in self.topic_vectorizers:
            lda_model, vectorizer = self.load_lda_models(dataset_name)
            if lda_model is None:
                return {}
        else:
            lda_model = self.lda_models[dataset_name]
            vectorizer = self.topic_vectorizers[dataset_name]
        
        try:
            feature_names = vectorizer.get_feature_names_out()
            topic_words = {}
            
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words_idx = topic.argsort()[-n_words:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_words[topic_idx] = top_words
            
            return topic_words
            
        except Exception as e:
            self.logger.error(f"Error getting topic words: {e}")
            return {}
    
    def extract_comprehensive_text_features(self, texts: List[str], dataset_name: str, 
                                          include_lda: bool = True, n_topics: int = 10) -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive text features including behavioral analysis and LDA.
        
        Args:
            texts: List of text strings
            dataset_name: Name of the dataset
            include_lda: Whether to include LDA features
            n_topics: Number of topics for LDA
            
        Returns:
            Tuple of (combined_features, feature_names)
        """
        self.logger.info("Extracting comprehensive text features...")
        
        all_features = []
        all_feature_names = []
        
        # Extract behavioral features
        behavioral_features, behavioral_names = self.extract_advanced_behavioral_features(texts)
        all_features.append(behavioral_features)
        all_feature_names.extend(behavioral_names)
        
        # Extract LDA features if requested
        if include_lda:
            lda_features, lda_names = self.extract_lda_features(texts, dataset_name, n_topics)
            all_features.append(lda_features)
            all_feature_names.extend(lda_names)
        
        # Combine all features
        if all_features:
            combined_features = np.hstack(all_features)
        else:
            combined_features = np.zeros((len(texts), 1))
            all_feature_names = ['dummy_feature']
        
        self.logger.info(f"Extracted {combined_features.shape[1]} comprehensive text features")
        return combined_features, all_feature_names

    def extract_features_for_text(self, text, feature_names=None):
        """
        Extract features for a single text input for prediction using saved components.
        
        Args:
            text: Single text string to analyze
            feature_names: List of expected feature names (should match training features)
            
        Returns:
            Tuple of (X_features, y_dummy, feature_names)
        """
        try:
            if feature_names is None:
                # Fallback to comprehensive text features only
                self.logger.warning("No feature names provided, using comprehensive text features")
                texts = [text] if isinstance(text, str) else text
                features, extracted_feature_names = self.extract_comprehensive_text_features(
                    texts, 
                    dataset_name='prediction_temp',
                    include_lda=False,
                    n_topics=5
                )
                return features, None, extracted_feature_names
            
            # Create a temporary DataFrame with the single text
            import pandas as pd
            temp_df = pd.DataFrame({
                'TEXT': [text],
                'CONTENT': [text],
                'CLEANED_TEXT': [text],
                'LABEL': [0]  # Dummy label
            })
            
            self.logger.info(f"Extracting {len(feature_names)} features for single text prediction")
            
            # Extract all feature types that were used during training
            all_features = []
            
            # 1. Text features (TF-IDF, embeddings, etc.)
            try:
                text_features = self._extract_enhanced_text_features(temp_df, 'prediction_temp')
                if text_features is not None and text_features.shape[1] > 0:
                    all_features.append(text_features)
                    self.logger.debug(f"Added {text_features.shape[1]} text features")
            except Exception as e:
                self.logger.warning(f"Could not extract text features: {e}")
            
            # 2. Behavioral features
            try:
                behavioral_features = self._extract_behavioral_features(temp_df)
                if behavioral_features is not None and behavioral_features.shape[1] > 0:
                    all_features.append(behavioral_features)
                    self.logger.debug(f"Added {behavioral_features.shape[1]} behavioral features")
            except Exception as e:
                self.logger.warning(f"Could not extract behavioral features: {e}")
            
            # 3. Sentiment features
            try:
                sentiment_features = self._extract_sentiment_features(temp_df)
                if sentiment_features is not None and sentiment_features.shape[1] > 0:
                    all_features.append(sentiment_features)
                    self.logger.debug(f"Added {sentiment_features.shape[1]} sentiment features")
            except Exception as e:
                self.logger.warning(f"Could not extract sentiment features: {e}")
            
            # 4. Basic features
            try:
                basic_features = self._extract_basic_features(temp_df)
                if basic_features is not None and basic_features.shape[1] > 0:
                    all_features.append(basic_features)
                    self.logger.debug(f"Added {basic_features.shape[1]} basic features")
            except Exception as e:
                self.logger.warning(f"Could not extract basic features: {e}")
            
            # 5. Theoretical framework features
            try:
                theoretical_features = self._extract_theoretical_features(temp_df)
                if theoretical_features is not None and theoretical_features.shape[1] > 0:
                    all_features.append(theoretical_features)
                    self.logger.debug(f"Added {theoretical_features.shape[1]} theoretical features")
            except Exception as e:
                self.logger.warning(f"Could not extract theoretical features: {e}")
            
            # Combine all features
            if all_features:
                combined_features = np.hstack(all_features)
                self.logger.info(f"Combined features shape: {combined_features.shape}")
                
                # If we have fewer features than expected, pad with zeros
                if combined_features.shape[1] < len(feature_names):
                    padding_size = len(feature_names) - combined_features.shape[1]
                    padding = np.zeros((1, padding_size))
                    combined_features = np.hstack([combined_features, padding])
                    self.logger.warning(f"Padded {padding_size} features to match expected size")
                
                # If we have more features than expected, truncate
                elif combined_features.shape[1] > len(feature_names):
                    combined_features = combined_features[:, :len(feature_names)]
                    self.logger.warning(f"Truncated features to match expected size: {len(feature_names)}")
                
                return combined_features, None, feature_names
            else:
                # No features extracted, return zeros
                self.logger.error("No features could be extracted")
                dummy_features = np.zeros((1, len(feature_names)))
                return dummy_features, None, feature_names
            
        except Exception as e:
            self.logger.error(f"Error extracting features for text: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return dummy features as fallback
            if feature_names:
                dummy_features = np.zeros((1, len(feature_names)))
                return dummy_features, None, feature_names
            else:
                dummy_features = np.zeros((1, 10))
                dummy_names = [f'feature_{i}' for i in range(10)]
                return dummy_features, None, dummy_names