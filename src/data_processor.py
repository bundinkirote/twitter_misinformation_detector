"""
Data Processor Module

This module provides comprehensive data cleaning and preprocessing functionality
for Twitter datasets. It handles text normalization, feature extraction,
and data quality validation for machine learning pipeline preparation.
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.utils.file_manager import FileManager

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class DataProcessor:
    """
    Data Processing and Cleaning Class
    
    Provides comprehensive data preprocessing capabilities including text cleaning,
    normalization, feature extraction, and quality validation for machine learning
    datasets. Supports various data formats and implements robust error handling.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_manager = FileManager()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
    
    def process_dataset(self, filepath, dataset_name):
        """
        Process uploaded dataset with comprehensive cleaning and validation.
        
        Args:
            filepath: Path to the dataset file
            dataset_name: Name identifier for the dataset
            
        Returns:
            Dictionary containing processing results and statistics
        """
        self.logger.info(f"Processing dataset: {dataset_name}")
        
        try:
            # Load data
            if filepath.endswith('.csv'):
                # First, try to detect if the CSV has headers in row 2 (common in network analysis exports)
                try:
                    # Read first few rows without headers to check structure
                    sample_df = pd.read_csv(filepath, header=None, nrows=3)
                    
                    # Check if first row contains mostly empty/header-like values
                    first_row_empty_ratio = sample_df.iloc[0].isna().sum() / len(sample_df.columns)
                    
                    # Also check if second row looks like proper column names
                    if len(sample_df) > 1:
                        second_row_values = sample_df.iloc[1].astype(str).tolist()
                        # Check if second row contains typical column names
                        column_indicators = ['tweet', 'name', 'user', 'id', 'date', 'count', 'text', 'description']
                        second_row_has_columns = any(
                            any(indicator in str(val).lower() for indicator in column_indicators)
                            for val in second_row_values if pd.notna(val)
                        )
                    else:
                        second_row_has_columns = False
                    
                    if first_row_empty_ratio > 0.8 and second_row_has_columns:
                        # Headers are likely in row 2, skip row 1
                        df = pd.read_csv(filepath, header=1)
                        self.logger.info("Detected headers in row 2, skipping first row")
                    else:
                        df = pd.read_csv(filepath)
                except Exception:
                    # Fallback to standard reading
                    df = pd.read_csv(filepath)
                    
            elif filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                raise ValueError("Unsupported file format")
            
            self.logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Clean and preprocess
            df_cleaned = self._clean_data(df)
            df_processed = self._preprocess_text(df_cleaned)
            df_final = self._add_basic_features(df_processed)
            
            # Create dataset directory structure
            self.file_manager.create_dataset_directory(dataset_name)
            
            # Save processed data
            processed_path = Path('datasets') / dataset_name / 'processed' / 'processed_data.csv'
            df_final.to_csv(processed_path, index=False)
            
            # Update dataset info
            self.file_manager.update_dataset_info(dataset_name, {
                'status': 'processed',
                'original_shape': df.shape,
                'processed_shape': df_final.shape,
                'processing_date': datetime.now().isoformat()
            })
            
            self.logger.info(f"Dataset processed successfully. Final shape: {df_final.shape}")
            return df_final
            
        except Exception as e:
            self.logger.error(f"Error processing dataset: {e}")
            raise
    
    def _clean_data(self, df):
        """Clean the dataset by removing rows with missing critical data."""
        self.logger.info("Cleaning data...")
        
        # Make a copy
        df_clean = df.copy()
        initial_count = len(df_clean)
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.strip().str.upper()
        
        # Identify critical text columns (at least one must have content)
        text_columns = ['DESCRIPTION', 'TWEET_CONTENT', 'NAME', 'TWEET']
        available_text_cols = [col for col in text_columns if col in df_clean.columns]
        
        # Remove rows where ALL text columns are empty/null
        if available_text_cols:
            # Create a mask for rows that have at least some text content
            text_mask = pd.Series(False, index=df_clean.index)
            for col in available_text_cols:
                # Check for non-null, non-empty strings
                col_has_content = (
                    df_clean[col].notna() & 
                    (df_clean[col].astype(str).str.strip() != '') &
                    (df_clean[col].astype(str).str.strip() != 'nan')
                )
                text_mask = text_mask | col_has_content
            
            # Keep only rows with some text content
            df_clean = df_clean[text_mask]
            text_removed = initial_count - len(df_clean)
            if text_removed > 0:
                self.logger.info(f"Removed {text_removed} rows with no text content")
        
        # Handle missing values in text columns (fill remaining with empty string)
        for col in available_text_cols:
            df_clean[col] = df_clean[col].fillna('').astype(str)
        
        # Handle numeric columns - convert to numeric and fill NaN with 0
        numeric_columns = [
            'RETWEET_COUNT', 'FAVORITE_COUNT', 'REPLY_COUNT', 'QUOTE_COUNT', 'IMPRESSION_COUNT',
            'FOLLOWERS_COUNT', 'FOLLOWING_COUNT', 'TWEET_COUNT', 'LISTED_COUNT', 'FAVOURITES_COUNT', 'MEDIA_COUNT',
            'DEGREE', 'IN_DEGREE', 'OUT_DEGREE', 'BETWEENNESS_CENTRALITY', 'CLOSENESS_CENTRALITY', 
            'EIGENVECTOR_CENTRALITY', 'PAGERANK', 'CLUSTERING_COEFFICIENT',
            'TWEET_ENGAGEMENT_SCORE', 'USER_INFLUENCE_SCORE', 'VERIFICATION_SCORE',
            # Legacy column names for backward compatibility
            'FOLLOWERS', 'FOLLOWED', 'TWEETS'
        ]
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        # Handle LABEL column
        if 'LABEL' in df_clean.columns:
            # Convert to numeric, remove rows where label cannot be determined
            df_clean['LABEL'] = pd.to_numeric(df_clean['LABEL'], errors='coerce')
            
            # Remove rows with invalid labels (NaN)
            label_mask = df_clean['LABEL'].notna()
            label_removed = (~label_mask).sum()
            df_clean = df_clean[label_mask]
            if label_removed > 0:
                self.logger.info(f"Removed {label_removed} rows with invalid labels")
            
            df_clean['LABEL'] = df_clean['LABEL'].astype(int)
        else:
            # If no label column, create a placeholder (for unlabeled data)
            df_clean['LABEL'] = 0
        
        # Handle boolean columns - convert TRUE/FALSE strings to 1/0
        boolean_columns = ['VERIFIED', 'IS_BLUE_VERIFIED', 'POSSIBLY_SENSITIVE', 'TWEETED_SEARCH_TERM']
        for col in boolean_columns:
            if col in df_clean.columns:
                # Handle boolean values (TRUE/FALSE) and numeric values
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str).str.upper().map({
                        'TRUE': 1, 'FALSE': 0, 'T': 1, 'F': 0, '1': 1, '0': 0
                    }).fillna(0).astype(int)
                else:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
        
        # Handle date columns
        date_columns = ['TWEET_DATE', 'JOINED_TWITTER_DATE']
        for col in date_columns:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                except Exception:
                    self.logger.warning(f"Could not parse dates in column {col}")
        
        # Handle ID columns - ensure they are strings to preserve full precision
        id_columns = ['AUTHOR_ID', 'TWEET_ID', 'USER_ID', 'PINNED_TWEET_ID']
        for col in id_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str)
        
        # Final NaN cleanup - ensure no NaN values remain
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                # For numeric columns, fill NaN with 0
                nan_count = df_clean[col].isna().sum()
                if nan_count > 0:
                    self.logger.info(f"Filling {nan_count} NaN values in {col} with 0")
                    df_clean[col] = df_clean[col].fillna(0)
            elif df_clean[col].dtype == 'object':
                # For object columns, fill NaN with empty string
                nan_count = df_clean[col].isna().sum()
                if nan_count > 0:
                    self.logger.info(f"Filling {nan_count} NaN values in {col} with empty string")
                    df_clean[col] = df_clean[col].fillna('')
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        duplicate_removed = len(df_clean) - (len(df_clean.drop_duplicates()) if len(df_clean) > 0 else 0)
        if duplicate_removed > 0:
            self.logger.info(f"Removed {duplicate_removed} duplicate rows")
        
        final_count = len(df_clean)
        total_removed = initial_count - final_count
        
        if total_removed > 0:
            self.logger.info(f"Data cleaning complete: {total_removed} rows removed, {final_count} rows remaining")
        
        # Verify no NaN values remain
        nan_check = df_clean.isna().sum().sum()
        if nan_check > 0:
            self.logger.warning(f"Warning: {nan_check} NaN values still present after cleaning")
            # Force remove any remaining NaN rows
            df_clean = df_clean.dropna()
            self.logger.info(f"Dropped remaining rows with NaN values. Final count: {len(df_clean)}")
        
        return df_clean
    
    def _preprocess_text(self, df):
        """Preprocess text columns."""
        self.logger.info("Preprocessing text...")
        
        df_processed = df.copy()
        
        # Prioritize TWEET as the main text content, include DESCRIPTION and NAME for additional context
        text_columns = ['TWEET', 'DESCRIPTION', 'NAME']
        available_text_cols = [col for col in text_columns if col in df_processed.columns]
        
        if available_text_cols:
            # Convert all columns to string before joining
            text_data = df_processed[available_text_cols].fillna('').astype(str)
            df_processed['COMBINED_TEXT'] = text_data.agg(' '.join, axis=1)
        else:
            # Fallback to legacy column names
            legacy_text_columns = ['TWEET_CONTENT', 'DESCRIPTION', 'NAME']
            available_legacy_cols = [col for col in legacy_text_columns if col in df_processed.columns]
            
            if available_legacy_cols:
                text_data = df_processed[available_legacy_cols].fillna('').astype(str)
                df_processed['COMBINED_TEXT'] = text_data.agg(' '.join, axis=1)
            else:
                # If no standard text columns, try to find any text-like column
                text_like_cols = []
                for col in df_processed.columns:
                    if df_processed[col].dtype == 'object' and col.upper() not in ['LABEL', 'ID', 'USER_ID', 'AUTHOR_ID', 'TWEET_ID']:
                        # Check if column contains text (not just numbers or short codes)
                        sample_values = df_processed[col].dropna().head(10)
                        if len(sample_values) > 0:
                            avg_length = sample_values.astype(str).str.len().mean()
                            if avg_length > 10:  # Likely contains meaningful text
                                text_like_cols.append(col)
                
                if text_like_cols:
                    # Convert all columns to string before joining
                    text_data = df_processed[text_like_cols].fillna('').astype(str)
                    df_processed['COMBINED_TEXT'] = text_data.agg(' '.join, axis=1)
                else:
                    df_processed['COMBINED_TEXT'] = ''
        
        # Clean text
        df_processed['CLEANED_TEXT'] = df_processed['COMBINED_TEXT'].apply(self._clean_text)
        
        return df_processed
    
    def _clean_text(self, text):
        """Clean individual text."""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _add_basic_features(self, df):
        """Add basic features to the dataset."""
        self.logger.info("Adding basic features...")
        
        df_features = df.copy()
        
        # Text length features
        df_features['TEXT_LENGTH'] = df_features['CLEANED_TEXT'].str.len()
        df_features['WORD_COUNT'] = df_features['CLEANED_TEXT'].str.split().str.len()
        
        # Readability features
        def safe_readability_score(text):
            try:
                if pd.isna(text) or text == '':
                    return 0
                return textstat.flesch_reading_ease(str(text))
            except Exception:
                return 0
        
        df_features['READABILITY_SCORE'] = df_features['COMBINED_TEXT'].apply(safe_readability_score)
        
        # Sentiment features
        sentiment_scores = df_features['COMBINED_TEXT'].apply(self._get_sentiment_scores)
        df_features['SENTIMENT_POSITIVE'] = sentiment_scores.apply(lambda x: x['pos'])
        df_features['SENTIMENT_NEGATIVE'] = sentiment_scores.apply(lambda x: x['neg'])
        df_features['SENTIMENT_NEUTRAL'] = sentiment_scores.apply(lambda x: x['neu'])
        df_features['SENTIMENT_COMPOUND'] = sentiment_scores.apply(lambda x: x['compound'])
        
        # Engagement features - create derived metrics from base engagement data
        if 'RETWEET_COUNT' in df_features.columns and 'FAVORITE_COUNT' in df_features.columns:
            df_features['TOTAL_ENGAGEMENT'] = (
                df_features['RETWEET_COUNT'] + 
                df_features['FAVORITE_COUNT'] + 
                df_features.get('REPLY_COUNT', 0) + 
                df_features.get('QUOTE_COUNT', 0)
            )
            
            # Engagement rate relative to impression count
            if 'IMPRESSION_COUNT' in df_features.columns:
                df_features['ENGAGEMENT_RATE'] = np.where(
                    df_features['IMPRESSION_COUNT'] > 0,
                    df_features['TOTAL_ENGAGEMENT'] / df_features['IMPRESSION_COUNT'],
                    0
                )
        
        # Account features - handle different column names
        followers_col = None
        following_col = None
        
        # Check for different follower column names
        for col in ['FOLLOWERS_COUNT', 'FOLLOWERS']:
            if col in df_features.columns:
                followers_col = col
                break
                
        # Check for different following column names  
        for col in ['FOLLOWING_COUNT', 'FOLLOWED']:
            if col in df_features.columns:
                following_col = col
                break
        
        if followers_col and following_col:
            df_features['FOLLOWERS_FOLLOWING_RATIO'] = np.where(
                df_features[following_col] > 0,
                df_features[followers_col] / df_features[following_col],
                0
            )
        
        # Account age and activity features
        if 'JOINED_TWITTER_DATE' in df_features.columns and 'TWEET_DATE' in df_features.columns:
            try:
                # Calculate account age in days
                df_features['ACCOUNT_AGE_DAYS'] = (
                    df_features['TWEET_DATE'] - df_features['JOINED_TWITTER_DATE']
                ).dt.days.fillna(0)
                
                # Calculate tweets per day (activity rate)
                if 'TWEET_COUNT' in df_features.columns:
                    df_features['TWEETS_PER_DAY'] = np.where(
                        df_features['ACCOUNT_AGE_DAYS'] > 0,
                        df_features['TWEET_COUNT'] / df_features['ACCOUNT_AGE_DAYS'],
                        0
                    )
            except Exception as e:
                self.logger.warning(f"Could not calculate account age features: {e}")
                df_features['ACCOUNT_AGE_DAYS'] = 0
                df_features['TWEETS_PER_DAY'] = 0
        
        # Network centrality normalization
        network_features = [
            'DEGREE', 'IN_DEGREE', 'OUT_DEGREE', 'BETWEENNESS_CENTRALITY', 
            'CLOSENESS_CENTRALITY', 'EIGENVECTOR_CENTRALITY', 'PAGERANK', 'CLUSTERING_COEFFICIENT'
        ]
        
        # Create normalized network features
        for feature in network_features:
            if feature in df_features.columns:
                # Create a normalized version (0-1 scale)
                col_values = df_features[feature]
                if col_values.max() > 0:
                    df_features[f'{feature}_NORMALIZED'] = col_values / col_values.max()
                else:
                    df_features[f'{feature}_NORMALIZED'] = 0
        
        # Verification status consolidation
        verification_features = []
        if 'VERIFIED' in df_features.columns:
            verification_features.append('VERIFIED')
        if 'IS_BLUE_VERIFIED' in df_features.columns:
            verification_features.append('IS_BLUE_VERIFIED')
        
        # Create combined verification score
        if verification_features:
            df_features['IS_VERIFIED'] = df_features[verification_features].max(axis=1)
        else:
            df_features['IS_VERIFIED'] = 0
        
        # Content type classification
        if 'RELATIONSHIP' in df_features.columns:
            # Create binary features for different relationship types
            relationship_types = df_features['RELATIONSHIP'].unique()
            for rel_type in relationship_types:
                if pd.notna(rel_type):
                    df_features[f'IS_{str(rel_type).upper()}'] = (
                        df_features['RELATIONSHIP'] == rel_type
                    ).astype(int)
        
        # Language classification features
        if 'LANGUAGE' in df_features.columns:
            # Create binary features for major languages
            major_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ar', 'hi', 'ja', 'ko', 'zh']
            for lang in major_languages:
                df_features[f'LANG_{lang.upper()}'] = (
                    df_features['LANGUAGE'] == lang
                ).astype(int)
            
            # Create feature for non-major languages
            df_features['LANG_OTHER'] = (~df_features['LANGUAGE'].isin(major_languages)).astype(int)
        
        # Media content indicators
        if 'MEDIA' in df_features.columns:
            df_features['HAS_MEDIA'] = (
                df_features['MEDIA'].notna() & 
                (df_features['MEDIA'].astype(str).str.strip() != '')
            ).astype(int)
        
        # URL content indicators
        if 'URLS' in df_features.columns:
            df_features['HAS_URLS'] = (
                df_features['URLS'].notna() & 
                (df_features['URLS'].astype(str).str.strip() != '')
            ).astype(int)
        
        # Social media engagement indicators
        if 'HASHTAGS' in df_features.columns:
            df_features['HASHTAG_COUNT'] = df_features['HASHTAGS'].astype(str).str.count('#')
        

        if 'MENTIONS' in df_features.columns:
            df_features['MENTION_COUNT'] = df_features['MENTIONS'].astype(str).str.count('@')
        
        # Final cleanup - handle any remaining NaN values
        for col in df_features.columns:
            if df_features[col].dtype in ['float64', 'int64']:
                nan_count = df_features[col].isna().sum()
                if nan_count > 0:
                    self.logger.info(f"Filling {nan_count} NaN values in feature {col} with 0")
                    df_features[col] = df_features[col].fillna(0)
            elif df_features[col].dtype == 'object':
                nan_count = df_features[col].isna().sum()
                if nan_count > 0:
                    self.logger.info(f"Filling {nan_count} NaN values in feature {col} with empty string")
                    df_features[col] = df_features[col].fillna('')
        
        # Ensure all numeric columns are properly typed
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'LABEL':  # Don't modify the label column
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
        
        # Final verification - ensure no NaN values remain
        final_nan_check = df_features.isna().sum().sum()
        if final_nan_check > 0:
            self.logger.warning(f"Warning: {final_nan_check} NaN values detected after feature creation")
            # Remove any rows that still have NaN values
            initial_rows = len(df_features)
            df_features = df_features.dropna()
            removed_rows = initial_rows - len(df_features)
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} rows with remaining NaN values")
        
        return df_features
    
    def _get_sentiment_scores(self, text):
        """Get sentiment scores for text."""
        if pd.isna(text) or text == '':
            return {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
        
        try:
            return self.sentiment_analyzer.polarity_scores(str(text))
        except Exception:
            return {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
    
    def get_dataset_summary(self, dataset_name):
        """Get summary statistics for a processed dataset."""
        try:
            processed_path = Path('datasets') / dataset_name / 'processed' / 'processed_data.csv'
            
            if not processed_path.exists():
                return None
            
            df = pd.read_csv(processed_path)
            
            summary = {
                'total_samples': len(df),
                'features': df.shape[1],
                'misinformation_count': int(df['LABEL'].sum()) if 'LABEL' in df.columns else 0,
                'misinformation_rate': float(df['LABEL'].mean() * 100) if 'LABEL' in df.columns else 0,
                'missing_values': df.isnull().sum().sum(),
                'text_stats': {
                    'avg_text_length': df['TEXT_LENGTH'].mean() if 'TEXT_LENGTH' in df.columns else 0,
                    'avg_word_count': df['WORD_COUNT'].mean() if 'WORD_COUNT' in df.columns else 0,
                    'avg_readability': df['READABILITY_SCORE'].mean() if 'READABILITY_SCORE' in df.columns else 0
                },
                'sentiment_stats': {
                    'avg_positive': df['SENTIMENT_POSITIVE'].mean() if 'SENTIMENT_POSITIVE' in df.columns else 0,
                    'avg_negative': df['SENTIMENT_NEGATIVE'].mean() if 'SENTIMENT_NEGATIVE' in df.columns else 0,
                    'avg_compound': df['SENTIMENT_COMPOUND'].mean() if 'SENTIMENT_COMPOUND' in df.columns else 0
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting dataset summary: {e}")
            return None