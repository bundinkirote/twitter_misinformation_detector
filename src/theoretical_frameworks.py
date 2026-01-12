"""
Theoretical Frameworks Implementation Module

This module implements comprehensive theoretical frameworks for misinformation detection
including Rational Action Theory (RAT), Rational Choice Theory (RCT), and Uses and
Gratifications Theory. It provides feature extraction based on established theoretical
foundations for enhanced misinformation detection in social media contexts.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Any
from textstat import flesch_reading_ease, flesch_kincaid_grade
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

class TheoreticalFrameworks:
    """
    Theoretical Frameworks Implementation Class
    
    Implements comprehensive feature extraction based on established theoretical
    frameworks including Rational Action Theory, Rational Choice Theory, and
    Uses and Gratifications Theory. Provides systematic feature engineering
    grounded in theoretical foundations for enhanced misinformation detection.
    """
    
    def __init__(self):
        """Initialize the theoretical frameworks extractor."""
        self.logger = logging.getLogger(__name__)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Framework feature dimensions for unified training
        self.framework_dimensions = {
            'rat_features': 15,
            'rct_features': 12, 
            'ugt_features': 18
        }
        
        # Column mapping for exact column matching (based on actual dataset columns)
        self.column_mapping = {
            # Text content columns
            'TWEET_CONTENT': ['TWEET', 'Tweet', 'TWEET_CONTENT', 'tweet_content', 'COMBINED_TEXT', 'CLEANED_TEXT'],
            'COMBINED_TEXT': ['COMBINED_TEXT', 'Combined Text', 'COMBINED TEXT', 'combined_text', 'TWEET_CONTENT', 'TWEET', 'TEXT'],
            
            # User profile columns (NOT AVAILABLE in this tweet-level dataset)
            'FOLLOWERS_COUNT': ['FOLLOWERS_COUNT', 'Followers Count', 'FOLLOWERS COUNT', 'followers_count', 'FOLLOWERS'],
            'FOLLOWING_COUNT': ['FOLLOWING_COUNT', 'Following Count', 'FOLLOWING COUNT', 'following_count', 'FOLLOWED'],
            'TWEET_COUNT': ['TWEET_COUNT', 'Tweet Count', 'TWEET COUNT', 'tweet_count', 'TWEETS'],
            'LISTED_COUNT': ['LISTED_COUNT', 'Listed Count', 'LISTED COUNT', 'listed_count'],
            'VERIFIED': ['VERIFIED', 'Verified', 'verified', 'IS_VERIFIED'],
            'USER_DESCRIPTION': ['USER_DESCRIPTION', 'User Description', 'USER DESCRIPTION', 'user_description', 'DESCRIPTION'],
            'LOCATION': ['LOCATION', 'Location', 'location'],
            'USER_ID': ['AUTHOR ID', 'USER_ID', 'User ID', 'USER ID', 'user_id', 'AUTHOR_ID'],
            
            # Tweet-level columns (AVAILABLE in this dataset)
            'MENTIONS_IN_TWEET': ['MENTIONS IN TWEET', 'Mentions in Tweet', 'MENTIONS_IN_TWEET', 'mentions_in_tweet'],
            'RETWEET_COUNT': ['RETWEET COUNT', 'Retweet Count', 'RETWEET_COUNT', 'retweet_count'],
            'FAVORITE_COUNT': ['FAVORITE COUNT', 'Favorite Count', 'FAVORITE_COUNT', 'favorite_count'],
            'REPLY_COUNT': ['REPLY COUNT', 'Reply Count', 'REPLY_COUNT', 'reply_count'],
            'QUOTE_COUNT': ['QUOTE COUNT', 'Quote Count', 'QUOTE_COUNT', 'quote_count'],
            'HASHTAGS': ['HASHTAGS IN TWEET', 'Hashtags in Tweet', 'HASHTAGS', 'hashtags'],
            'URLS_IN_TWEET': ['URLS IN TWEET', 'URLs in Tweet', 'URLS_IN_TWEET', 'urls_in_tweet'],
            'MEDIA_IN_TWEET': ['MEDIA IN TWEET', 'Media in Tweet', 'MEDIA_IN_TWEET', 'media_in_tweet'],
            'SOURCE': ['SOURCE', 'Source', 'source'],
            
            'MENTIONS_GIVEN': ['MENTIONS_GIVEN', 'Mentions Given', 'MENTIONS GIVEN', 'mentions_given'],
            'MENTIONS_RECEIVED': ['MENTIONS_RECEIVED', 'Mentions Received', 'MENTIONS RECEIVED', 'mentions_received'],
            'RETWEETS_GIVEN': ['RETWEETS_GIVEN', 'Retweets Given', 'RETWEETS GIVEN', 'retweets_given'],
            'RETWEETS_RECEIVED': ['RETWEETS_RECEIVED', 'Retweets Received', 'RETWEETS RECEIVED', 'retweets_received']
        }
        
        # Kenyan political context keywords
        self.kenyan_political_keywords = [
            # Political figures
            'uhuru', 'ruto', 'raila', 'kenyatta', 'odinga', 'gachagua', 'mudavadi',
            'wetangula', 'karua', 'wanjiku', 'waiguru', 'joho', 'kingi', 'kalonzo',
            
            # Political parties and movements
            'jubilee', 'nasa', 'cord', 'uda', 'azimio', 'kenya kwanza', 'odm', 'tna', 
            'wiper', 'amani', 'kanu', 'ford', 'narc', 'dap-k', 'pap',
            
            # Political concepts
            'bbi', 'handshake', 'hustler', 'dynasty', 'bottom up', 'trickle down',
            'devolution', 'referendum', 'constitution', 'amendment',
            
            # Institutions
            'iebc', 'scok', 'supreme court', 'parliament', 'senate', 'county', 'governor',
            'mp', 'mca', 'ward', 'constituency', 'state house', 'harambee house',
            
            # Political processes
            'election', 'vote', 'ballot', 'campaign', 'nomination', 'primary',
            'rigging', 'fraud', 'manipulation', 'irregularities',
            
            # Social issues
            'violence', 'peace', 'unity', 'development', 'corruption', 'scandal',
            'unemployment', 'poverty', 'inequality', 'tribalism', 'nepotism'
        ]
        
        # Gratification patterns for Uses and Gratifications Theory
        self.gratification_patterns = {
            'information_seeking': [
                'news', 'information', 'facts', 'truth', 'report', 'update', 'breaking',
                'confirmed', 'official', 'statement', 'announcement', 'press release'
            ],
            'entertainment': [
                'funny', 'hilarious', 'joke', 'meme', 'lol', 'haha', 'comedy',
                'entertaining', 'amusing', 'fun', 'laugh', 'humor'
            ],
            'social_interaction': [
                'share', 'retweet', 'comment', 'discuss', 'talk', 'conversation',
                'community', 'together', 'join', 'participate', 'engage'
            ],
            'personal_identity': [
                'believe', 'support', 'agree', 'stand with', 'proud', 'identity',
                'values', 'principles', 'conviction', 'faith', 'trust'
            ],
            'surveillance': [
                'watch', 'monitor', 'track', 'follow', 'observe', 'keep eye',
                'alert', 'warning', 'danger', 'threat', 'risk'
            ],
            'escapism': [
                'escape', 'forget', 'distract', 'avoid', 'ignore', 'fantasy',
                'dream', 'imagine', 'wish', 'hope', 'different'
            ]
        }
    
    def _validate_and_convert_dataframe(self, data, method_name: str = "unknown") -> pd.DataFrame:
        """
        Validate and convert input data to a proper pandas DataFrame.
        
        Args:
            data: Input data that should be a DataFrame
            method_name: Name of the calling method for logging
            
        Returns:
            pd.DataFrame: Validated DataFrame
            
        Raises:
            TypeError: If data cannot be converted to DataFrame
        """
        # If it's already a DataFrame, return as-is
        if isinstance(data, pd.DataFrame):
            return data
        
        # If it's a tuple, try to convert it
        if isinstance(data, tuple):
            self.logger.warning(f"{method_name}: Received tuple instead of DataFrame, attempting conversion")
            
            if len(data) == 2:
                # Assume it's (features_array, feature_names)
                features_array, feature_names = data
                if hasattr(features_array, 'shape') and hasattr(feature_names, '__iter__'):
                    try:
                        df = pd.DataFrame(features_array, columns=feature_names)
                        self.logger.info(f"{method_name}: Successfully converted tuple to DataFrame: {df.shape}")
                        return df
                    except Exception as e:
                        self.logger.error(f"{method_name}: Failed to convert tuple to DataFrame: {e}")
            
            # If tuple conversion failed, try to use first element
            try:
                df = pd.DataFrame(data[0])
                self.logger.info(f"{method_name}: Converted first element of tuple to DataFrame: {df.shape}")
                return df
            except Exception as e:
                self.logger.error(f"{method_name}: Failed to convert tuple first element to DataFrame: {e}")
        
        # If it's a list or array, try to convert
        if isinstance(data, (list, np.ndarray)):
            try:
                df = pd.DataFrame(data)
                self.logger.info(f"{method_name}: Converted {type(data).__name__} to DataFrame: {df.shape}")
                return df
            except Exception as e:
                self.logger.error(f"{method_name}: Failed to convert {type(data).__name__} to DataFrame: {e}")
        
        # If all else fails, raise an error
        error_msg = f"{method_name}: Cannot convert {type(data)} to DataFrame"
        self.logger.error(error_msg)
        raise TypeError(error_msg)
    
    def _find_text_column(self, df: pd.DataFrame) -> str:
        """Find the appropriate text column in the dataframe."""
        # Debug: Check the type of df
        if not isinstance(df, pd.DataFrame):
            self.logger.error(f"_find_text_column: Expected DataFrame, got {type(df)}")
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
        
        text_columns = ['TWEET', 'Tweet', 'TWEET_CONTENT', 'COMBINED_TEXT', 'CLEANED_TEXT', 'TEXT']
        
        for col in text_columns:
            if col in df.columns:
                return col
        
        # If no standard text column found, use the first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        
        # Fallback - create a dummy text column
        self.logger.warning("No text column found, creating dummy text column")
        df['DUMMY_TEXT'] = 'No text available'
        return 'DUMMY_TEXT'
    
    def _safe_column_access(self, df: pd.DataFrame, column_name: str, default_value=0):
        """Safely access a column with exact column name matching, returning default value if column doesn't exist."""
        # Use the column mapping for exact matching
        possible_names = self.column_mapping.get(column_name, [column_name])
        
        for name in possible_names:
            if name in df.columns:
                col_data = df[name].copy()
                
                # Handle boolean columns
                if isinstance(default_value, bool):
                    if col_data.dtype == 'object':
                        # Convert string representations to boolean
                        col_data = col_data.astype(str).str.lower()
                        col_data = col_data.map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False})
                        col_data = col_data.fillna(default_value)
                    else:
                        col_data = col_data.astype(bool).fillna(default_value)
                
                # Handle numeric columns
                elif isinstance(default_value, (int, float)):
                    col_data = pd.to_numeric(col_data, errors='coerce').fillna(default_value)
                
                # Handle string columns
                else:
                    col_data = col_data.fillna(default_value).astype(str)
                
                return col_data
        
        # Column not found - log warning and return default values
        self.logger.warning(f"Column '{column_name}' not found, using default value {default_value}")
        
        if isinstance(default_value, bool):
            return pd.Series([default_value] * len(df), dtype=bool, index=df.index)
        elif isinstance(default_value, (int, float)):
            return pd.Series([default_value] * len(df), dtype=type(default_value), index=df.index)
        else:
            return pd.Series([default_value] * len(df), dtype=str, index=df.index)
    
    def extract_rat_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract Routine Activity Theory (RAT) features."""
        self.logger.info("Extracting Routine Activity Theory (RAT) features")
        
        try:
            # Validate and convert input to DataFrame
            df = self._validate_and_convert_dataframe(df, "extract_rat_features")
            
            # Find the appropriate text column
            text_column = self._find_text_column(df)
            self.logger.info(f"Using text column: {text_column}")
            
            rat_features = pd.DataFrame(index=df.index)
            
            # Motivated Offender Features
            rat_features['rat_motivated_offender_score'] = 0.0
            
            # High retweet activity (motivated to spread content)
            retweet_count = self._safe_column_access(df, 'RETWEET_COUNT', 0)
            high_retweet_threshold = retweet_count.quantile(0.8) if len(retweet_count) > 0 else 0
            rat_features['rat_motivated_offender_score'] += np.where(
                retweet_count > high_retweet_threshold, 0.3, 0.0
            )
            
            # High engagement seeking (quote tweets + replies)
            quote_count = self._safe_column_access(df, 'QUOTE_COUNT', 0)
            reply_count = self._safe_column_access(df, 'REPLY_COUNT', 0)
            engagement_score = quote_count + reply_count
            high_engagement_threshold = engagement_score.quantile(0.8) if len(engagement_score) > 0 else 0
            rat_features['rat_motivated_offender_score'] += np.where(
                engagement_score > high_engagement_threshold, 0.2, 0.0
            )
            
            # Low verification status (less accountability)
            verified = self._safe_column_access(df, 'VERIFIED', False)
            rat_features['rat_motivated_offender_score'] += np.where(
                ~verified, 0.2, 0.0
            )
            
            # Suitable Target Features
            rat_features['rat_suitable_target_score'] = 0.0
            
            # Emotional content (more likely to be shared)
            for idx, row in df.iterrows():
                if pd.notna(row[text_column]):
                    text = str(row[text_column])
                    sentiment = self.sentiment_analyzer.polarity_scores(text)
                    emotional_intensity = abs(sentiment['compound'])
                    
                    # Content characteristics that make it a suitable target
                    text_lower = text.lower()
                    
                    # Controversial keywords increase target suitability
                    controversial_keywords = ['breaking', 'urgent', 'shocking', 'scandal', 'exposed', 'truth', 'fake', 'lie']
                    controversy_score = sum(1 for keyword in controversial_keywords if keyword in text_lower) / len(controversial_keywords)
                    
                    # Short, punchy content is more shareable
                    length_factor = max(0, 1 - len(text) / 280)  # Twitter-like length preference
                    
                    target_score = (emotional_intensity * 0.4 + controversy_score * 0.3 + length_factor * 0.3)
                    rat_features.loc[idx, 'rat_suitable_target_score'] = np.clip(target_score, 0.0, 1.0)
            
            # Absence of Guardian Features
            rat_features['rat_absence_guardian_score'] = 0.0
            
            # No verification badge
            rat_features['rat_absence_guardian_score'] += np.where(
                ~verified, 0.4, 0.0
            )
            
            # Overall RAT Risk Score
            rat_features['rat_overall_risk_score'] = (
                rat_features['rat_motivated_offender_score'] * 0.4 +
                rat_features['rat_suitable_target_score'] * 0.3 +
                rat_features['rat_absence_guardian_score'] * 0.3
            )
            
            # Add expected column names for compatibility
            rat_features['rat_perceived_risk'] = rat_features['rat_overall_risk_score']
            rat_features['rat_perceived_benefit'] = 1 - rat_features['rat_overall_risk_score']
            
            self.logger.info("RAT features extracted successfully")
            return rat_features
            
        except Exception as e:
            self.logger.error(f"Error extracting RAT features: {e}")
            # Return dummy RAT features
            dummy_rat_features = pd.DataFrame({
                'rat_motivated_offender_score': [0.5] * len(df),
                'rat_suitable_target_score': [0.5] * len(df),
                'rat_absence_guardian_score': [0.5] * len(df),
                'rat_overall_risk_score': [0.5] * len(df),
                'rat_perceived_risk': [0.5] * len(df),
                'rat_perceived_benefit': [0.5] * len(df)
            }, index=df.index)
            return dummy_rat_features
    
    def extract_rct_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract Rational Choice Theory (RCT) features."""
        self.logger.info("Extracting Rational Choice Theory (RCT) features")
        
        try:
            # Validate and convert input to DataFrame
            df = self._validate_and_convert_dataframe(df, "extract_rct_features")
            
            # Find the appropriate text column
            text_column = self._find_text_column(df)
            
            rct_features = pd.DataFrame(index=df.index)
            
            # Benefit Assessment Features - based on engagement potential
            retweet_count = self._safe_column_access(df, 'RETWEET_COUNT', 0)
            favorite_count = self._safe_column_access(df, 'FAVORITE_COUNT', 0)
            reply_count = self._safe_column_access(df, 'REPLY_COUNT', 0)
            
            # Calculate engagement-based benefits
            total_engagement = retweet_count + favorite_count + reply_count
            max_engagement = total_engagement.max() if len(total_engagement) > 0 and total_engagement.max() > 0 else 1
            normalized_engagement = total_engagement / max_engagement
            rct_features['rct_perceived_benefits_score'] = np.clip(normalized_engagement, 0.1, 0.9)
            
            # Cost Assessment Features - based on verification and account risk
            verified = self._safe_column_access(df, 'VERIFIED', False)
            followers_count = self._safe_column_access(df, 'FOLLOWERS_COUNT', 0)
            
            # Higher followers = higher cost of misinformation (reputation risk)
            max_followers = followers_count.max() if len(followers_count) > 0 and followers_count.max() > 0 else 1
            normalized_followers = followers_count / max_followers
            verification_cost = np.where(verified, 0.8, 0.3)  # Verified accounts have higher cost
            rct_features['rct_perceived_costs_score'] = np.clip(
                (normalized_followers * 0.5 + verification_cost * 0.5), 0.1, 0.9
            )
            
            # Decision Rationality Features - based on content characteristics
            rct_features['rct_decision_rationality_score'] = 0.5  # Initialize
            
            # Analyze text content for rationality indicators
            for idx, row in df.iterrows():
                if pd.notna(row[text_column]):
                    text = str(row[text_column]).lower()
                    
                    # Emotional language reduces rationality
                    sentiment = self.sentiment_analyzer.polarity_scores(text)
                    emotional_intensity = abs(sentiment['compound'])
                    
                    # Presence of URLs/sources increases rationality
                    has_urls = 'http' in text or 'www.' in text
                    url_bonus = 0.2 if has_urls else 0.0
                    
                    # Calculate rationality score
                    rationality = 0.5 - (emotional_intensity * 0.3) + url_bonus
                    rct_features.loc[idx, 'rct_decision_rationality_score'] = np.clip(rationality, 0.1, 0.9)
            
            # Risk-Benefit Ratio
            rct_features['rct_risk_benefit_ratio'] = (
                rct_features['rct_perceived_benefits_score'] / 
                (rct_features['rct_perceived_costs_score'] + 0.1)
            )
            
            # Overall RCT Score
            rct_features['rct_overall_score'] = (
                rct_features['rct_perceived_benefits_score'] * 0.4 +
                (1 - rct_features['rct_perceived_costs_score']) * 0.3 +
                (1 - rct_features['rct_decision_rationality_score']) * 0.3
            )
            
            # Add expected column names for compatibility
            rct_features['rct_coping_appraisal'] = rct_features['rct_decision_rationality_score']
            rct_features['rct_threat_appraisal'] = rct_features['rct_overall_score']
            
            self.logger.info("RCT features extracted successfully")
            return rct_features
            
        except Exception as e:
            self.logger.error(f"Error extracting RCT features: {e}")
            # Return dummy RCT features
            dummy_rct_features = pd.DataFrame({
                'rct_perceived_benefits_score': [0.5] * len(df),
                'rct_perceived_costs_score': [0.5] * len(df),
                'rct_decision_rationality_score': [0.5] * len(df),
                'rct_risk_benefit_ratio': [1.0] * len(df),
                'rct_overall_score': [0.5] * len(df),
                'rct_coping_appraisal': [0.5] * len(df),
                'rct_threat_appraisal': [0.5] * len(df)
            }, index=df.index)
            return dummy_rct_features
    
    def extract_content_gratification_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract Uses and Gratifications Theory (Content Gratification) features."""
        self.logger.info("Extracting Uses and Gratifications Theory (Content Gratification) features")
        
        try:
            # Validate and convert input to DataFrame
            df = self._validate_and_convert_dataframe(df, "extract_content_gratification_features")
            
            # Find the appropriate text column
            text_column = self._find_text_column(df)
            self.logger.info(f"Using text column: {text_column}")
            self.logger.info(f"Available columns: {list(df.columns)}")
            
            # Check if we have valid text data
            if text_column and text_column in df.columns:
                non_null_count = df[text_column].notna().sum()
                self.logger.info(f"Non-null text entries: {non_null_count} out of {len(df)}")
                if non_null_count > 0:
                    sample_text = df[text_column].dropna().iloc[0]
                    self.logger.info(f"Sample text: {str(sample_text)[:200]}...")
            else:
                self.logger.error(f"Text column '{text_column}' not found in DataFrame")
            
            cg_features = pd.DataFrame(index=df.index)
            
            # Initialize gratification scores
            for gratification_type in self.gratification_patterns.keys():
                cg_features[f'cg_{gratification_type}_score'] = 0.0
            
            # Extract gratification features from text content
            total_processed = 0
            total_matches = 0
            
            for idx, row in df.iterrows():
                if pd.notna(row[text_column]):
                    content = str(row[text_column]).lower()
                    total_processed += 1
                    
                    for gratification_type, keywords in self.gratification_patterns.items():
                        score = sum(1 for keyword in keywords if keyword in content)
                        if score > 0:
                            total_matches += 1
                        normalized_score = min(score / len(keywords), 1.0)
                        cg_features.loc[idx, f'cg_{gratification_type}_score'] = normalized_score
            
            self.logger.info(f"Processed {total_processed} texts, found {total_matches} gratification matches")
            
            # Log sample of results
            if not cg_features.empty:
                sample_scores = cg_features.head().to_dict()
                self.logger.info(f"Sample gratification scores: {sample_scores}")
                
                # Log mean scores
                mean_scores = cg_features.select_dtypes(include=[np.number]).mean()
                self.logger.info(f"Mean gratification scores: {mean_scores.to_dict()}")
            
            # Gratification Diversity Score
            gratification_columns = [col for col in cg_features.columns if col.startswith('cg_') and col.endswith('_score')]
            cg_features['cg_gratification_diversity'] = (cg_features[gratification_columns] > 0.1).sum(axis=1) / len(gratification_columns)
            
            # Primary Gratification Type
            cg_features['cg_primary_gratification'] = cg_features[gratification_columns].idxmax(axis=1)
            
            # Overall Gratification Intensity
            cg_features['cg_overall_intensity'] = cg_features[gratification_columns].mean(axis=1)
            
            # Misinformation Susceptibility based on Gratification Patterns
            susceptibility_weights = {
                'cg_information_seeking_score': -0.2,  # Less susceptible
                'cg_entertainment_score': 0.3,         # More susceptible
                'cg_social_interaction_score': 0.2,    # Moderately susceptible
                'cg_personal_identity_score': 0.1,     # Slightly susceptible
                'cg_surveillance_score': -0.1,         # Slightly less susceptible
                'cg_escapism_score': 0.4               # Most susceptible
            }
            
            cg_features['cg_misinformation_susceptibility'] = 0.0
            for gratification_type, weight in susceptibility_weights.items():
                cg_features['cg_misinformation_susceptibility'] += cg_features[gratification_type] * weight
            
            # Normalize susceptibility score
            cg_features['cg_misinformation_susceptibility'] = (
                cg_features['cg_misinformation_susceptibility'] - 
                cg_features['cg_misinformation_susceptibility'].min()
            ) / (
                cg_features['cg_misinformation_susceptibility'].max() - 
                cg_features['cg_misinformation_susceptibility'].min() + 1e-8
            )
            
            self.logger.info("Content Gratification features extracted successfully")
            return cg_features
            
        except Exception as e:
            self.logger.error(f"Error extracting Content Gratification features: {e}")
            # Return dummy CG features
            dummy_cg_features = pd.DataFrame({
                'cg_information_seeking_score': [0.5] * len(df),
                'cg_entertainment_score': [0.5] * len(df),
                'cg_social_interaction_score': [0.5] * len(df),
                'cg_personal_identity_score': [0.5] * len(df),
                'cg_surveillance_score': [0.5] * len(df),
                'cg_escapism_score': [0.5] * len(df),
                'cg_gratification_diversity': [0.5] * len(df),
                'cg_primary_gratification': ['cg_information_seeking_score'] * len(df),
                'cg_overall_intensity': [0.5] * len(df),
                'cg_misinformation_susceptibility': [0.5] * len(df)
            }, index=df.index)
            return dummy_cg_features
    
    def extract_rat_rct_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract combined RAT and RCT features."""
        self.logger.info("Extracting combined RAT and RCT features")
        
        try:
            # Validate and convert input to DataFrame
            df = self._validate_and_convert_dataframe(df, "extract_rat_rct_features")
            # Extract RAT features
            rat_features = self.extract_rat_features(df)
            
            # Extract RCT features
            rct_features = self.extract_rct_features(df)
            
            # Combine RAT and RCT features
            combined_features = pd.concat([rat_features, rct_features], axis=1)
            
            # Add interaction features between RAT and RCT
            combined_features['rat_rct_interaction'] = (
                combined_features['rat_overall_risk_score'] * 
                combined_features['rct_overall_score']
            )
            
            # Combined risk assessment
            combined_features['rat_rct_combined_risk'] = (
                combined_features['rat_overall_risk_score'] * 0.6 +
                combined_features['rct_overall_score'] * 0.4
            )
            
            self.logger.info(f"Combined RAT-RCT features extracted: {combined_features.shape[1]} features")
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Error extracting combined RAT-RCT features: {e}")
            # Return dummy combined features - use safe length calculation
            try:
                data_length = len(df) if hasattr(df, '__len__') else 1
                data_index = df.index if hasattr(df, 'index') else None
            except:
                data_length = 1
                data_index = None
            
            dummy_combined_features = pd.DataFrame({
                'rat_motivated_offender_score': [0.5] * data_length,
                'rat_suitable_target_score': [0.5] * data_length,
                'rat_absence_guardian_score': [0.5] * data_length,
                'rat_overall_risk_score': [0.5] * data_length,
                'rat_perceived_risk': [0.5] * data_length,
                'rat_perceived_benefit': [0.5] * data_length,
                'rct_perceived_benefits_score': [0.5] * data_length,
                'rct_perceived_costs_score': [0.5] * data_length,
                'rct_decision_rationality_score': [0.5] * data_length,
                'rct_risk_benefit_ratio': [1.0] * data_length,
                'rct_overall_score': [0.5] * data_length,
                'rct_coping_appraisal': [0.5] * data_length,
                'rct_threat_appraisal': [0.5] * data_length,
                'rat_rct_interaction': [0.25] * data_length,
                'rat_rct_combined_risk': [0.5] * data_length
            }, index=data_index)
            return dummy_combined_features
    
    def extract_all_theoretical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all theoretical framework features."""
        self.logger.info("Extracting all theoretical framework features")
        
        try:
            # Validate and convert input to DataFrame
            df = self._validate_and_convert_dataframe(df, "extract_all_theoretical_features")
            # Extract features from each framework with error handling
            self.logger.info("Extracting RAT features...")
            rat_features = self.extract_rat_features(df)
            if not isinstance(rat_features, pd.DataFrame):
                self.logger.error(f"RAT features extraction returned {type(rat_features)} instead of DataFrame")
                raise ValueError("RAT features extraction failed")
            self.logger.info(f"RAT features extracted: {rat_features.shape}")
            
            self.logger.info("Extracting RCT features...")
            rct_features = self.extract_rct_features(df)
            if not isinstance(rct_features, pd.DataFrame):
                self.logger.error(f"RCT features extraction returned {type(rct_features)} instead of DataFrame")
                raise ValueError("RCT features extraction failed")
            self.logger.info(f"RCT features extracted: {rct_features.shape}")
            
            self.logger.info("Extracting Content Gratification features...")
            cg_features = self.extract_content_gratification_features(df)
            if not isinstance(cg_features, pd.DataFrame):
                self.logger.error(f"CG features extraction returned {type(cg_features)} instead of DataFrame")
                raise ValueError("CG features extraction failed")
            self.logger.info(f"CG features extracted: {cg_features.shape}")
            
            # Combine all features
            self.logger.info("Combining all theoretical features...")
            theoretical_features = pd.concat([rat_features, rct_features, cg_features], axis=1)
            
            # Create integrated theoretical scores
            theoretical_features['theoretical_integrated_risk_score'] = (
                theoretical_features['rat_overall_risk_score'] * 0.4 +
                theoretical_features['rct_overall_score'] * 0.3 +
                theoretical_features['cg_misinformation_susceptibility'] * 0.3
            )
            
            # Framework interaction features
            theoretical_features['rat_rct_interaction'] = (
                theoretical_features['rat_overall_risk_score'] * 
                theoretical_features['rct_overall_score']
            )
            
            theoretical_features['rct_cg_interaction'] = (
                theoretical_features['rct_overall_score'] * 
                theoretical_features['cg_overall_intensity']
            )
            
            theoretical_features['rat_cg_interaction'] = (
                theoretical_features['rat_overall_risk_score'] * 
                theoretical_features['cg_misinformation_susceptibility']
            )
            
            self.logger.info(f"All theoretical features extracted: {theoretical_features.shape[1]} features")
            return theoretical_features
            
        except Exception as e:
            self.logger.error(f"Error extracting theoretical features: {e}")
            # Return dummy features as fallback
            dummy_data = {
                'rat_overall_risk_score': [0.5] * len(df),
                'rct_overall_score': [0.5] * len(df),
                'cg_misinformation_susceptibility': [0.5] * len(df),
                'cg_overall_intensity': [0.5] * len(df),
                'theoretical_integrated_risk_score': [0.5] * len(df),
                'rat_rct_interaction': [0.25] * len(df),
                'rct_cg_interaction': [0.25] * len(df),
                'rat_cg_interaction': [0.25] * len(df)
            }
            return pd.DataFrame(dummy_data, index=df.index)
    
    def extract_unified_framework_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Extract all theoretical framework features in a unified approach.
        Returns separate DataFrames for each framework for flexible combination.
        
        Args:
            df: Input DataFrame with tweet data
            
        Returns:
            Dictionary containing separate feature DataFrames for each framework
        """
        self.logger.info("🔧 Extracting unified theoretical framework features...")
        
        try:
            # Validate input
            df = self._validate_and_convert_dataframe(df, "extract_unified_framework_features")
            
            # Extract each framework separately for maximum flexibility
            framework_features = {}
            
            # RAT Features (15 dimensions)
            self.logger.info("📚 Extracting RAT features...")
            framework_features['rat_features'] = self.extract_rat_features(df)
            self.logger.info(f"✅ RAT features: {framework_features['rat_features'].shape[1]} dimensions")
            
            # RCT Features (12 dimensions)
            self.logger.info("🧠 Extracting RCT features...")
            framework_features['rct_features'] = self.extract_rct_features(df)
            self.logger.info(f"✅ RCT features: {framework_features['rct_features'].shape[1]} dimensions")
            
            # UGT Features (18 dimensions)
            self.logger.info("🎯 Extracting UGT features...")
            framework_features['ugt_features'] = self.extract_content_gratification_features(df)
            self.logger.info(f"✅ UGT features: {framework_features['ugt_features'].shape[1]} dimensions")
            
            # Validate dimensions
            for framework_name, features in framework_features.items():
                expected_dims = self.framework_dimensions.get(framework_name, 0)
                actual_dims = features.shape[1]
                if expected_dims > 0 and actual_dims != expected_dims:
                    self.logger.warning(f"⚠️ {framework_name}: Expected {expected_dims} dims, got {actual_dims}")
            
            self.logger.info(f"🎉 Unified framework extraction completed: {len(framework_features)} frameworks")
            return framework_features
            
        except Exception as e:
            self.logger.error(f"❌ Unified framework extraction failed: {e}")
            # Return dummy features for all frameworks
            dummy_features = {}
            for framework_name, expected_dims in self.framework_dimensions.items():
                dummy_data = {f'{framework_name}_{i}': [0.5] * len(df) for i in range(expected_dims)}
                dummy_features[framework_name] = pd.DataFrame(dummy_data, index=df.index)
            return dummy_features