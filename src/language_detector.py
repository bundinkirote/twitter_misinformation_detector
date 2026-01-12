"""
Language Detection and Transformer Selection Module

This module provides comprehensive language detection capabilities with specialized
support for multilingual content analysis. It implements language-adaptive transformer
routing, model selection strategies, and multilingual processing for optimal performance
in misinformation detection across different linguistic contexts.
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
from pathlib import Path

# Language detection imports
try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Transformers imports
try:
    import warnings
    # Suppress transformers warnings about unused weights
    warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
    
    from transformers import AutoTokenizer, AutoModel, pipeline
    from transformers import logging as transformers_logging
    # Set transformers logging to ERROR to suppress warnings
    transformers_logging.set_verbosity_error()
    
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class LanguageDetector:
    """
    Language Detection and Transformer Selection Class
    
    Implements comprehensive language detection and adaptive transformer routing
    for multilingual content analysis. Provides specialized detection for English
    and Kiswahili content with appropriate transformer model selection for optimal
    performance in multilingual misinformation detection tasks.
    """
    
    def __init__(self, file_manager):
        self.logger = logging.getLogger(__name__)
        self.file_manager = file_manager
        
        # Language patterns for Kenyan context
        self.kiswahili_patterns = [
            r'\b(na|ya|wa|za|la|ma|ku|ni|si|tu|mu|li|ki|vi|u|i|a|e|o)\b',
            r'\b(hii|hiyo|hizo|haya|hawa|wale|yale|vile)\b',
            r'\b(sana|kabisa|tu|pia|lakini|kwa|katika|juu|chini)\b',
            r'\b(habari|mambo|poa|sawa|asante|karibu|pole|haraka)\b',
            r'\b(serikali|rais|wabunge|uchaguzi|siasa|upinzani)\b'
        ]
        
        # Transformer configurations
        self.transformer_configs = {
            'english_models': {
                'bert-base-uncased': {
                    'type': 'monolingual',
                    'languages': ['en'],
                    'use_case': 'English text embeddings',
                    'performance': 'high'
                },
                'distilbert-base-uncased': {
                    'type': 'monolingual',
                    'languages': ['en'],
                    'use_case': 'English text embeddings (faster)',
                    'performance': 'medium-high'
                }
            },
            'multilingual_models': {
                'xlm-roberta-base': {
                    'type': 'multilingual',
                    'languages': ['en', 'sw', 'mixed'],
                    'use_case': 'Multilingual text embeddings',
                    'performance': 'high'
                },
                'distilbert-base-multilingual-cased': {
                    'type': 'multilingual',
                    'languages': ['en', 'sw', 'mixed'],
                    'use_case': 'Multilingual text embeddings (faster)',
                    'performance': 'medium-high'
                }
            }
        }
        
        # Initialize models cache
        self.loaded_models = {}
        self.tokenizers = {}
        
        # Smart model manager for automatic downloading and local storage
        from .smart_model_manager import get_smart_model_manager
        self.model_manager = get_smart_model_manager()
        
        self.logger.info("Language detector initialized")
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with language detection results
        """
        if not text or len(text.strip()) < 3:
            return {
                'primary_language': 'unknown',
                'confidence': 0.0,
                'is_mixed': False,
                'languages': [],
                'method': 'insufficient_text'
            }
        
        # Clean text for analysis
        clean_text = self._clean_text_for_detection(text)
        
        # Method 1: Pattern-based detection for Kiswahili
        kiswahili_score = self._calculate_kiswahili_score(clean_text)
        
        # Method 2: Library-based detection (if available)
        library_result = None
        if LANGDETECT_AVAILABLE:
            library_result = self._detect_with_library(clean_text)
        
        # Method 3: Mixed language detection
        is_mixed = self._detect_mixed_language(clean_text)
        
        # Combine results
        result = self._combine_detection_results(
            clean_text, kiswahili_score, library_result, is_mixed
        )
        
        return result
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for language detection."""
        # Remove URLs, mentions, hashtags
        clean_text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        # Remove extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text.lower()
    
    def _calculate_kiswahili_score(self, text: str) -> float:
        """Calculate Kiswahili language score based on patterns."""
        total_matches = 0
        total_words = len(text.split())
        
        if total_words == 0:
            return 0.0
        
        for pattern in self.kiswahili_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            total_matches += matches
        
        # Normalize by text length
        score = min(total_matches / total_words, 1.0)
        return score
    
    def _detect_with_library(self, text: str) -> Optional[Dict[str, Any]]:
        """Use langdetect library for detection."""
        try:
            # Get primary language
            primary_lang = detect(text)
            
            # Get confidence scores
            lang_probs = detect_langs(text)
            
            return {
                'primary': primary_lang,
                'probabilities': [(lang.lang, lang.prob) for lang in lang_probs]
            }
        except LangDetectException:
            return None
    
    def _detect_mixed_language(self, text: str) -> bool:
        """Detect if text contains mixed languages."""
        # Look for code-switching patterns
        words = text.split()
        if len(words) < 3:
            return False
        
        # Check for English-Kiswahili mixing patterns
        english_indicators = ['the', 'and', 'is', 'are', 'this', 'that', 'but', 'for']
        kiswahili_indicators = ['na', 'ni', 'wa', 'ya', 'za', 'la', 'ma', 'ku']
        
        has_english = any(word in english_indicators for word in words)
        has_kiswahili = any(word in kiswahili_indicators for word in words)
        
        return has_english and has_kiswahili
    
    def _combine_detection_results(self, text: str, kiswahili_score: float, 
                                 library_result: Optional[Dict], is_mixed: bool) -> Dict[str, Any]:
        """Combine different detection methods."""
        
        # Determine primary language
        if is_mixed:
            primary_language = 'mixed'
            confidence = 0.8
        elif kiswahili_score > 0.3:
            primary_language = 'sw'  # Kiswahili
            confidence = min(kiswahili_score + 0.2, 0.9)
        elif library_result and library_result['primary'] == 'en':
            primary_language = 'en'  # English
            confidence = max([prob for lang, prob in library_result['probabilities'] if lang == 'en'], default=0.7)
        elif library_result and library_result['primary'] == 'sw':
            primary_language = 'sw'
            confidence = max([prob for lang, prob in library_result['probabilities'] if lang == 'sw'], default=0.7)
        else:
            # Default to English for unknown
            primary_language = 'en'
            confidence = 0.5
        
        # Build language list
        languages = []
        if library_result:
            languages = [(lang, prob) for lang, prob in library_result['probabilities']]
        else:
            if primary_language == 'mixed':
                languages = [('en', 0.5), ('sw', 0.5)]
            else:
                languages = [(primary_language, confidence)]
        
        return {
            'primary_language': primary_language,
            'confidence': confidence,
            'is_mixed': is_mixed,
            'languages': languages,
            'kiswahili_score': kiswahili_score,
            'method': 'combined',
            'text_length': len(text.split())
        }
    
    def select_transformer(self, language_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select appropriate transformer based on language detection.
        
        Args:
            language_result: Result from detect_language()
            
        Returns:
            Dictionary with transformer selection details
        """
        primary_lang = language_result['primary_language']
        is_mixed = language_result['is_mixed']
        confidence = language_result['confidence']
        
        # Selection logic
        if primary_lang == 'en' and not is_mixed and confidence > 0.7:
            # Use monolingual English model
            selected_model = 'bert-base-uncased'
            model_type = 'monolingual'
            reason = 'High confidence English text'
        elif primary_lang in ['sw', 'mixed'] or is_mixed or confidence < 0.7:
            # Use multilingual model
            selected_model = 'xlm-roberta-base'
            model_type = 'multilingual'
            reason = 'Kiswahili, mixed, or uncertain language'
        else:
            # Default to multilingual for safety
            selected_model = 'xlm-roberta-base'
            model_type = 'multilingual'
            reason = 'Default multilingual selection'
        
        return {
            'selected_model': selected_model,
            'model_type': model_type,
            'reason': reason,
            'language_detected': primary_lang,
            'confidence': confidence,
            'is_mixed': is_mixed,
            'selection_timestamp': datetime.now().isoformat()
        }
    
    def process_dataset_languages(self, dataset_name: str) -> Dict[str, Any]:
        """
        Process entire dataset for language detection and transformer selection.
        
        Args:
            dataset_name: Name of the dataset to process
            
        Returns:
            Dictionary with language analysis results
        """
        self.logger.info(f"Processing language detection for dataset: {dataset_name}")
        
        try:
            # Load processed data
            processed_file = Path('datasets') / dataset_name / 'processed' / 'processed_data.csv'
            if not processed_file.exists():
                raise FileNotFoundError('Processed dataset not found')
            
            df = pd.read_csv(processed_file)
            
            # Find text column - check multiple possible column names
            text_column = None
            possible_text_columns = ['COMBINED_TEXT', 'TWEET_CONTENT', 'TWEET', 'TEXT', 'CONTENT', 'MESSAGE', 'DESCRIPTION']
            
            for col in possible_text_columns:
                if col in df.columns:
                    text_column = col
                    break
            
            if not text_column:
                # If no standard text column found, look for any column with substantial text content
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Check if column contains meaningful text (average length > 20 characters)
                        sample_values = df[col].dropna().head(100)
                        if len(sample_values) > 0:
                            avg_length = sample_values.astype(str).str.len().mean()
                            if avg_length > 20:
                                text_column = col
                                self.logger.info(f"Using column '{col}' as text column (avg length: {avg_length:.1f})")
                                break
                
                if not text_column:
                    raise ValueError('No suitable text column found in dataset')
            
            # Process each text sample
            results = []
            language_counts = {'en': 0, 'sw': 0, 'mixed': 0, 'unknown': 0}
            transformer_selections = {'monolingual': 0, 'multilingual': 0}
            
            # Limit processing for performance (can be removed for full processing)
            sample_size = min(len(df), 1000)
            df_sample = df.head(sample_size)
            
            for idx, row in df_sample.iterrows():
                text = str(row[text_column]) if pd.notna(row[text_column]) else ""
                
                if len(text.strip()) < 3:
                    continue
                
                # Detect language
                lang_result = self.detect_language(text)
                
                # Select transformer
                transformer_result = self.select_transformer(lang_result)
                
                # Combine results
                combined_result = {
                    'index': idx,
                    'text_preview': text[:100] + '...' if len(text) > 100 else text,
                    **lang_result,
                    **transformer_result
                }
                
                results.append(combined_result)
                
                # Update counts
                primary_lang = lang_result['primary_language']
                language_counts[primary_lang] = language_counts.get(primary_lang, 0) + 1
                
                model_type = transformer_result['model_type']
                transformer_selections[model_type] += 1
            
            # Calculate statistics
            total_processed = len(results)
            language_distribution = {
                lang: count / total_processed for lang, count in language_counts.items()
                if count > 0
            }
            
            # Determine dominant language and recommended transformer
            dominant_language = max(language_counts, key=language_counts.get)
            recommended_transformer = 'multilingual' if (
                language_counts['sw'] + language_counts['mixed'] > language_counts['en'] * 0.3
            ) else 'monolingual'
            
            # Create summary
            summary = {
                'dataset_name': dataset_name,
                'analysis_date': datetime.now().isoformat(),
                'total_samples_analyzed': total_processed,
                'total_samples_in_dataset': len(df),
                'dominant_language': dominant_language,
                'language_distribution': language_distribution,
                'language_counts': language_counts,
                'transformer_selections': transformer_selections,
                'recommended_transformer': recommended_transformer,
                'detailed_results': results[:50],  # Store first 50 for display
                'sample_texts': {
                    'english': [r['text_preview'] for r in results if r['primary_language'] == 'en'][:5],
                    'kiswahili': [r['text_preview'] for r in results if r['primary_language'] == 'sw'][:5],
                    'mixed': [r['text_preview'] for r in results if r['primary_language'] == 'mixed'][:5]
                }
            }
            
            # Save results
            self.file_manager.save_results(dataset_name, summary, 'language_detection')
            
            # Update dataset info
            self.file_manager.update_dataset_info(dataset_name, {
                'language_analysis_completed': True,
                'dominant_language': dominant_language,
                'recommended_transformer': recommended_transformer,
                'language_distribution': language_distribution
            })
            
            self.logger.info(f"Language analysis completed. Dominant language: {dominant_language}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in language detection: {e}")
            raise
    
    def get_transformer_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """
        Get transformer embeddings for texts using specified model.
        
        Args:
            texts: List of texts to embed
            model_name: Name of transformer model to use
            
        Returns:
            Numpy array of embeddings
        """
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available, returning dummy embeddings")
            return np.random.rand(len(texts), 768)  # Standard BERT embedding size
        
        try:
            # Load model if not cached
            if model_name not in self.loaded_models:
                self.logger.info(f"Loading transformer model: {model_name}")
                
                if 'sentence-transformers' in model_name or model_name == 'all-MiniLM-L6-v2':
                    model = self.model_manager.load_sentence_transformer(model_name)
                    self.loaded_models[model_name] = model
                else:
                    tokenizer, model = self.model_manager.load_transformers_model(model_name)
                    self.loaded_models[model_name] = model
                    self.tokenizers[model_name] = tokenizer
            
            model = self.loaded_models[model_name]
            
            # Generate embeddings
            if isinstance(model, SentenceTransformer):
                embeddings = model.encode(texts)
            else:
                tokenizer = self.tokenizers[model_name]
                embeddings = []
                
                for text in texts:
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                                     padding=True, max_length=512)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Use CLS token embedding
                        embedding = outputs.last_hidden_state[:, 0, :].numpy()
                        embeddings.append(embedding[0])
                
                embeddings = np.array(embeddings)
            
            self.logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            # Return dummy embeddings as fallback
            return np.random.rand(len(texts), 768)

def main():
    """Test language detection functionality."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test samples
    test_texts = [
        "This is fake news about the Kenyan election results.",
        "Hii ni habari za uwongo kuhusu matokeo ya uchaguzi wa Kenya.",
        "This fake news ina spread misinformation kuhusu siasa za Kenya.",
        "Breaking: President Ruto announces new policies for economic recovery.",
        "Rais Ruto ametangaza sera mpya za kuimarisha uchumi wa nchi."
    ]
    
    from src.utils.file_manager import FileManager
    file_manager = FileManager()
    detector = LanguageDetector(file_manager)
    
    print("🌍 LANGUAGE DETECTION TEST")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        result = detector.detect_language(text)
        transformer = detector.select_transformer(result)
        
        print(f"\n{i}. Text: {text[:50]}...")
        print(f"   Language: {result['primary_language']} (confidence: {result['confidence']:.2f})")
        print(f"   Mixed: {result['is_mixed']}")
        print(f"   Transformer: {transformer['selected_model']} ({transformer['model_type']})")
        print(f"   Reason: {transformer['reason']}")

if __name__ == "__main__":
    main()