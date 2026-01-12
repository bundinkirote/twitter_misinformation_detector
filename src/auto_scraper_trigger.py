"""
Auto-Scraper Trigger System Module

This module provides automated scraping trigger capabilities based on content analysis
and context evaluation. It implements intelligent content monitoring, keyword detection,
and automated data collection triggers for dynamic fact-checking corpus expansion
in misinformation detection systems.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from src.data_collector import DataCollector
from src.fact_check_validator import FactCheckValidator
from src.utils.file_manager import FileManager


class AutoScraperTrigger:
    """
    Automated Scraping Trigger Class
    
    Implements intelligent content analysis and automated scraping triggers based on
    tweet content, context evaluation, and keyword detection. Provides dynamic
    fact-checking corpus expansion capabilities for enhanced misinformation detection.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_collector = DataCollector()
        self.file_manager = FileManager()
        self.fact_checker = FactCheckValidator(self.file_manager)
        
        # Keywords that trigger scraping
        self.trigger_keywords = {
            'politics': ['ruto', 'raila', 'uhuru', 'parliament', 'government', 'election', 'president'],
            'economy': ['finance bill', 'tax', 'economy', 'budget', 'shilling', 'inflation'],
            'social': ['protest', 'gen z', 'demonstration', 'strike', 'youth'],
            'corruption': ['corruption', 'scandal', 'bribery', 'fraud', 'embezzlement'],
            'health': ['covid', 'vaccine', 'health', 'hospital', 'disease'],
            'education': ['school', 'university', 'education', 'student', 'teacher']
        }
        
        # Source reliability scores
        self.source_reliability = {
            'verified_account': 0.8,
            'news_media': 0.9,
            'government_official': 0.7,
            'unknown': 0.3
        }
    
    def analyze_tweet_content(self, tweet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tweet content and determine if scraping should be triggered."""
        try:
            text = tweet_data.get('text', '')
            author = tweet_data.get('author', {})
            engagement = tweet_data.get('engagement', {})
            
            # Content analysis
            content_analysis = self._analyze_content_relevance(text)
            
            # Author credibility analysis
            author_analysis = self._analyze_author_credibility(author)
            
            # Engagement analysis
            engagement_analysis = self._analyze_engagement_patterns(engagement)
            
            # Calculate overall trigger score
            trigger_score = self._calculate_trigger_score(
                content_analysis, author_analysis, engagement_analysis
            )
            
            # Determine if scraping should be triggered
            should_trigger = trigger_score > 0.6  # Threshold
            
            result = {
                'should_trigger_scraping': should_trigger,
                'trigger_score': trigger_score,
                'content_analysis': content_analysis,
                'author_analysis': author_analysis,
                'engagement_analysis': engagement_analysis,
                'recommended_sources': self._recommend_scraping_sources(content_analysis),
                'urgency_level': self._determine_urgency(trigger_score, engagement_analysis)
            }
            
            if should_trigger:
                self.logger.info(f"Scraping triggered for tweet with score: {trigger_score:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing tweet content: {e}")
            return {'should_trigger_scraping': False, 'error': str(e)}
    
    def _analyze_content_relevance(self, text: str) -> Dict[str, Any]:
        """Analyze content for relevance and potential misinformation indicators."""
        text_lower = text.lower()
        
        # Check for trigger keywords
        triggered_categories = []
        keyword_matches = []
        
        for category, keywords in self.trigger_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                triggered_categories.append(category)
                keyword_matches.extend(matches)
        
        # Check for misinformation indicators
        misinfo_indicators = [
            'breaking', 'urgent', 'exclusive', 'leaked', 'hidden truth',
            'they don\'t want you to know', 'mainstream media', 'cover up'
        ]
        
        misinfo_score = sum(1 for indicator in misinfo_indicators if indicator in text_lower)
        
        # Check for factual claims
        claim_patterns = [
            r'\d+%', r'\d+\s*(million|billion|trillion)', r'according to',
            r'study shows', r'research reveals', r'data indicates'
        ]
        
        factual_claims = sum(1 for pattern in claim_patterns if re.search(pattern, text_lower))
        
        relevance_score = (
            len(triggered_categories) * 0.3 +
            len(keyword_matches) * 0.1 +
            min(misinfo_score * 0.2, 1.0) +
            min(factual_claims * 0.15, 0.6)
        )
        
        return {
            'relevance_score': min(relevance_score, 1.0),
            'triggered_categories': triggered_categories,
            'keyword_matches': keyword_matches,
            'misinformation_indicators': misinfo_score,
            'factual_claims': factual_claims,
            'topics_for_scraping': list(set(keyword_matches))[:5]  # Top 5
        }
    
    def _analyze_author_credibility(self, author: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze author credibility and reliability."""
        verification_status = author.get('verified', False)
        follower_count = author.get('followers_count', 0)
        account_age = author.get('account_age_days', 0)
        bio = author.get('bio', '').lower()
        
        credibility_score = 0.0
        
        # Verification bonus
        if verification_status:
            credibility_score += 0.3
        
        # Follower count (logarithmic scale)
        if follower_count > 100000:
            credibility_score += 0.2
        elif follower_count > 10000:
            credibility_score += 0.15
        elif follower_count > 1000:
            credibility_score += 0.1
        
        # Account age
        if account_age > 365:
            credibility_score += 0.2
        elif account_age > 180:
            credibility_score += 0.1
        
        # Bio indicators
        news_indicators = ['journalist', 'news', 'reporter', 'media', 'editor']
        if any(indicator in bio for indicator in news_indicators):
            credibility_score += 0.2
        
        # Government/official indicators
        official_indicators = ['official', 'government', 'ministry', 'office of']
        if any(indicator in bio for indicator in official_indicators):
            credibility_score += 0.15
        
        return {
            'credibility_score': min(credibility_score, 1.0),
            'verified': verification_status,
            'follower_count': follower_count,
            'account_age': account_age,
            'author_type': self._classify_author_type(author)
        }
    
    def _analyze_engagement_patterns(self, engagement: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze engagement patterns for viral/trending content."""
        retweets = engagement.get('retweets', 0)
        likes = engagement.get('likes', 0)
        replies = engagement.get('replies', 0)
        shares = engagement.get('shares', 0)
        
        total_engagement = retweets + likes + replies + shares
        
        # Calculate engagement velocity (if timestamp available)
        engagement_score = 0.0
        
        if total_engagement > 10000:
            engagement_score += 0.4
        elif total_engagement > 1000:
            engagement_score += 0.3
        elif total_engagement > 100:
            engagement_score += 0.2
        elif total_engagement > 10:
            engagement_score += 0.1
        
        # Reply-to-like ratio (high replies might indicate controversy)
        if likes > 0:
            controversy_ratio = replies / likes
            if controversy_ratio > 0.5:  # High controversy
                engagement_score += 0.2
        
        return {
            'engagement_score': min(engagement_score, 1.0),
            'total_engagement': total_engagement,
            'retweets': retweets,
            'likes': likes,
            'replies': replies,
            'is_trending': total_engagement > 1000,
            'controversy_level': 'high' if replies > likes else 'low'
        }
    
    def _calculate_trigger_score(self, content: Dict, author: Dict, engagement: Dict) -> float:
        """Calculate overall trigger score for scraping."""
        # Weighted combination
        score = (
            content['relevance_score'] * 0.5 +  # Content is most important
            author['credibility_score'] * 0.2 +  # Author credibility
            engagement['engagement_score'] * 0.3  # Engagement level
        )
        
        return min(score, 1.0)
    
    def _recommend_scraping_sources(self, content_analysis: Dict) -> List[str]:
        """Recommend which sources to scrape based on content."""
        sources = ['wikipedia']  # Always include Wikipedia
        
        categories = content_analysis.get('triggered_categories', [])
        
        # Add specialized sources based on content
        if 'politics' in categories or 'corruption' in categories:
            sources.extend(['pesacheck', 'africacheck'])
        
        if 'economy' in categories:
            sources.extend(['pesacheck'])  # PesaCheck covers economic claims
        
        return list(set(sources))
    
    def _determine_urgency(self, trigger_score: float, engagement: Dict) -> str:
        """Determine urgency level for scraping."""
        if trigger_score > 0.8 and engagement.get('is_trending', False):
            return 'urgent'
        elif trigger_score > 0.7:
            return 'high'
        elif trigger_score > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _classify_author_type(self, author: Dict) -> str:
        """Classify author type for credibility assessment."""
        bio = author.get('bio', '').lower()
        verified = author.get('verified', False)
        
        if verified:
            if any(word in bio for word in ['news', 'journalist', 'reporter', 'media']):
                return 'verified_news_media'
            elif any(word in bio for word in ['official', 'government', 'minister']):
                return 'verified_official'
            else:
                return 'verified_other'
        else:
            return 'unverified'
    
    async def trigger_scraping_pipeline(self, tweet_data: Dict) -> Dict[str, Any]:
        """Execute the full scraping pipeline when triggered."""
        try:
            # Analyze tweet
            analysis = self.analyze_tweet_content(tweet_data)
            
            if not analysis.get('should_trigger_scraping', False):
                return {
                    'scraping_executed': False,
                    'reason': 'Trigger threshold not met',
                    'analysis': analysis
                }
            
            # Extract topics for scraping
            topics = analysis['content_analysis'].get('topics_for_scraping', [])
            sources = analysis.get('recommended_sources', ['wikipedia'])
            
            # Execute scraping
            scraping_results = []
            
            # Build targeted corpus
            corpus_result = self.data_collector.build_corpus(
                topics=topics,
                max_articles_per_source=15
            )
            
            # Perform fact-checking on the original tweet
            fact_check_result = self.fact_checker.validate_with_external_sources(tweet_data.get('text', ''))
            
            return {
                'scraping_executed': True,
                'analysis': analysis,
                'corpus_result': corpus_result,
                'fact_check_result': fact_check_result,
                'topics_scraped': topics,
                'sources_used': sources,
                'urgency_level': analysis.get('urgency_level', 'medium'),
                'execution_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in scraping pipeline: {e}")
            return {
                'scraping_executed': False,
                'error': str(e)
            }


def process_tweet_for_scraping(tweet_text: str, author_info: Dict = None, engagement_info: Dict = None) -> Dict:
    """Convenience function to process a tweet and trigger scraping if needed."""
    trigger_system = AutoScraperTrigger()
    
    tweet_data = {
        'text': tweet_text,
        'author': author_info or {},
        'engagement': engagement_info or {}
    }
    
    # Run synchronously for simplicity
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(trigger_system.trigger_scraping_pipeline(tweet_data))
        return result
    finally:
        loop.close()