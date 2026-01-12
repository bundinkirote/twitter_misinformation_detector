"""
Data Collector and Corpus Builder Module

This module provides comprehensive data collection and corpus building capabilities
for fact-checked data from various sources. It implements web scraping, data
aggregation, and corpus construction for building training datasets in misinformation
detection research with support for multiple fact-checking organizations.
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import urllib.parse
import re
from tqdm import tqdm

class DataCollector:
    """
    Data Collection and Corpus Building Class
    
    Implements comprehensive data collection from multiple fact-checking sources
    including PesaCheck, Africa Check, Wikipedia API, and local news sources.
    Provides systematic corpus building capabilities with data validation,
    deduplication, and quality assurance for misinformation detection research.
    """
    
    def __init__(self, corpus_dir: str = "corpus"):
        self.logger = logging.getLogger(__name__)
        self.corpus_dir = Path(corpus_dir)
        self.corpus_dir.mkdir(exist_ok=True)
        
        # Configure session with headers to avoid blocking
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Source configurations
        self.sources = {
            'pesacheck': {
                'base_url': 'https://pesacheck.org',
                'search_path': '/search',
                'fact_check_patterns': ['/fact-check/', '/verification/'],
                'enabled': True
            },
            'africacheck': {
                'base_url': 'https://africacheck.org',
                'search_path': '/reports',
                'fact_check_patterns': ['/reports/', '/fact-checks/'],
                'enabled': True
            },
            'wikipedia': {
                'base_url': 'https://en.wikipedia.org',
                'api_url': 'https://en.wikipedia.org/w/api.php',
                'enabled': True
            },
            'nation': {
                'base_url': 'https://nation.co.ke',
                'search_path': '/search',
                'enabled': False  # Disabled due to potential blocking
            }
        }
        
        # Rate limiting
        self.request_delay = 2  # seconds between requests
        
    def build_corpus(self, topics: List[str] = None, max_articles_per_source: int = 50) -> Dict[str, Any]:
        """
        Build comprehensive fact-check corpus.
        
        Args:
            topics: List of topics to search for
            max_articles_per_source: Maximum articles to collect per source
            
        Returns:
            Dictionary with corpus statistics and file paths
        """
        if topics is None:
            topics = [
                'Kenya politics', 'William Ruto', 'Raila Odinga', 'Finance Bill 2024',
                'Gen Z protests', 'Kenyan elections', 'corruption Kenya',
                'COVID-19 Kenya', 'economic policies Kenya', 'handshake Kenya'
            ]
        
        self.logger.info(f"Starting corpus building for {len(topics)} topics")
        
        corpus_data = {
            'metadata': {
                'build_date': datetime.now().isoformat(),
                'topics': topics,
                'sources_attempted': list(self.sources.keys()),
                'total_articles': 0
            },
            'articles': []
        }
        
        # Collect from each source
        for source_name, source_config in self.sources.items():
            if not source_config['enabled']:
                continue
                
            self.logger.info(f"Collecting from {source_name}...")
            
            try:
                if source_name == 'pesacheck':
                    articles = self._scrape_pesacheck(topics, max_articles_per_source)
                elif source_name == 'africacheck':
                    articles = self._scrape_africacheck(topics, max_articles_per_source)
                elif source_name == 'wikipedia':
                    articles = self._scrape_wikipedia(topics, max_articles_per_source)
                else:
                    articles = []
                
                corpus_data['articles'].extend(articles)
                self.logger.info(f"Collected {len(articles)} articles from {source_name}")
                
            except Exception as e:
                self.logger.error(f"Error collecting from {source_name}: {e}")
                continue
        
        corpus_data['metadata']['total_articles'] = len(corpus_data['articles'])
        
        # Save corpus
        corpus_file = self.corpus_dir / f"fact_check_corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, indent=2, ensure_ascii=False)
        
        # Create CSV for easier analysis
        if corpus_data['articles']:
            df = pd.DataFrame(corpus_data['articles'])
            csv_file = corpus_file.with_suffix('.csv')
            df.to_csv(csv_file, index=False)
            
        self.logger.info(f"Corpus built: {len(corpus_data['articles'])} articles saved to {corpus_file}")
        
        return {
            'total_articles': len(corpus_data['articles']),
            'corpus_file': str(corpus_file),
            'csv_file': str(csv_file) if corpus_data['articles'] else None,
            'sources_used': [name for name, config in self.sources.items() if config['enabled']]
        }
    
    def _scrape_pesacheck(self, topics: List[str], max_articles: int) -> List[Dict]:
        """Scrape PesaCheck for fact-checked articles."""
        articles = []
        
        try:
            # PesaCheck main page
            response = self.session.get(f"{self.sources['pesacheck']['base_url']}")
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find fact-check articles (adjust selectors based on actual site structure)
            fact_check_links = soup.find_all('a', href=True)
            fact_check_urls = []
            
            for link in fact_check_links:
                href = link.get('href', '')
                if any(pattern in href for pattern in self.sources['pesacheck']['fact_check_patterns']):
                    if href.startswith('/'):
                        href = self.sources['pesacheck']['base_url'] + href
                    fact_check_urls.append(href)
            
            # Limit articles
            fact_check_urls = fact_check_urls[:max_articles]
            
            for url in tqdm(fact_check_urls, desc="Scraping PesaCheck"):
                try:
                    article_data = self._extract_pesacheck_article(url)
                    if article_data:
                        articles.append(article_data)
                    
                    time.sleep(self.request_delay)  # Rate limiting
                    
                except Exception as e:
                    self.logger.warning(f"Error scraping PesaCheck article {url}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error accessing PesaCheck: {e}")
        
        return articles
    
    def _extract_pesacheck_article(self, url: str) -> Optional[Dict]:
        """Extract fact-check data from PesaCheck article."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article components (adjust selectors based on actual site)
            title = soup.find('h1')
            title_text = title.get_text().strip() if title else ""
            
            # Look for verdict/rating
            verdict_elem = soup.find(['div', 'span'], class_=re.compile(r'verdict|rating|conclusion', re.I))
            verdict = verdict_elem.get_text().strip() if verdict_elem else "UNKNOWN"
            
            # Extract claim being fact-checked
            claim_elem = soup.find(['div', 'p'], class_=re.compile(r'claim|statement', re.I))
            claim = claim_elem.get_text().strip() if claim_elem else title_text
            
            # Extract content
            content_divs = soup.find_all(['div', 'p'], class_=re.compile(r'content|article|body', re.I))
            content = ' '.join([div.get_text().strip() for div in content_divs[:3]])  # First 3 paragraphs
            
            # Extract date
            date_elem = soup.find(['time', 'span'], class_=re.compile(r'date|published', re.I))
            date_str = date_elem.get('datetime') or date_elem.get_text() if date_elem else ""
            
            return {
                'title': title_text,
                'claim': claim,
                'verdict': self._normalize_verdict(verdict),
                'content': content,
                'url': url,
                'source': 'pesacheck',
                'date_published': date_str,
                'scrape_date': datetime.now().isoformat(),
                'word_count': len(content.split()),
                'language': 'en'  # Assume English for PesaCheck
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting PesaCheck article {url}: {e}")
            return None
    
    def _scrape_africacheck(self, topics: List[str], max_articles: int) -> List[Dict]:
        """Scrape Africa Check for fact-checked articles."""
        articles = []
        
        try:
            # Access Africa Check reports section
            response = self.session.get(f"{self.sources['africacheck']['base_url']}/reports")
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            article_urls = []
            
            for link in article_links:
                href = link.get('href', '')
                if any(pattern in href for pattern in self.sources['africacheck']['fact_check_patterns']):
                    if href.startswith('/'):
                        href = self.sources['africacheck']['base_url'] + href
                    article_urls.append(href)
            
            article_urls = article_urls[:max_articles]
            
            for url in tqdm(article_urls, desc="Scraping Africa Check"):
                try:
                    article_data = self._extract_africacheck_article(url)
                    if article_data:
                        articles.append(article_data)
                    
                    time.sleep(self.request_delay)
                    
                except Exception as e:
                    self.logger.warning(f"Error scraping Africa Check article {url}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error accessing Africa Check: {e}")
        
        return articles
    
    def _extract_africacheck_article(self, url: str) -> Optional[Dict]:
        """Extract fact-check data from Africa Check article."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract components (similar to PesaCheck but adapted for Africa Check)
            title = soup.find(['h1', 'h2'])
            title_text = title.get_text().strip() if title else ""
            
            # Look for verdict
            verdict_elem = soup.find(['div', 'span'], text=re.compile(r'(true|false|misleading|unproven)', re.I))
            verdict = verdict_elem.get_text().strip() if verdict_elem else "UNKNOWN"
            
            # Extract main content
            content_divs = soup.find_all(['div', 'p'], class_=re.compile(r'content|entry', re.I))
            content = ' '.join([div.get_text().strip() for div in content_divs[:5]])
            
            return {
                'title': title_text,
                'claim': title_text,  # Title often contains the claim
                'verdict': self._normalize_verdict(verdict),
                'content': content,
                'url': url,
                'source': 'africacheck',
                'date_published': "",
                'scrape_date': datetime.now().isoformat(),
                'word_count': len(content.split()),
                'language': 'en'
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting Africa Check article {url}: {e}")
            return None
    
    def _scrape_wikipedia(self, topics: List[str], max_articles: int) -> List[Dict]:
        """Scrape Wikipedia for factual information using API."""
        articles = []
        
        try:
            for topic in topics[:max_articles//len(topics) + 1]:
                # Search Wikipedia API
                search_params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': topic,
                    'srlimit': 5
                }
                
                response = self.session.get(self.sources['wikipedia']['api_url'], params=search_params)
                response.raise_for_status()
                
                data = response.json()
                
                for page in data.get('query', {}).get('search', []):
                    try:
                        # Get page content
                        content_params = {
                            'action': 'query',
                            'format': 'json',
                            'prop': 'extracts',
                            'pageids': page['pageid'],
                            'exintro': True,
                            'explaintext': True,
                            'exsectionformat': 'plain'
                        }
                        
                        content_response = self.session.get(self.sources['wikipedia']['api_url'], params=content_params)
                        content_data = content_response.json()
                        
                        page_data = content_data.get('query', {}).get('pages', {}).get(str(page['pageid']), {})
                        extract = page_data.get('extract', '')
                        
                        if extract:
                            articles.append({
                                'title': page['title'],
                                'claim': page['title'],  # Use title as claim
                                'verdict': 'FACTUAL',  # Wikipedia is generally factual
                                'content': extract,
                                'url': f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page['title'])}",
                                'source': 'wikipedia',
                                'date_published': "",
                                'scrape_date': datetime.now().isoformat(),
                                'word_count': len(extract.split()),
                                'language': 'en'
                            })
                        
                        time.sleep(self.request_delay)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing Wikipedia page {page.get('title', '')}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error accessing Wikipedia: {e}")
        
        return articles
    
    def _normalize_verdict(self, verdict: str) -> str:
        """Normalize verdict to standard categories."""
        verdict = verdict.lower().strip()
        
        if any(word in verdict for word in ['true', 'correct', 'factual', 'accurate']):
            return 'TRUE'
        elif any(word in verdict for word in ['false', 'incorrect', 'wrong', 'fake']):
            return 'FALSE'
        elif any(word in verdict for word in ['misleading', 'partial', 'mixed', 'disputed']):
            return 'MISLEADING'
        elif any(word in verdict for word in ['unproven', 'unclear', 'insufficient']):
            return 'UNPROVEN'
        else:
            return 'UNKNOWN'
    
    def load_existing_corpus(self, corpus_file: str) -> pd.DataFrame:
        """Load existing corpus file."""
        corpus_path = Path(corpus_file)
        
        if corpus_path.suffix == '.json':
            with open(corpus_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data.get('articles', []))
        elif corpus_path.suffix == '.csv':
            return pd.read_csv(corpus_path)
        else:
            raise ValueError(f"Unsupported file format: {corpus_path.suffix}")
    
    def update_corpus(self, existing_corpus_file: str, additional_topics: List[str] = None) -> Dict[str, Any]:
        """Update existing corpus with new data."""
        # Load existing corpus
        existing_df = self.load_existing_corpus(existing_corpus_file)
        existing_urls = set(existing_df['url'].tolist()) if 'url' in existing_df.columns else set()
        
        # Build new corpus
        new_corpus_data = self.build_corpus(additional_topics)
        
        # Load new articles and filter out duplicates
        new_df = pd.read_csv(new_corpus_data['csv_file']) if new_corpus_data['csv_file'] else pd.DataFrame()
        
        if not new_df.empty:
            # Filter out existing URLs
            new_df = new_df[~new_df['url'].isin(existing_urls)]
            
            # Combine with existing
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Save updated corpus
            updated_file = self.corpus_dir / f"updated_corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            combined_df.to_csv(updated_file, index=False)
            
            return {
                'total_articles': len(combined_df),
                'new_articles': len(new_df),
                'existing_articles': len(existing_df),
                'updated_corpus_file': str(updated_file)
            }
        
        return {
            'total_articles': len(existing_df),
            'new_articles': 0,
            'existing_articles': len(existing_df),
            'updated_corpus_file': existing_corpus_file
        }
    
    def _scrape_full_article(self, url: str, headers: Dict) -> str:
        """Scrape full content from article URL."""
        try:
            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            # Try to find main content
            content_selectors = [
                'article', '.content', '.post-content', '.entry-content', 
                '.article-content', 'main', '.main-content'
            ]
            
            content = ''
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(strip=True, separator=' ')
                    break
            
            if not content:
                # Fallback to body content
                body = soup.find('body')
                if body:
                    content = body.get_text(strip=True, separator=' ')
            
            return content[:5000]  # Limit content length
            
        except Exception as e:
            self.logger.warning(f"Error scraping full article {url}: {e}")
            return ""
    
    def _extract_claim_from_content(self, content: str) -> str:
        """Extract the main claim from article content."""
        if not content:
            return "Unknown claim"
        
        sentences = content.split('.')[:3]  # Take first 3 sentences
        claim = '. '.join(sentences).strip()
        
        return claim[:500] if claim else "Unknown claim"
    
    def auto_trigger_scraping(self, text: str, keywords: List[str] = None) -> Dict[str, Any]:
        """Automatically trigger scraping based on text content/tweet information."""
        try:
            # Default keywords for Kenyan political content
            if not keywords:
                keywords = [
                    'ruto', 'raila', 'uhuru', 'kenya', 'nairobi', 'parliament',
                    'election', 'politics', 'government', 'president', 'deputy president',
                    'finance bill', 'tax', 'protest', 'gen z', 'corruption'
                ]
            
            # Check if text contains relevant keywords
            text_lower = text.lower()
            relevant_topics = []
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    relevant_topics.append(keyword)
            
            if not relevant_topics:
                return {
                    'scraping_triggered': False,
                    'reason': 'No relevant keywords found',
                    'topics': []
                }
            
            # Extract entities and topics for scraping
            topics_to_scrape = list(set(relevant_topics))[:5]  # Limit to 5 topics
            
            self.logger.info(f"Auto-triggering scraping for topics: {topics_to_scrape}")
            
            # Build targeted corpus
            corpus_result = self.build_corpus(topics_to_scrape, max_articles_per_source=10)
            
            return {
                'scraping_triggered': True,
                'topics': topics_to_scrape,
                'corpus_data': corpus_result,
                'reason': f'Found {len(topics_to_scrape)} relevant topics'
            }
            
        except Exception as e:
            self.logger.error(f"Error in auto-trigger scraping: {e}")
            return {
                'scraping_triggered': False,
                'reason': f'Error: {str(e)}',
                'topics': []
            }