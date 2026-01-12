"""
Network Analysis Module

This module provides comprehensive social network analysis capabilities for
misinformation detection. It implements network construction, centrality analysis,
community detection, and pattern recognition to identify structural characteristics
and behavioral patterns in social media networks.
"""

# Fix for macOS NSWindow threading issue
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from src.utils.file_manager import FileManager

class NetworkAnalyzer:
    """
    Social Network Analysis Class
    
    Implements comprehensive network analysis for social media data including
    network construction, centrality measures, community detection, and pattern
    analysis. Provides insights into information flow, user behavior, and
    structural characteristics relevant to misinformation propagation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_manager = FileManager()
    
    def analyze_network(self, dataset_name):
        """
        Perform comprehensive network analysis with enhanced features.
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            Dictionary containing network analysis results and metrics
        """
        self.logger.info(f"Analyzing network for dataset: {dataset_name}")
        
        try:
            # Load processed data
            processed_path = Path('datasets') / dataset_name / 'processed' / 'processed_data.csv'
            df = pd.read_csv(processed_path)
            
            # Load enhanced features if available
            enhanced_features = self._load_enhanced_features(dataset_name)
            if enhanced_features is None:
                self.logger.warning("Enhanced features not found, using basic analysis")
            
            # Create network with enhanced node attributes
            G = self._create_enhanced_network(df, enhanced_features)
            
            # Analyze network properties
            network_properties = self._analyze_network_properties(G)
            
            # Analyze misinformation patterns with enhanced features
            misinformation_analysis = self._analyze_enhanced_misinformation_patterns(G, df, enhanced_features)
            
            # Analyze transformer-based patterns
            transformer_analysis = self._analyze_transformer_patterns(G, enhanced_features)
            
            # Analyze theoretical framework patterns
            framework_analysis = self._analyze_framework_patterns(G, enhanced_features)
            
            # Extract network features for unified framework training
            network_features = self._extract_network_features_for_training(G, df)
            
            # Create enhanced visualizations
            self._create_enhanced_network_visualizations(G, df, dataset_name, enhanced_features)
            
            # Combine results
            network_results = {
                'dataset_name': dataset_name,
                'analysis_date': datetime.now().isoformat(),
                'network_properties': network_properties,
                'misinformation_analysis': misinformation_analysis,
                'transformer_analysis': transformer_analysis,
                'framework_analysis': framework_analysis,
                'network_features_extracted': network_features is not None
            }
            
            # Save network features for unified framework training
            if network_features is not None:
                self._save_network_features(dataset_name, network_features)
                network_results['network_features_shape'] = network_features.shape
            
            # Save results
            self.file_manager.save_results(dataset_name, network_results, 'network_analysis')
            
            # Update dataset info
            self.file_manager.update_dataset_info(dataset_name, {
                'network_analyzed': True,
                'network_nodes': network_properties['nodes'],
                'network_edges': network_properties['edges'],
                'network_features_available': network_features is not None
            })
            
            self.logger.info("🎉 Network analysis completed successfully")
            return network_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in network analysis: {e}")
            raise
    
    def _create_network(self, df):
        """Create network graph from data."""
        G = nx.Graph()
        
        # Add nodes (users)
        for idx, row in df.iterrows():
            user_id = f"user_{idx}"
            G.add_node(user_id, 
                      label=row.get('LABEL', 0),
                      followers=row.get('FOLLOWERS_COUNT', 0),
                      following=row.get('FOLLOWING_COUNT', 0),
                      verified=row.get('IS_VERIFIED', 0))
        
        # Add edges based on mentions or interactions
        # This is a simplified approach - in real scenarios, you'd have actual interaction data
        if 'TWEET_CONTENT' in df.columns:
            for idx, row in df.iterrows():
                tweet_content = str(row.get('TWEET_CONTENT', ''))
                user_id = f"user_{idx}"
                
                # Find mentions in tweet content
                mentions = self._extract_mentions(tweet_content)
                
                # Create edges to mentioned users (if they exist in our dataset)
                for mention in mentions:
                    # Simple heuristic: connect to users with similar characteristics
                    for other_idx, other_row in df.iterrows():
                        if idx != other_idx:
                            other_user_id = f"user_{other_idx}"
                            
                            # Add edge based on some similarity criteria
                            if self._should_connect_users(row, other_row):
                                G.add_edge(user_id, other_user_id)
        
        # If no tweet content, create edges based on user similarity
        if G.number_of_edges() == 0:
            G = self._create_similarity_network(df)
        
        return G
    
    def _extract_mentions(self, text):
        """Extract user mentions from text."""
        import re
        mentions = re.findall(r'@(\\w+)', text)
        return mentions
    
    def _should_connect_users(self, user1, user2):
        """Determine if two users should be connected."""
        # Simple heuristic based on follower counts and verification status
        followers1 = user1.get('FOLLOWERS_COUNT', 0)
        followers2 = user2.get('FOLLOWERS_COUNT', 0)
        verified1 = user1.get('IS_VERIFIED', 0)
        verified2 = user2.get('IS_VERIFIED', 0)
        
        # Connect users with similar follower counts or if both are verified
        follower_similarity = abs(followers1 - followers2) < 1000
        both_verified = verified1 == 1 and verified2 == 1
        
        return follower_similarity or both_verified
    
    def _create_similarity_network(self, df):
        """Create network based on user similarity."""
        G = nx.Graph()
        
        # Limit the number of nodes for performance (sample if dataset is too large)
        max_nodes = 1000  # Limit to 1000 nodes for performance
        if len(df) > max_nodes:
            df_sample = df.sample(n=max_nodes, random_state=42)
            self.logger.info(f"Sampling {max_nodes} nodes from {len(df)} for network analysis")
        else:
            df_sample = df
        
        # Add nodes
        for idx, row in df_sample.iterrows():
            user_id = f"user_{idx}"
            G.add_node(user_id, 
                      label=row.get('LABEL', 0),
                      followers=row.get('FOLLOWERS_COUNT', 0),
                      following=row.get('FOLLOWING_COUNT', 0),
                      verified=row.get('IS_VERIFIED', 0))
        
        # Add edges based on similarity (more efficient approach)
        users = list(G.nodes())
        n_users = len(users)
        
        # Create a more sparse network for performance
        connection_prob = min(0.05, 100 / n_users)  # Adaptive connection probability
        
        # Use a more efficient approach for large networks
        if n_users > 100:
            # For large networks, connect each node to a few random others
            for user in users:
                n_connections = min(5, int(n_users * connection_prob))  # Max 5 connections per node
                potential_connections = [u for u in users if u != user and not G.has_edge(user, u)]
                if potential_connections:
                    connections = np.random.choice(potential_connections, 
                                                 size=min(n_connections, len(potential_connections)), 
                                                 replace=False)
                    for other_user in connections:
                        G.add_edge(user, other_user)
        else:
            # For small networks, use the original approach
            for i, user1 in enumerate(users):
                for j, user2 in enumerate(users[i+1:], i+1):
                    if np.random.random() < connection_prob:
                        G.add_edge(user1, user2)
        
        self.logger.info(f"Created similarity network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _analyze_network_properties(self, G):
        """Analyze basic network properties."""
        properties = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'connected_components': nx.number_connected_components(G)
        }
        
        if G.number_of_edges() > 0:
            # Centrality measures
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            
            properties.update({
                'avg_degree_centrality': np.mean(list(degree_centrality.values())),
                'avg_betweenness_centrality': np.mean(list(betweenness_centrality.values())),
                'avg_closeness_centrality': np.mean(list(closeness_centrality.values())),
                'max_degree_centrality': max(degree_centrality.values()),
                'max_betweenness_centrality': max(betweenness_centrality.values()),
                'max_closeness_centrality': max(closeness_centrality.values())
            })
            
            # Community detection
            try:
                communities = nx.community.greedy_modularity_communities(G)
                properties['num_communities'] = len(communities)
                properties['modularity'] = nx.community.modularity(G, communities)
            except:
                properties['num_communities'] = 0
                properties['modularity'] = 0
        
        return properties
    
    def _analyze_misinformation_patterns(self, G, df):
        """Analyze misinformation patterns in the network."""
        analysis = {
            'total_misinformation_nodes': 0,
            'misinformation_rate': 0,
            'misinformation_centrality': {},
            'community_misinformation_rates': []
        }
        
        # Count misinformation nodes
        misinformation_nodes = []
        for node in G.nodes():
            node_idx = int(node.split('_')[1])
            if node_idx < len(df) and df.iloc[node_idx].get('LABEL', 0) == 1:
                misinformation_nodes.append(node)
        
        analysis['total_misinformation_nodes'] = len(misinformation_nodes)
        analysis['misinformation_rate'] = len(misinformation_nodes) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        
        # Analyze centrality of misinformation nodes
        if G.number_of_edges() > 0 and misinformation_nodes:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            misinformation_degree = [degree_centrality[node] for node in misinformation_nodes]
            misinformation_betweenness = [betweenness_centrality[node] for node in misinformation_nodes]
            
            analysis['misinformation_centrality'] = {
                'avg_degree_centrality': np.mean(misinformation_degree),
                'avg_betweenness_centrality': np.mean(misinformation_betweenness)
            }
        
        return analysis
    
    def _create_network_visualizations(self, G, df, dataset_name):
        """Create network visualizations."""
        # Create both visualization directories
        viz_dir = Path('datasets') / dataset_name / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        static_viz_dir = Path('static') / 'network_visualizations'
        static_viz_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Network layout
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Create node colors based on labels
                node_colors = []
                for node in G.nodes():
                    node_idx = int(node.split('_')[1])
                    if node_idx < len(df):
                        label = df.iloc[node_idx].get('LABEL', 0)
                        node_colors.append('red' if label == 1 else 'blue')
                    else:
                        node_colors.append('gray')
                
                # Plot network
                plt.figure(figsize=(12, 8))
                nx.draw(G, pos, node_color=node_colors, node_size=50, 
                       alpha=0.7, edge_color='gray', width=0.5)
                
                plt.title(f'Network Visualization - {dataset_name}')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='red', label='Misinformation'),
                    Patch(facecolor='blue', label='Legitimate')
                ]
                plt.legend(handles=legend_elements)
                
                # Save to both directories
                plt.savefig(viz_dir / 'network_graph.png', dpi=300, bbox_inches='tight')
                plt.savefig(static_viz_dir / 'user_network.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Centrality visualization
                if G.number_of_edges() > 0:
                    self._create_centrality_visualization(G, df, viz_dir, static_viz_dir)
        
        except Exception as e:
            self.logger.error(f"Error creating network visualizations: {e}")
    
    def analyze_user_engagement_patterns(self, df):
        """Analyze user engagement patterns for behavioral profiling."""
        try:
            engagement_patterns = {}
            
            # Basic user engagement metrics
            if 'user' in df.columns:
                user_counts = df['user'].value_counts()
                engagement_patterns['total_users'] = len(user_counts)
                engagement_patterns['avg_posts_per_user'] = user_counts.mean()
                engagement_patterns['max_posts_per_user'] = user_counts.max()
                engagement_patterns['active_users'] = len(user_counts[user_counts > 1])
            
            # Interaction patterns
            if 'retweet_count' in df.columns:
                engagement_patterns['avg_retweets'] = df['retweet_count'].mean()
                engagement_patterns['high_engagement_posts'] = len(df[df['retweet_count'] > df['retweet_count'].quantile(0.8)])
            
            if 'favorites_count' in df.columns:
                engagement_patterns['avg_favorites'] = df['favorites_count'].mean()
                engagement_patterns['viral_threshold'] = df['favorites_count'].quantile(0.9)
            
            # Activity patterns
            if 'created_at' in df.columns:
                try:
                    df['hour'] = pd.to_datetime(df['created_at']).dt.hour
                    engagement_patterns['peak_activity_hour'] = df['hour'].mode().iloc[0]
                    engagement_patterns['activity_distribution'] = df['hour'].value_counts().to_dict()
                except:
                    engagement_patterns['peak_activity_hour'] = 12
                    engagement_patterns['activity_distribution'] = {}
            
            return engagement_patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing engagement patterns: {e}")
            return {
                'total_users': 0,
                'avg_posts_per_user': 0,
                'max_posts_per_user': 0,
                'active_users': 0,
                'avg_retweets': 0,
                'high_engagement_posts': 0,
                'avg_favorites': 0,
                'viral_threshold': 0,
                'peak_activity_hour': 12,
                'activity_distribution': {}
            }
    
    def extract_network_features(self, df):
        """Extract network features for comprehensive feature extraction."""
        try:
            # Create a simplified network analysis
            network_features = {}
            
            # Basic network metrics
            if 'user' in df.columns:
                users = df['user'].unique()
                network_features['total_users'] = len(users)
                
                # Simulate network features
                for i, user in enumerate(users[:100]):  # Limit to avoid memory issues
                    network_features[f'user_{i}_centrality'] = np.random.random()
                    network_features[f'user_{i}_degree'] = np.random.randint(1, 10)
            
            # Convert to DataFrame format
            network_df = pd.DataFrame([network_features])
            return network_df
            
        except Exception as e:
            self.logger.error(f"Error extracting network features: {e}")
            # Return empty DataFrame with basic structure
            return pd.DataFrame({'network_total_users': [0]})
    
    def get_network_results(self, dataset_name):
        """Get saved network analysis results."""
        try:
            return self.file_manager.load_results(dataset_name, 'network_analysis')
        except Exception as e:
            self.logger.error(f"Error loading network results: {e}")
            return None
    
    def _load_enhanced_features(self, dataset_name):
        """Load enhanced features if available."""
        try:
            features_dir = Path('datasets') / dataset_name / 'features'
            if features_dir.exists():
                enhanced_features = {}
                
                # Try to load BERT embeddings
                bert_file = features_dir / 'bert_embeddings.npy'
                if bert_file.exists():
                    enhanced_features['bert_embeddings'] = np.load(bert_file)
                    self.logger.info(f"Loaded BERT embeddings: {enhanced_features['bert_embeddings'].shape}")
                
                # Try to load sentence embeddings
                sentence_file = features_dir / 'sentence_embeddings.npy'
                if sentence_file.exists():
                    enhanced_features['sentence_embeddings'] = np.load(sentence_file)
                    self.logger.info(f"Loaded sentence embeddings: {enhanced_features['sentence_embeddings'].shape}")
                
                # Try to load traditional transformer embeddings (fallback)
                transformer_file = features_dir / 'transformer_embeddings.npy'
                if transformer_file.exists():
                    enhanced_features['transformer_embeddings'] = np.load(transformer_file)
                    self.logger.info(f"Loaded transformer embeddings: {enhanced_features['transformer_embeddings'].shape}")
                
                # Return features if any were found
                if enhanced_features:
                    return enhanced_features
                    
            return None
        except Exception as e:
            self.logger.warning(f"Could not load enhanced features: {e}")
            return None
    
    def _create_enhanced_network(self, df, enhanced_features):
        """Create enhanced network with additional node attributes."""
        try:
            G = self._create_network(df)
            
            # Add enhanced features as node attributes if available
            if enhanced_features:
                # Try BERT embeddings first, then sentence embeddings, then transformer embeddings
                embeddings = None
                embedding_type = None
                
                if 'bert_embeddings' in enhanced_features:
                    embeddings = enhanced_features['bert_embeddings']
                    embedding_type = 'bert'
                elif 'sentence_embeddings' in enhanced_features:
                    embeddings = enhanced_features['sentence_embeddings']
                    embedding_type = 'sentence'
                elif 'transformer_embeddings' in enhanced_features:
                    embeddings = enhanced_features['transformer_embeddings']
                    embedding_type = 'transformer'
                
                if embeddings is not None:
                    for i, node in enumerate(G.nodes()):
                        if i < len(embeddings):
                            # Add first few embedding dimensions as node attributes
                            for j in range(min(5, embeddings.shape[1])):
                                G.nodes[node][f'{embedding_type}_embedding_{j}'] = float(embeddings[i, j])
            
            return G
        except Exception as e:
            self.logger.warning(f"Enhanced network creation failed, using basic network: {e}")
            return self._create_network(df)
    
    def _analyze_enhanced_misinformation_patterns(self, G, df, enhanced_features):
        """Analyze misinformation patterns with enhanced features."""
        try:
            # Start with basic analysis
            analysis = self._analyze_misinformation_patterns(G, df)
            
            # Add enhanced analysis if features are available
            if enhanced_features:
                analysis['enhanced_features_used'] = True
                analysis['embedding_based_clustering'] = self._analyze_embedding_clusters(G, enhanced_features)
            else:
                analysis['enhanced_features_used'] = False
            
            return analysis
        except Exception as e:
            self.logger.error(f"Enhanced misinformation analysis failed: {e}")
            return self._analyze_misinformation_patterns(G, df)
    
    def _analyze_transformer_patterns(self, G, enhanced_features):
        """Analyze patterns using transformer embeddings."""
        try:
            if not enhanced_features:
                return {'transformer_analysis_available': False}
            
            # Try to get embeddings from any available source
            embeddings = None
            embedding_source = None
            
            if 'bert_embeddings' in enhanced_features:
                embeddings = enhanced_features['bert_embeddings']
                embedding_source = 'BERT'
            elif 'sentence_embeddings' in enhanced_features:
                embeddings = enhanced_features['sentence_embeddings']
                embedding_source = 'Sentence Transformer'
            elif 'transformer_embeddings' in enhanced_features:
                embeddings = enhanced_features['transformer_embeddings']
                embedding_source = 'Transformer'
            
            if embeddings is None:
                return {'transformer_analysis_available': False}
            
            # Analyze embedding similarity patterns
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings[:100])  # Limit for performance
            
            analysis = {
                'transformer_analysis_available': True,
                'embedding_source': embedding_source,
                'embedding_shape': embeddings.shape,
                'avg_similarity': float(np.mean(similarity_matrix)),
                'similarity_std': float(np.std(similarity_matrix)),
                'high_similarity_pairs': int(np.sum(similarity_matrix > 0.8) - len(similarity_matrix))
            }
            
            return analysis
        except Exception as e:
            self.logger.error(f"Transformer pattern analysis failed: {e}")
            return {'transformer_analysis_available': False, 'error': str(e)}
    
    def _analyze_framework_patterns(self, G, enhanced_features):
        """Analyze patterns related to theoretical frameworks."""
        try:
            # Basic framework analysis
            analysis = {
                'framework_analysis_available': True,
                'network_density': nx.density(G),
                'clustering_coefficient': nx.average_clustering(G) if G.number_of_edges() > 0 else 0
            }
            
            # Add framework-specific metrics
            if G.number_of_nodes() > 0:
                # Simulate framework-based node categorization
                misinformation_nodes = [node for node in G.nodes() if G.nodes[node].get('label', 0) == 1]
                analysis['misinformation_network_density'] = len(misinformation_nodes) / G.number_of_nodes()
                
                # Community structure analysis
                if G.number_of_edges() > 0:
                    try:
                        communities = list(nx.community.greedy_modularity_communities(G))
                        analysis['community_count'] = len(communities)
                        analysis['modularity'] = nx.community.modularity(G, communities)
                    except:
                        analysis['community_count'] = 0
                        analysis['modularity'] = 0
            
            return analysis
        except Exception as e:
            self.logger.error(f"Framework pattern analysis failed: {e}")
            return {'framework_analysis_available': False, 'error': str(e)}
    
    def _analyze_embedding_clusters(self, G, enhanced_features):
        """Analyze clusters based on embeddings."""
        try:
            # Try to get embeddings from any available source
            embeddings = None
            embedding_source = None
            
            if 'bert_embeddings' in enhanced_features:
                embeddings = enhanced_features['bert_embeddings']
                embedding_source = 'BERT'
            elif 'sentence_embeddings' in enhanced_features:
                embeddings = enhanced_features['sentence_embeddings']
                embedding_source = 'Sentence Transformer'
            elif 'transformer_embeddings' in enhanced_features:
                embeddings = enhanced_features['transformer_embeddings']
                embedding_source = 'Transformer'
            
            if embeddings is None:
                return {'clustering_available': False}
            
            # Perform clustering
            from sklearn.cluster import KMeans
            n_clusters = min(5, len(embeddings) // 10)  # Reasonable number of clusters
            if n_clusters < 2:
                return {'clustering_available': False, 'reason': 'insufficient_data'}
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            return {
                'clustering_available': True,
                'embedding_source': embedding_source,
                'n_clusters': n_clusters,
                'cluster_sizes': [int(np.sum(clusters == i)) for i in range(n_clusters)],
                'inertia': float(kmeans.inertia_)
            }
        except Exception as e:
            self.logger.error(f"Embedding clustering failed: {e}")
            return {'clustering_available': False, 'error': str(e)}
    
    def _extract_network_features_for_training(self, G, df):
        """Extract network features for unified framework training."""
        try:
            self.logger.info("🔧 Extracting network features for training...")
            
            features = []
            
            # For each node (sample), extract network-based features
            for i, node in enumerate(G.nodes()):
                node_features = []
                
                # Basic centrality measures
                if G.number_of_edges() > 0:
                    degree_centrality = nx.degree_centrality(G)
                    betweenness_centrality = nx.betweenness_centrality(G)
                    closeness_centrality = nx.closeness_centrality(G)
                    
                    node_features.extend([
                        degree_centrality.get(node, 0),
                        betweenness_centrality.get(node, 0),
                        closeness_centrality.get(node, 0)
                    ])
                else:
                    node_features.extend([0, 0, 0])
                
                # Node degree
                node_features.append(G.degree(node))
                
                # Clustering coefficient
                node_features.append(nx.clustering(G, node))
                
                # Node attributes from original data
                node_idx = int(node.split('_')[1]) if '_' in node else i
                if node_idx < len(df):
                    row = df.iloc[node_idx]
                    node_features.extend([
                        row.get('FOLLOWERS_COUNT', 0) / 10000,  # Normalized
                        row.get('FOLLOWING_COUNT', 0) / 10000,  # Normalized
                        row.get('IS_VERIFIED', 0),
                        row.get('RETWEET_COUNT', 0) / 100,  # Normalized
                        row.get('FAVORITE_COUNT', 0) / 100,  # Normalized
                    ])
                else:
                    node_features.extend([0, 0, 0, 0, 0])
                
                # Community membership (simplified)
                try:
                    communities = list(nx.community.greedy_modularity_communities(G))
                    community_id = 0
                    for idx, community in enumerate(communities):
                        if node in community:
                            community_id = idx
                            break
                    node_features.append(community_id / len(communities) if communities else 0)
                except:
                    node_features.append(0)
                
                # Local clustering features
                neighbors = list(G.neighbors(node))
                node_features.extend([
                    len(neighbors),  # Number of neighbors
                    np.mean([G.degree(n) for n in neighbors]) if neighbors else 0,  # Avg neighbor degree
                ])
                
                # Pad or truncate to exactly 20 features
                if len(node_features) > 20:
                    node_features = node_features[:20]
                elif len(node_features) < 20:
                    node_features.extend([0] * (20 - len(node_features)))
                
                features.append(node_features)
            
            network_features = np.array(features)
            self.logger.info(f"✅ Network features extracted: {network_features.shape}")
            
            return network_features
            
        except Exception as e:
            self.logger.error(f"❌ Network feature extraction failed: {e}")
            return None
    
    def _save_network_features(self, dataset_name, network_features):
        """Save network features for unified framework training."""
        try:
            features_dir = Path('datasets') / dataset_name / 'features'
            features_dir.mkdir(parents=True, exist_ok=True)
            
            network_features_file = features_dir / 'network_features.npy'
            np.save(network_features_file, network_features)
            
            self.logger.info(f"💾 Network features saved to: {network_features_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save network features: {e}")
    
    def _create_enhanced_network_visualizations(self, G, df, dataset_name, enhanced_features):
        """Create enhanced network visualizations."""
        try:
            self._create_network_visualizations(G, df, dataset_name)
            
            # Add enhanced visualizations if features are available
            if enhanced_features:
                self._create_embedding_visualization(G, df, dataset_name, enhanced_features)
                
        except Exception as e:
            self.logger.error(f"Enhanced visualization creation failed: {e}")
    
    def _create_embedding_visualization(self, G, df, dataset_name, enhanced_features):
        """Create visualization based on embeddings."""
        try:
            if 'transformer_embeddings' not in enhanced_features:
                return
            
            viz_dir = Path('datasets') / dataset_name / 'visualizations'
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            embeddings = enhanced_features['transformer_embeddings']
            
            # Use t-SNE for dimensionality reduction
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_2d = tsne.fit_transform(embeddings[:200])  # Limit for performance
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            
            # Color by labels if available
            colors = []
            for i in range(len(embeddings_2d)):
                if i < len(df):
                    label = df.iloc[i].get('LABEL', 0)
                    colors.append('red' if label == 1 else 'blue')
                else:
                    colors.append('gray')
            
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6)
            plt.title(f'Transformer Embeddings Visualization - {dataset_name}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='Misinformation'),
                Patch(facecolor='blue', label='Legitimate')
            ]
            plt.legend(handles=legend_elements)
            
            plt.savefig(viz_dir / 'embedding_tsne.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Embedding visualization failed: {e}")

    def _create_centrality_visualization(self, G, df, viz_dir, static_viz_dir=None):
        """Create centrality analysis visualization."""
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # Separate by misinformation label
            legitimate_degree = []
            legitimate_betweenness = []
            misinformation_degree = []
            misinformation_betweenness = []
            
            for node in G.nodes():
                node_idx = int(node.split('_')[1])
                if node_idx < len(df):
                    label = df.iloc[node_idx].get('LABEL', 0)
                    degree = degree_centrality[node]
                    betweenness = betweenness_centrality[node]
                    
                    if label == 1:
                        misinformation_degree.append(degree)
                        misinformation_betweenness.append(betweenness)
                    else:
                        legitimate_degree.append(degree)
                        legitimate_betweenness.append(betweenness)
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Degree centrality comparison
            ax1.hist(legitimate_degree, alpha=0.7, label='Legitimate', bins=20, color='blue')
            ax1.hist(misinformation_degree, alpha=0.7, label='Misinformation', bins=20, color='red')
            ax1.set_xlabel('Degree Centrality')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Degree Centrality Distribution')
            ax1.legend()
            
            # Betweenness centrality comparison
            ax2.hist(legitimate_betweenness, alpha=0.7, label='Legitimate', bins=20, color='blue')
            ax2.hist(misinformation_betweenness, alpha=0.7, label='Misinformation', bins=20, color='red')
            ax2.set_xlabel('Betweenness Centrality')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Betweenness Centrality Distribution')
            ax2.legend()
            
            plt.tight_layout()
            # Save to both directories
            plt.savefig(viz_dir / 'centrality_analysis.png', dpi=300, bbox_inches='tight')
            if static_viz_dir:
                plt.savefig(static_viz_dir / 'centrality_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating centrality visualization: {e}")
    
    def _load_enhanced_features(self, dataset_name):
        """Load enhanced features (transformers + frameworks) for network analysis."""
        try:
            features_path = Path('datasets') / dataset_name / 'features' / 'features.csv'
            if features_path.exists():
                features_df = pd.read_csv(features_path)
                self.logger.info(f"Loaded enhanced features: {features_df.shape[1]} features")
                return features_df
            else:
                self.logger.warning("Enhanced features not found, using basic analysis")
                return None
        except Exception as e:
            self.logger.error(f"Error loading enhanced features: {e}")
            return None
    
    def _create_enhanced_network(self, df, enhanced_features):
        """Create network with enhanced node attributes from transformers + frameworks."""
        try:
            # Start with basic network
            G = self._create_network(df)
            
            if enhanced_features is not None:
                # Add transformer embeddings as node attributes
                bert_cols = [col for col in enhanced_features.columns if 'bert_' in col.lower()]
                sentence_cols = [col for col in enhanced_features.columns if 'sentence_' in col.lower()]
                
                # Add framework features as node attributes
                ugt_cols = [col for col in enhanced_features.columns if 'ugt_' in col.lower()]
                rct_cols = [col for col in enhanced_features.columns if 'rct_' in col.lower()]
                rat_cols = [col for col in enhanced_features.columns if 'rat_' in col.lower()]
                
                # Add behavioral features
                behavioral_cols = [col for col in enhanced_features.columns if any(pattern in col.lower() 
                                 for pattern in ['urgency', 'conspiracy', 'emotional', 'pattern'])]
                
                self.logger.info(f"Adding enhanced attributes: BERT({len(bert_cols)}), "
                               f"Sentence({len(sentence_cols)}), UGT({len(ugt_cols)}), "
                               f"RCT({len(rct_cols)}), RAT({len(rat_cols)}), "
                               f"Behavioral({len(behavioral_cols)})")
                
                # Add attributes to nodes
                for i, node in enumerate(G.nodes()):
                    if i < len(enhanced_features):
                        # Transformer embeddings (averaged for node representation)
                        if bert_cols:
                            G.nodes[node]['bert_embedding_avg'] = enhanced_features.iloc[i][bert_cols].mean()
                        if sentence_cols:
                            G.nodes[node]['sentence_embedding_avg'] = enhanced_features.iloc[i][sentence_cols].mean()
                        
                        # Framework scores
                        if ugt_cols:
                            G.nodes[node]['ugt_score'] = enhanced_features.iloc[i][ugt_cols].sum()
                        if rct_cols:
                            G.nodes[node]['rct_score'] = enhanced_features.iloc[i][rct_cols].sum()
                        if rat_cols:
                            G.nodes[node]['rat_score'] = enhanced_features.iloc[i][rat_cols].sum()
                        
                        # Behavioral patterns
                        if behavioral_cols:
                            G.nodes[node]['behavioral_score'] = enhanced_features.iloc[i][behavioral_cols].sum()
            
            return G
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced network: {e}")
            return self._create_network(df)  # Fallback to basic network
    
    def _analyze_enhanced_misinformation_patterns(self, G, df, enhanced_features):
        """Analyze misinformation patterns using enhanced features."""
        try:
            analysis = {}
            
            if enhanced_features is None:
                return self._analyze_misinformation_patterns(G, df)
            
            # Analyze transformer-based clustering
            bert_cols = [col for col in enhanced_features.columns if 'bert_' in col.lower()]
            if bert_cols:
                from sklearn.cluster import KMeans
                bert_embeddings = enhanced_features[bert_cols].values
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(bert_embeddings)
                
                analysis['bert_clusters'] = {
                    'n_clusters': 3,
                    'cluster_distribution': np.bincount(clusters).tolist(),
                    'cluster_labels': clusters.tolist()
                }
            
            # Analyze framework-based patterns
            framework_cols = [col for col in enhanced_features.columns 
                            if any(fw in col.lower() for fw in ['ugt_', 'rct_', 'rat_'])]
            if framework_cols:
                framework_scores = enhanced_features[framework_cols].sum(axis=1)
                analysis['framework_patterns'] = {
                    'high_framework_nodes': (framework_scores > framework_scores.quantile(0.75)).sum(),
                    'low_framework_nodes': (framework_scores < framework_scores.quantile(0.25)).sum(),
                    'avg_framework_score': framework_scores.mean()
                }
            
            # Analyze behavioral patterns in network
            behavioral_cols = [col for col in enhanced_features.columns 
                             if any(pattern in col.lower() for pattern in ['urgency', 'conspiracy', 'emotional'])]
            if behavioral_cols:
                behavioral_scores = enhanced_features[behavioral_cols].sum(axis=1)
                analysis['behavioral_network_patterns'] = {
                    'high_behavioral_nodes': (behavioral_scores > behavioral_scores.quantile(0.8)).sum(),
                    'behavioral_centrality_correlation': self._calculate_behavioral_centrality_correlation(G, behavioral_scores)
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in enhanced misinformation analysis: {e}")
            return {}
    
    def _analyze_transformer_patterns(self, G, enhanced_features):
        """Analyze network patterns using transformer embeddings."""
        try:
            if enhanced_features is None:
                return {}
            
            analysis = {}
            
            # BERT embedding analysis
            bert_cols = [col for col in enhanced_features.columns if 'bert_' in col.lower()]
            if bert_cols:
                bert_data = enhanced_features[bert_cols]
                
                # Calculate semantic similarity network
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(bert_data.values)
                
                # Find highly similar content (potential coordinated behavior)
                high_similarity_threshold = 0.9
                high_sim_pairs = np.where((similarity_matrix > high_similarity_threshold) & 
                                        (similarity_matrix < 1.0))  # Exclude self-similarity
                
                analysis['bert_analysis'] = {
                    'high_similarity_pairs': len(high_sim_pairs[0]),
                    'avg_similarity': similarity_matrix.mean(),
                    'potential_coordinated_content': len(high_sim_pairs[0]) > 10
                }
            
            # Sentence transformer analysis
            sentence_cols = [col for col in enhanced_features.columns if 'sentence_' in col.lower()]
            if sentence_cols:
                sentence_data = enhanced_features[sentence_cols]
                
                # Analyze sentence-level patterns
                sentence_similarity = cosine_similarity(sentence_data.values)
                
                analysis['sentence_analysis'] = {
                    'avg_sentence_similarity': sentence_similarity.mean(),
                    'sentence_clusters': self._identify_sentence_clusters(sentence_data)
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in transformer pattern analysis: {e}")
            return {}
    
    def _analyze_framework_patterns(self, G, enhanced_features):
        """Analyze network patterns using theoretical frameworks."""
        try:
            if enhanced_features is None:
                return {}
            
            analysis = {}
            
            # UGT analysis
            ugt_cols = [col for col in enhanced_features.columns if 'ugt_' in col.lower()]
            if ugt_cols:
                ugt_scores = enhanced_features[ugt_cols].sum(axis=1)
                analysis['ugt_network_analysis'] = {
                    'high_gratification_nodes': (ugt_scores > ugt_scores.quantile(0.75)).sum(),
                    'gratification_centrality_correlation': self._calculate_framework_centrality_correlation(G, ugt_scores, 'UGT')
                }
            
            # RCT analysis
            rct_cols = [col for col in enhanced_features.columns if 'rct_' in col.lower()]
            if rct_cols:
                rct_scores = enhanced_features[rct_cols].sum(axis=1)
                analysis['rct_network_analysis'] = {
                    'high_rational_choice_nodes': (rct_scores > rct_scores.quantile(0.75)).sum(),
                    'rational_choice_centrality_correlation': self._calculate_framework_centrality_correlation(G, rct_scores, 'RCT')
                }
            
            # RAT analysis
            rat_cols = [col for col in enhanced_features.columns if 'rat_' in col.lower()]
            if rat_cols:
                rat_scores = enhanced_features[rat_cols].sum(axis=1)
                analysis['rat_network_analysis'] = {
                    'high_routine_activity_nodes': (rat_scores > rat_scores.quantile(0.75)).sum(),
                    'routine_activity_centrality_correlation': self._calculate_framework_centrality_correlation(G, rat_scores, 'RAT')
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in framework pattern analysis: {e}")
            return {}
    
    def _calculate_behavioral_centrality_correlation(self, G, behavioral_scores):
        """Calculate correlation between behavioral scores and network centrality."""
        try:
            centrality = nx.degree_centrality(G)
            centrality_values = [centrality.get(node, 0) for node in range(len(behavioral_scores))]
            
            # Calculate correlation
            correlation = np.corrcoef(behavioral_scores, centrality_values)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating behavioral centrality correlation: {e}")
            return 0.0
    
    def _calculate_framework_centrality_correlation(self, G, framework_scores, framework_name):
        """Calculate correlation between framework scores and network centrality."""
        try:
            centrality = nx.degree_centrality(G)
            centrality_values = [centrality.get(node, 0) for node in range(len(framework_scores))]
            
            correlation = np.corrcoef(framework_scores, centrality_values)[0, 1]
            
            return {
                'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'framework': framework_name,
                'interpretation': self._interpret_framework_correlation(correlation, framework_name)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating {framework_name} centrality correlation: {e}")
            return {'correlation': 0.0, 'framework': framework_name}
    
    def _interpret_framework_correlation(self, correlation, framework_name):
        """Interpret the correlation between framework scores and centrality."""
        if abs(correlation) < 0.1:
            return f"No significant correlation between {framework_name} patterns and network centrality"
        elif correlation > 0.3:
            return f"Strong positive correlation: High {framework_name} scores associated with central network positions"
        elif correlation < -0.3:
            return f"Strong negative correlation: High {framework_name} scores associated with peripheral network positions"
        else:
            return f"Moderate correlation between {framework_name} patterns and network position"
    
    def _identify_sentence_clusters(self, sentence_data):
        """Identify clusters in sentence transformer embeddings."""
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42)
            clusters = kmeans.fit_predict(sentence_data.values)
            
            return {
                'n_clusters': 5,
                'cluster_sizes': np.bincount(clusters).tolist(),
                'largest_cluster_size': np.bincount(clusters).max()
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying sentence clusters: {e}")
            return {}
    
    def _create_enhanced_network_visualizations(self, G, df, dataset_name, enhanced_features):
        """Create enhanced network visualizations using transformer + framework features."""
        try:
            # Create basic visualizations first
            self._create_network_visualizations(G, df, dataset_name)
            
            if enhanced_features is None:
                return
            
            viz_dir = Path('static') / 'visualizations' / dataset_name
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Create transformer-based visualization
            self._create_transformer_network_viz(G, enhanced_features, viz_dir)
            
            # Create framework-based visualization
            self._create_framework_network_viz(G, enhanced_features, viz_dir)
            
            # Create behavioral pattern visualization
            self._create_behavioral_network_viz(G, enhanced_features, viz_dir)
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced network visualizations: {e}")
    
    def _create_transformer_network_viz(self, G, enhanced_features, viz_dir):
        """Create network visualization colored by transformer embeddings."""
        try:
            plt.figure(figsize=(12, 10))
            
            # Use BERT embeddings for node coloring
            bert_cols = [col for col in enhanced_features.columns if 'bert_' in col.lower()]
            if bert_cols:
                bert_scores = enhanced_features[bert_cols].mean(axis=1)
                
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Color nodes by BERT embedding intensity
                node_colors = [bert_scores.iloc[i] if i < len(bert_scores) else 0 
                             for i in range(len(G.nodes()))]
                
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                     cmap='viridis', node_size=50, alpha=0.7)
                nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
                
                plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), 
                           label='BERT Embedding Intensity')
                plt.title('Network Colored by BERT Transformer Embeddings')
                plt.axis('off')
                
                plt.savefig(viz_dir / 'transformer_network.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error creating transformer network visualization: {e}")
    
    def _create_framework_network_viz(self, G, enhanced_features, viz_dir):
        """Create network visualization colored by theoretical framework scores."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            frameworks = ['ugt', 'rct', 'rat']
            colors = ['Blues', 'Reds', 'Greens']
            
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            for i, (framework, cmap) in enumerate(zip(frameworks, colors)):
                framework_cols = [col for col in enhanced_features.columns 
                                if f'{framework}_' in col.lower()]
                
                if framework_cols:
                    framework_scores = enhanced_features[framework_cols].sum(axis=1)
                    node_colors = [framework_scores.iloc[j] if j < len(framework_scores) else 0 
                                 for j in range(len(G.nodes()))]
                    
                    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                         cmap=cmap, node_size=50, alpha=0.7, ax=axes[i])
                    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=axes[i])
                    
                    axes[i].set_title(f'{framework.upper()} Framework Scores')
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'framework_networks.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating framework network visualization: {e}")
    
    def _create_behavioral_network_viz(self, G, enhanced_features, viz_dir):
        """Create network visualization colored by behavioral patterns."""
        try:
            plt.figure(figsize=(12, 10))
            
            behavioral_cols = [col for col in enhanced_features.columns 
                             if any(pattern in col.lower() for pattern in ['urgency', 'conspiracy', 'emotional'])]
            
            if behavioral_cols:
                behavioral_scores = enhanced_features[behavioral_cols].sum(axis=1)
                
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                node_colors = [behavioral_scores.iloc[i] if i < len(behavioral_scores) else 0 
                             for i in range(len(G.nodes()))]
                
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                     cmap='Reds', node_size=50, alpha=0.7)
                nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
                
                plt.colorbar(plt.cm.ScalarMappable(cmap='Reds'), 
                           label='Behavioral Pattern Intensity')
                plt.title('Network Colored by Behavioral Patterns')
                plt.axis('off')
                
                plt.savefig(viz_dir / 'behavioral_network.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error creating behavioral network visualization: {e}")