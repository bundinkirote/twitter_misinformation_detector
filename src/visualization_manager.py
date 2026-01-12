"""
Visualization Manager for Unified Framework
Ensures all visualizations are properly generated and accessible
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from datetime import datetime

class VisualizationManager:
    """Manages all visualizations for the unified framework."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_unified_framework_visualizations(self, dataset_name, results):
        """Create comprehensive visualizations for unified framework results."""
        try:
            self.logger.info(f"🎨 Creating unified framework visualizations for {dataset_name}")
            
            # Create visualization directories
            viz_dir = Path('static') / 'plots' / dataset_name
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Framework comparison chart
            self._create_framework_comparison_chart(results, viz_dir)
            
            # Individual framework ranking
            if 'individual_framework_analysis' in results:
                self._create_individual_framework_chart(results['individual_framework_analysis'], viz_dir)
            
            # Algorithm performance comparison
            self._create_algorithm_performance_chart(results['results'], viz_dir)
            
            # Hyperparameter tuning results
            self._create_hyperparameter_tuning_chart(results['results'], viz_dir)
            
            # Category-based performance
            if 'comparative_analysis' in results:
                self._create_category_performance_chart(results['comparative_analysis'], viz_dir)
            
            # Zero-shot comparison
            if 'zero_shot_results' in results:
                self._create_zero_shot_comparison_chart(results, viz_dir)
            
            self.logger.info("✅ All unified framework visualizations created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating unified framework visualizations: {e}")
            return False
    
    def _create_framework_comparison_chart(self, results, viz_dir):
        """Create framework effectiveness comparison chart."""
        try:
            if 'comparative_analysis' not in results or 'framework_effectiveness' not in results['comparative_analysis']:
                return
            
            framework_data = results['comparative_analysis']['framework_effectiveness']
            
            frameworks = list(framework_data.keys())
            f1_scores = [data['mean_f1'] for data in framework_data.values()]
            std_scores = [data.get('std_f1', 0) for data in framework_data.values()]
            
            # Create bar chart
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(frameworks)), f1_scores, yerr=std_scores, 
                          capsize=5, alpha=0.8, color=sns.color_palette("husl", len(frameworks)))
            
            plt.xlabel('Framework Combinations')
            plt.ylabel('Mean F1 Score')
            plt.title('Framework Effectiveness Comparison')
            plt.xticks(range(len(frameworks)), [f.replace('_', ' ').title() for f in frameworks], 
                      rotation=45, ha='right')
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, f1_scores)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'framework_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating framework comparison chart: {e}")
    
    def _create_individual_framework_chart(self, framework_analysis, viz_dir):
        """Create individual framework ranking chart."""
        try:
            if 'framework_ranking' not in framework_analysis:
                return
            
            ranking_data = framework_analysis['framework_ranking']
            
            frameworks = [item['framework'].upper() for item in ranking_data]
            f1_scores = [item['mean_f1'] for item in ranking_data]
            ranks = [item['rank'] for item in ranking_data]
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, 6))
            colors = ['#28a745', '#ffc107', '#dc3545'][:len(frameworks)]
            bars = plt.barh(range(len(frameworks)), f1_scores, color=colors, alpha=0.8)
            
            plt.xlabel('Mean F1 Score')
            plt.ylabel('Theoretical Framework')
            plt.title('Individual Framework Impact Ranking')
            plt.yticks(range(len(frameworks)), frameworks)
            
            # Add value labels
            for i, (bar, score, rank) in enumerate(zip(bars, f1_scores, ranks)):
                plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                        f'#{rank}: {score:.3f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'individual_framework_ranking.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating individual framework chart: {e}")
    
    def _create_algorithm_performance_chart(self, results_data, viz_dir):
        """Create algorithm performance comparison across frameworks."""
        try:
            # Collect algorithm performance data
            algorithm_data = {}
            
            for combo_name, combo_results in results_data.items():
                if 'algorithms' not in combo_results:
                    continue
                    
                for algo_name, algo_results in combo_results['algorithms'].items():
                    if 'error' not in algo_results and 'metrics' in algo_results:
                        if algo_name not in algorithm_data:
                            algorithm_data[algo_name] = []
                        algorithm_data[algo_name].append({
                            'framework': combo_name,
                            'f1_score': algo_results['metrics']['f1_score'],
                            'accuracy': algo_results['metrics']['accuracy']
                        })
            
            if not algorithm_data:
                return
            
            # Create grouped bar chart
            algorithms = list(algorithm_data.keys())
            frameworks = list(set([item['framework'] for algo_data in algorithm_data.values() 
                                 for item in algo_data]))
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # F1 Score comparison
            x = np.arange(len(algorithms))
            width = 0.8 / len(frameworks)
            
            for i, framework in enumerate(frameworks):
                f1_scores = []
                for algo in algorithms:
                    scores = [item['f1_score'] for item in algorithm_data[algo] 
                             if item['framework'] == framework]
                    f1_scores.append(scores[0] if scores else 0)
                
                ax1.bar(x + i * width, f1_scores, width, 
                       label=framework.replace('_', ' ').title(), alpha=0.8)
            
            ax1.set_xlabel('Algorithms')
            ax1.set_ylabel('F1 Score')
            ax1.set_title('Algorithm F1 Performance Across Frameworks')
            ax1.set_xticks(x + width * (len(frameworks) - 1) / 2)
            ax1.set_xticklabels([algo.replace('_', ' ').title() for algo in algorithms], 
                               rotation=45, ha='right')
            ax1.legend()
            
            # Accuracy comparison
            for i, framework in enumerate(frameworks):
                accuracies = []
                for algo in algorithms:
                    scores = [item['accuracy'] for item in algorithm_data[algo] 
                             if item['framework'] == framework]
                    accuracies.append(scores[0] if scores else 0)
                
                ax2.bar(x + i * width, accuracies, width, 
                       label=framework.replace('_', ' ').title(), alpha=0.8)
            
            ax2.set_xlabel('Algorithms')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Algorithm Accuracy Across Frameworks')
            ax2.set_xticks(x + width * (len(frameworks) - 1) / 2)
            ax2.set_xticklabels([algo.replace('_', ' ').title() for algo in algorithms], 
                               rotation=45, ha='right')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'algorithm_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating algorithm performance chart: {e}")
    
    def _create_hyperparameter_tuning_chart(self, results_data, viz_dir):
        """Create hyperparameter tuning improvement chart."""
        try:
            tuning_improvements = []
            
            for combo_name, combo_results in results_data.items():
                if 'algorithms' not in combo_results:
                    continue
                    
                for algo_name, algo_results in combo_results['algorithms'].items():
                    if ('error' not in algo_results and 
                        'hyperparameter_tuning' in algo_results and 
                        'best_cv_score' in algo_results['hyperparameter_tuning']):
                        
                        tuned_score = algo_results['hyperparameter_tuning']['best_cv_score']
                        actual_score = algo_results['metrics']['f1_score']
                        
                        tuning_improvements.append({
                            'algorithm': algo_name,
                            'framework': combo_name,
                            'tuned_cv_score': tuned_score,
                            'actual_score': actual_score,
                            'improvement': actual_score - tuned_score
                        })
            
            if not tuning_improvements:
                return
            
            # Create improvement chart
            df = pd.DataFrame(tuning_improvements)
            
            plt.figure(figsize=(12, 8))
            
            # Group by algorithm
            algorithms = df['algorithm'].unique()
            x = np.arange(len(algorithms))
            width = 0.35
            
            tuned_scores = [df[df['algorithm'] == algo]['tuned_cv_score'].mean() 
                           for algo in algorithms]
            actual_scores = [df[df['algorithm'] == algo]['actual_score'].mean() 
                            for algo in algorithms]
            
            bars1 = plt.bar(x - width/2, tuned_scores, width, label='CV Score (Tuned)', alpha=0.8)
            bars2 = plt.bar(x + width/2, actual_scores, width, label='Test Score (Actual)', alpha=0.8)
            
            plt.xlabel('Algorithms')
            plt.ylabel('Score')
            plt.title('Hyperparameter Tuning: CV vs Test Performance')
            plt.xticks(x, [algo.replace('_', ' ').title() for algo in algorithms], 
                      rotation=45, ha='right')
            plt.legend()
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating hyperparameter tuning chart: {e}")
    
    def _create_category_performance_chart(self, comparative_analysis, viz_dir):
        """Create category-based performance chart."""
        try:
            if 'framework_effectiveness' not in comparative_analysis:
                return
            
            framework_data = comparative_analysis['framework_effectiveness']
            
            # Categorize frameworks
            categories = {
                'Baseline': [],
                'Transformer Enhanced': [],
                'Individual Frameworks': [],
                'Framework Enhanced': [],
                'Network Enhanced': [],
                'Complete Integration': []
            }
            
            for framework, data in framework_data.items():
                if 'baseline' in framework:
                    categories['Baseline'].append(data['mean_f1'])
                elif 'transformer' in framework and 'only' in framework:
                    categories['Transformer Enhanced'].append(data['mean_f1'])
                elif 'only_embeddings' in framework:
                    categories['Individual Frameworks'].append(data['mean_f1'])
                elif 'enhanced' in framework:
                    categories['Framework Enhanced'].append(data['mean_f1'])
                elif 'network' in framework:
                    categories['Network Enhanced'].append(data['mean_f1'])
                elif 'full' in framework:
                    categories['Complete Integration'].append(data['mean_f1'])
            
            # Calculate category averages
            category_means = {}
            category_stds = {}
            
            for category, scores in categories.items():
                if scores:
                    category_means[category] = np.mean(scores)
                    category_stds[category] = np.std(scores) if len(scores) > 1 else 0
            
            if not category_means:
                return
            
            # Create chart
            plt.figure(figsize=(12, 8))
            
            categories_list = list(category_means.keys())
            means = list(category_means.values())
            stds = [category_stds[cat] for cat in categories_list]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories_list)))
            bars = plt.bar(range(len(categories_list)), means, yerr=stds, 
                          capsize=5, alpha=0.8, color=colors)
            
            plt.xlabel('Framework Categories')
            plt.ylabel('Mean F1 Score')
            plt.title('Performance by Framework Category')
            plt.xticks(range(len(categories_list)), categories_list, rotation=45, ha='right')
            
            # Add value labels
            for bar, mean in zip(bars, means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'category_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating category performance chart: {e}")
    
    def _create_zero_shot_comparison_chart(self, results, viz_dir):
        """Create zero-shot vs trained model comparison."""
        try:
            if 'zero_shot_results' not in results:
                return
            
            zero_shot_data = results['zero_shot_results']
            
            # Get best traditional model performance
            best_traditional = None
            if 'comparative_analysis' in results and 'best_overall' in results['comparative_analysis']:
                best_traditional = results['comparative_analysis']['best_overall']
            
            if not best_traditional:
                return
            
            # Create comparison
            methods = ['Zero-shot BART', 'Best Traditional Model']
            f1_scores = [
                zero_shot_data.get('f1_score', 0),
                best_traditional['score']
            ]
            accuracies = [
                zero_shot_data.get('accuracy', 0),
                best_traditional['accuracy']
            ]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # F1 Score comparison
            bars1 = ax1.bar(methods, f1_scores, color=['#ff7f0e', '#1f77b4'], alpha=0.8)
            ax1.set_ylabel('F1 Score')
            ax1.set_title('F1 Score: Zero-shot vs Traditional')
            ax1.set_ylim(0, 1)
            
            for bar, score in zip(bars1, f1_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Accuracy comparison
            bars2 = ax2.bar(methods, accuracies, color=['#ff7f0e', '#1f77b4'], alpha=0.8)
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy: Zero-shot vs Traditional')
            ax2.set_ylim(0, 1)
            
            for bar, score in zip(bars2, accuracies):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'zero_shot_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating zero-shot comparison chart: {e}")
    
    def create_network_visualizations(self, dataset_name, network_results):
        """Create network analysis visualizations."""
        try:
            self.logger.info(f"🕸️ Creating network visualizations for {dataset_name}")
            
            viz_dir = Path('static') / 'network_visualizations' / dataset_name
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Network metrics overview
            self._create_network_metrics_chart(network_results, viz_dir)
            
            # Framework analysis visualization
            if network_results.get('framework_analysis'):
                self._create_framework_network_chart(network_results['framework_analysis'], viz_dir)
            
            # Transformer analysis visualization
            if network_results.get('transformer_analysis'):
                self._create_transformer_network_chart(network_results['transformer_analysis'], viz_dir)
            
            self.logger.info("✅ Network visualizations created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating network visualizations: {e}")
            return False
    
    def _create_network_metrics_chart(self, network_results, viz_dir):
        """Create network metrics overview chart."""
        try:
            metrics = {
                'Nodes': network_results.get('network_properties', {}).get('nodes', 0),
                'Edges': network_results.get('network_properties', {}).get('edges', 0),
                'Communities': network_results.get('framework_analysis', {}).get('community_count', 0),
                'Density': network_results.get('framework_analysis', {}).get('network_density', 0)
            }
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Nodes and Edges
            ax1.bar(['Nodes', 'Edges'], [metrics['Nodes'], metrics['Edges']], 
                   color=['#1f77b4', '#ff7f0e'], alpha=0.8)
            ax1.set_title('Network Size')
            ax1.set_ylabel('Count')
            
            # Communities
            ax2.bar(['Communities'], [metrics['Communities']], color=['#2ca02c'], alpha=0.8)
            ax2.set_title('Community Structure')
            ax2.set_ylabel('Count')
            
            # Density
            ax3.bar(['Network Density'], [metrics['Density']], color=['#d62728'], alpha=0.8)
            ax3.set_title('Network Density')
            ax3.set_ylabel('Density')
            ax3.set_ylim(0, 1)
            
            # Summary text
            ax4.axis('off')
            summary_text = f"""Network Analysis Summary:
            
• Total Nodes: {metrics['Nodes']:,}
• Total Edges: {metrics['Edges']:,}
• Communities: {metrics['Communities']}
• Network Density: {metrics['Density']:.3f}

Enhanced with:
• Transformer embeddings
• Theoretical framework analysis
• 20 network features extracted
"""
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'network_metrics_overview.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating network metrics chart: {e}")
    
    def _create_framework_network_chart(self, framework_analysis, viz_dir):
        """Create framework-specific network analysis chart."""
        try:
            if not framework_analysis.get('framework_analysis_available'):
                return
            
            metrics = {
                'Network Density': framework_analysis.get('network_density', 0),
                'Clustering Coefficient': framework_analysis.get('clustering_coefficient', 0),
                'Modularity': framework_analysis.get('modularity', 0)
            }
            
            plt.figure(figsize=(10, 6))
            
            bars = plt.bar(metrics.keys(), metrics.values(), 
                          color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
            
            plt.title('Theoretical Framework Network Analysis')
            plt.ylabel('Metric Value')
            plt.ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, metrics.values()):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(viz_dir / 'framework_network_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating framework network chart: {e}")
    
    def _create_transformer_network_chart(self, transformer_analysis, viz_dir):
        """Create transformer-specific network analysis chart."""
        try:
            if not transformer_analysis.get('transformer_analysis_available'):
                return
            
            metrics = {
                'Avg Similarity': transformer_analysis.get('avg_similarity', 0),
                'Similarity Std': transformer_analysis.get('similarity_std', 0),
                'High Similarity Pairs': transformer_analysis.get('high_similarity_pairs', 0) / 100  # Normalize
            }
            
            plt.figure(figsize=(10, 6))
            
            bars = plt.bar(metrics.keys(), metrics.values(), 
                          color=['#9467bd', '#8c564b', '#e377c2'], alpha=0.8)
            
            plt.title('Transformer Embeddings Network Analysis')
            plt.ylabel('Metric Value')
            
            # Add value labels
            for bar, key, value in zip(bars, metrics.keys(), metrics.values()):
                if key == 'High Similarity Pairs':
                    label = f'{int(value * 100)}'  # Show original count
                else:
                    label = f'{value:.3f}'
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        label, ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(viz_dir / 'transformer_network_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating transformer network chart: {e}")