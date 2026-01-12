"""
Insights Generator Module

This module provides comprehensive insights generation capabilities for the misinformation
detection system. It analyzes results at each pipeline stage and generates contextual
insights, recommendations, and actionable intelligence for system optimization and
decision-making support.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

class InsightsGenerator:
    """
    Insights Generation Class
    
    Implements comprehensive insights generation for different stages of the machine
    learning pipeline. Provides contextual analysis, recommendations, and actionable
    intelligence based on dataset characteristics, model performance, and system
    behavior for informed decision-making and optimization.
    """
    
    def __init__(self):
        """Initialize the insights generator."""
        self.insights_cache = {}
        
    def generate_dataset_insights(self, dataset_stats: Dict[str, Any], dataset_name: str) -> List[Dict[str, str]]:
        """
        Generate insights for dataset overview page.
        
        Args:
            dataset_stats: Dictionary containing dataset statistics
            dataset_name: Name of the dataset
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        if not dataset_stats:
            return insights
            
        # Sample size insights
        total_samples = dataset_stats.get('total_samples', 0)
        if total_samples > 0:
            if total_samples < 1000:
                insights.append({
                    "type": "warning",
                    "icon": "fas fa-exclamation-triangle",
                    "title": "Small Dataset Size",
                    "description": f"Dataset contains {total_samples:,} samples. Consider collecting more data for better model performance.",
                    "recommendation": "Aim for at least 1,000+ samples for robust misinformation detection models.",
                    "priority": "high"
                })
            elif total_samples < 5000:
                insights.append({
                    "type": "info",
                    "icon": "fas fa-info-circle",
                    "title": "Moderate Dataset Size",
                    "description": f"Dataset contains {total_samples:,} samples. This is adequate for initial model training.",
                    "recommendation": "Consider expanding dataset for production deployment.",
                    "priority": "medium"
                })
            else:
                insights.append({
                    "type": "success",
                    "icon": "fas fa-check-circle",
                    "title": "Robust Dataset Size",
                    "description": f"Dataset contains {total_samples:,} samples. Excellent size for reliable model training.",
                    "recommendation": "Dataset size is optimal for comprehensive analysis.",
                    "priority": "low"
                })
        
        # Class balance insights
        misinformation_rate = dataset_stats.get('misinformation_rate', 0)
        if misinformation_rate > 0:
            if misinformation_rate < 20 or misinformation_rate > 80:
                insights.append({
                    "type": "warning",
                    "icon": "fas fa-balance-scale",
                    "title": "Class Imbalance Detected",
                    "description": f"Misinformation rate: {misinformation_rate:.1f}%. Significant class imbalance may affect model performance.",
                    "recommendation": "Consider using stratified sampling, SMOTE, or class weighting techniques.",
                    "priority": "high"
                })
            else:
                insights.append({
                    "type": "success",
                    "icon": "fas fa-balance-scale",
                    "title": "Balanced Dataset",
                    "description": f"Misinformation rate: {misinformation_rate:.1f}%. Good class balance for training.",
                    "recommendation": "Maintain this balance in train/test splits.",
                    "priority": "low"
                })
        
        # Data quality insights
        missing_values = dataset_stats.get('missing_values', 0)
        if missing_values > 0:
            missing_percentage = (missing_values / (total_samples * dataset_stats.get('features', 1))) * 100
            if missing_percentage > 10:
                insights.append({
                    "type": "warning",
                    "icon": "fas fa-exclamation-circle",
                    "title": "High Missing Data",
                    "description": f"{missing_percentage:.1f}% of data points are missing. This may impact model quality.",
                    "recommendation": "Implement robust imputation strategies or consider data collection improvements.",
                    "priority": "high"
                })
            else:
                insights.append({
                    "type": "info",
                    "icon": "fas fa-info-circle",
                    "title": "Minimal Missing Data",
                    "description": f"{missing_percentage:.1f}% missing data. Manageable with standard preprocessing.",
                    "recommendation": "Use simple imputation methods like mean/mode filling.",
                    "priority": "low"
                })
        
        # Feature insights
        num_features = dataset_stats.get('features', 0)
        if num_features > 0:
            if num_features < 10:
                insights.append({
                    "type": "info",
                    "icon": "fas fa-columns",
                    "title": "Limited Feature Set",
                    "description": f"Dataset has {num_features} features. Consider feature engineering for better performance.",
                    "recommendation": "Extract additional features like text length, sentiment, or behavioral patterns.",
                    "priority": "medium"
                })
            elif num_features > 100:
                insights.append({
                    "type": "warning",
                    "icon": "fas fa-columns",
                    "title": "High-Dimensional Dataset",
                    "description": f"Dataset has {num_features} features. Risk of overfitting and curse of dimensionality.",
                    "recommendation": "Consider feature selection, PCA, or regularization techniques.",
                    "priority": "medium"
                })
        
        return insights
    
    def generate_training_insights(self, training_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate insights for model training results.
        
        Args:
            training_results: Dictionary containing training results
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        if not training_results:
            return insights
        
        # Model performance insights
        best_model = training_results.get('best_model', {})
        best_score = best_model.get('f1_score', 0)
        
        if best_score > 0.9:
            insights.append({
                "type": "success",
                "icon": "fas fa-trophy",
                "title": "Excellent Model Performance",
                "description": f"Best F1-score: {best_score:.3f}. Outstanding performance for misinformation detection.",
                "recommendation": "Model is ready for deployment. Consider A/B testing in production.",
                "priority": "low"
            })
        elif best_score > 0.8:
            insights.append({
                "type": "success",
                "icon": "fas fa-thumbs-up",
                "title": "Good Model Performance",
                "description": f"Best F1-score: {best_score:.3f}. Solid performance suitable for most applications.",
                "recommendation": "Consider hyperparameter tuning for potential improvements.",
                "priority": "medium"
            })
        elif best_score > 0.7:
            insights.append({
                "type": "warning",
                "icon": "fas fa-chart-line",
                "title": "Moderate Model Performance",
                "description": f"Best F1-score: {best_score:.3f}. Performance is acceptable but has room for improvement.",
                "recommendation": "Try feature engineering, ensemble methods, or more data collection.",
                "priority": "high"
            })
        else:
            insights.append({
                "type": "danger",
                "icon": "fas fa-exclamation-triangle",
                "title": "Poor Model Performance",
                "description": f"Best F1-score: {best_score:.3f}. Model performance is below acceptable threshold.",
                "recommendation": "Revisit data quality, feature engineering, and model selection strategies.",
                "priority": "critical"
            })
        
        # Model comparison insights
        models_trained = training_results.get('models_trained', [])
        if len(models_trained) > 1:
            performance_variance = np.std([model.get('f1_score', 0) for model in models_trained])
            if performance_variance < 0.05:
                insights.append({
                    "type": "info",
                    "icon": "fas fa-equals",
                    "title": "Consistent Model Performance",
                    "description": f"Low variance ({performance_variance:.3f}) across models suggests stable dataset.",
                    "recommendation": "Choose the simplest model for better interpretability and faster inference.",
                    "priority": "low"
                })
            else:
                insights.append({
                    "type": "info",
                    "icon": "fas fa-chart-bar",
                    "title": "Variable Model Performance",
                    "description": f"High variance ({performance_variance:.3f}) suggests some models are better suited for this data.",
                    "recommendation": "Focus on the top-performing models and consider ensemble methods.",
                    "priority": "medium"
                })
        
        return insights
    
    def generate_evaluation_insights(self, evaluation_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate insights for model evaluation results.
        
        Args:
            evaluation_results: Dictionary containing evaluation results
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        if not evaluation_results:
            return insights
        
        # Precision vs Recall trade-off
        precision = evaluation_results.get('precision', 0)
        recall = evaluation_results.get('recall', 0)
        
        if precision > recall + 0.1:
            insights.append({
                "type": "info",
                "icon": "fas fa-crosshairs",
                "title": "High Precision, Lower Recall",
                "description": f"Precision: {precision:.3f}, Recall: {recall:.3f}. Model is conservative in flagging misinformation.",
                "recommendation": "Good for applications where false positives are costly. Consider lowering threshold for higher recall.",
                "priority": "medium"
            })
        elif recall > precision + 0.1:
            insights.append({
                "type": "info",
                "icon": "fas fa-net",
                "title": "High Recall, Lower Precision",
                "description": f"Precision: {precision:.3f}, Recall: {recall:.3f}. Model catches most misinformation but with false positives.",
                "recommendation": "Good for screening applications. Consider raising threshold or ensemble methods to improve precision.",
                "priority": "medium"
            })
        else:
            insights.append({
                "type": "success",
                "icon": "fas fa-balance-scale",
                "title": "Balanced Precision-Recall",
                "description": f"Precision: {precision:.3f}, Recall: {recall:.3f}. Well-balanced performance.",
                "recommendation": "Excellent balance for general misinformation detection applications.",
                "priority": "low"
            })
        
        # ROC-AUC insights
        roc_auc = evaluation_results.get('roc_auc', 0)
        if roc_auc > 0.9:
            insights.append({
                "type": "success",
                "icon": "fas fa-chart-area",
                "title": "Excellent Discriminative Power",
                "description": f"ROC-AUC: {roc_auc:.3f}. Model has excellent ability to distinguish between classes.",
                "recommendation": "Model demonstrates strong discriminative capabilities across all thresholds.",
                "priority": "low"
            })
        elif roc_auc > 0.8:
            insights.append({
                "type": "success",
                "icon": "fas fa-chart-area",
                "title": "Good Discriminative Power",
                "description": f"ROC-AUC: {roc_auc:.3f}. Model shows good classification performance.",
                "recommendation": "Solid performance suitable for most practical applications.",
                "priority": "low"
            })
        elif roc_auc > 0.7:
            insights.append({
                "type": "warning",
                "icon": "fas fa-chart-area",
                "title": "Moderate Discriminative Power",
                "description": f"ROC-AUC: {roc_auc:.3f}. Model performance is acceptable but could be improved.",
                "recommendation": "Consider feature engineering or more sophisticated algorithms.",
                "priority": "medium"
            })
        
        return insights
    
    def generate_network_insights(self, network_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate insights for network analysis results.
        
        Args:
            network_results: Dictionary containing network analysis results
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        if not network_results:
            return insights
        
        # Network density insights
        density = network_results.get('density', 0)
        if density > 0.7:
            insights.append({
                "type": "warning",
                "icon": "fas fa-project-diagram",
                "title": "High Network Density",
                "description": f"Network density: {density:.3f}. Very dense network may indicate coordinated behavior.",
                "recommendation": "Investigate potential bot networks or coordinated inauthentic behavior.",
                "priority": "high"
            })
        elif density > 0.3:
            insights.append({
                "type": "info",
                "icon": "fas fa-project-diagram",
                "title": "Moderate Network Density",
                "description": f"Network density: {density:.3f}. Normal interaction patterns observed.",
                "recommendation": "Monitor for emerging clusters or unusual connection patterns.",
                "priority": "medium"
            })
        else:
            insights.append({
                "type": "success",
                "icon": "fas fa-project-diagram",
                "title": "Sparse Network",
                "description": f"Network density: {density:.3f}. Healthy, organic interaction patterns.",
                "recommendation": "Network shows natural communication patterns.",
                "priority": "low"
            })
        
        # Centrality insights
        avg_centrality = network_results.get('average_centrality', 0)
        if avg_centrality > 0.8:
            insights.append({
                "type": "warning",
                "icon": "fas fa-bullseye",
                "title": "High Centralization",
                "description": f"Average centrality: {avg_centrality:.3f}. Network dominated by few highly connected nodes.",
                "recommendation": "Investigate influential accounts for potential manipulation or bot activity.",
                "priority": "high"
            })
        
        return insights
    
    def generate_feature_insights(self, feature_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate insights for feature extraction results.
        
        Args:
            feature_results: Dictionary containing feature extraction results
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        if not feature_results:
            return insights
        
        # Feature count insights
        total_features = feature_results.get('total_features', 0)
        if total_features > 1000:
            insights.append({
                "type": "warning",
                "icon": "fas fa-columns",
                "title": "High-Dimensional Feature Space",
                "description": f"Extracted {total_features:,} features. High dimensionality may lead to overfitting.",
                "recommendation": "Consider feature selection techniques like mutual information or L1 regularization.",
                "priority": "medium"
            })
        elif total_features < 50:
            insights.append({
                "type": "info",
                "icon": "fas fa-columns",
                "title": "Limited Feature Set",
                "description": f"Extracted {total_features} features. May benefit from additional feature engineering.",
                "recommendation": "Consider extracting more linguistic, behavioral, or network features.",
                "priority": "medium"
            })
        else:
            insights.append({
                "type": "success",
                "icon": "fas fa-columns",
                "title": "Optimal Feature Count",
                "description": f"Extracted {total_features} features. Good balance for model training.",
                "recommendation": "Feature count is in the optimal range for most ML algorithms.",
                "priority": "low"
            })
        
        # Feature type distribution
        feature_types = feature_results.get('feature_types', {})
        if feature_types:
            text_features = feature_types.get('text_features', 0)
            numerical_features = feature_types.get('numerical_features', 0)
            
            if text_features > numerical_features * 3:
                insights.append({
                    "type": "info",
                    "icon": "fas fa-font",
                    "title": "Text-Heavy Feature Set",
                    "description": f"Text features dominate ({text_features} vs {numerical_features} numerical).",
                    "recommendation": "Consider adding more numerical features like engagement metrics or temporal patterns.",
                    "priority": "medium"
                })
            elif numerical_features > text_features * 3:
                insights.append({
                    "type": "info",
                    "icon": "fas fa-calculator",
                    "title": "Numerical-Heavy Feature Set",
                    "description": f"Numerical features dominate ({numerical_features} vs {text_features} text).",
                    "recommendation": "Consider adding more text-based features like sentiment or linguistic complexity.",
                    "priority": "medium"
                })
            else:
                insights.append({
                    "type": "success",
                    "icon": "fas fa-balance-scale",
                    "title": "Balanced Feature Types",
                    "description": f"Good balance between text ({text_features}) and numerical ({numerical_features}) features.",
                    "recommendation": "Feature type distribution is well-balanced for comprehensive analysis.",
                    "priority": "low"
                })
        
        return insights
    
    def generate_comprehensive_insights(self, comprehensive_features: Dict[str, Any], df) -> List[Dict[str, str]]:
        """Generate insights for comprehensive feature analysis."""
        insights = []
        
        try:
            # Gratification insights
            if 'gratification_analysis' in comprehensive_features:
                grat = comprehensive_features['gratification_analysis']
                
                insights.append({
                    'type': 'gratification',
                    'title': 'Content Gratification Analysis',
                    'content': f"Entertainment seeking: {grat.get('entertainment_seeking', 0):.3f}, "
                              f"Information seeking: {grat.get('information_seeking', 0):.3f}, "
                              f"Social interaction: {grat.get('social_interaction', 0):.3f}"
                })
            
            # Theoretical framework insights
            if 'theoretical_insights' in comprehensive_features:
                theo = comprehensive_features['theoretical_insights']
                
                insights.append({
                    'type': 'theoretical',
                    'title': 'Theoretical Framework Analysis',
                    'content': f"Risk perception: {theo.get('rat_risk_perception', 0):.3f}, "
                              f"Benefit perception: {theo.get('rat_benefit_perception', 0):.3f}, "
                              f"Coping appraisal: {theo.get('rct_coping_appraisal', 0):.3f}"
                })
            
            # Feature breakdown insights
            if 'feature_breakdown' in comprehensive_features:
                breakdown = comprehensive_features['feature_breakdown']
                total_features = sum(breakdown.values()) if isinstance(breakdown, dict) else 0
                
                insights.append({
                    'type': 'features',
                    'title': 'Feature Engineering Summary',
                    'content': f"Total features extracted: {total_features}. "
                              f"Enhanced mode includes gratification and theoretical features."
                })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive insights: {e}")
            return [{
                'type': 'error',
                'title': 'Insight Generation Error',
                'content': f'Error generating insights: {str(e)}'
            }]
    
    def generate_hyperparameter_insights(self, hyperparameter_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate insights for hyperparameter optimization results.
        
        Args:
            hyperparameter_results: Dictionary containing hyperparameter optimization results
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        if not hyperparameter_results:
            return insights
        
        # Optimization improvement insights
        baseline_score = hyperparameter_results.get('baseline_score', 0)
        optimized_score = hyperparameter_results.get('best_score', 0)
        
        if optimized_score > baseline_score:
            improvement = ((optimized_score - baseline_score) / baseline_score) * 100
            if improvement > 10:
                insights.append({
                    "type": "success",
                    "icon": "fas fa-chart-line",
                    "title": "Significant Improvement",
                    "description": f"Hyperparameter tuning improved performance by {improvement:.1f}%.",
                    "recommendation": "Excellent optimization results. Consider using these parameters for production.",
                    "priority": "low"
                })
            elif improvement > 5:
                insights.append({
                    "type": "success",
                    "icon": "fas fa-thumbs-up",
                    "title": "Moderate Improvement",
                    "description": f"Hyperparameter tuning improved performance by {improvement:.1f}%.",
                    "recommendation": "Good optimization results. Parameters are ready for deployment.",
                    "priority": "low"
                })
            else:
                insights.append({
                    "type": "info",
                    "icon": "fas fa-info-circle",
                    "title": "Minimal Improvement",
                    "description": f"Hyperparameter tuning improved performance by {improvement:.1f}%.",
                    "recommendation": "Small improvement suggests default parameters were already good.",
                    "priority": "medium"
                })
        else:
            insights.append({
                "type": "warning",
                "icon": "fas fa-exclamation-triangle",
                "title": "No Improvement",
                "description": "Hyperparameter tuning did not improve performance.",
                "recommendation": "Consider different optimization strategies or feature engineering.",
                "priority": "high"
            })
        
        # Search space insights
        search_iterations = hyperparameter_results.get('search_iterations', 0)
        if search_iterations < 50:
            insights.append({
                "type": "info",
                "icon": "fas fa-search",
                "title": "Limited Search Space",
                "description": f"Only {search_iterations} parameter combinations tested.",
                "recommendation": "Consider expanding search space or using more iterations for better optimization.",
                "priority": "medium"
            })
        
        return insights
    
    def cache_insights(self, key: str, insights: List[Dict[str, str]]) -> None:
        """
        Cache insights for later retrieval.
        
        Args:
            key: Cache key
            insights: List of insights to cache
        """
        self.insights_cache[key] = {
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_cached_insights(self, key: str) -> Optional[List[Dict[str, str]]]:
        """
        Retrieve cached insights.
        
        Args:
            key: Cache key
            
        Returns:
            Cached insights or None if not found
        """
        cached = self.insights_cache.get(key)
        if cached:
            return cached['insights']
        return None
    
    def clear_cache(self) -> None:
        """Clear the insights cache."""
        self.insights_cache.clear()
    
    def generate_content_analysis_insights(self, content_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate insights for content analysis results.
        
        Args:
            content_results: Dictionary containing content analysis results
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        if not content_results:
            return insights
        
        try:
            # Sentiment distribution insights
            sentiment_dist = content_results.get('sentiment_distribution', {})
            if sentiment_dist:
                positive = sentiment_dist.get('positive', 0)
                negative = sentiment_dist.get('negative', 0)
                neutral = sentiment_dist.get('neutral', 0)
                total = positive + negative + neutral
                
                if total > 0:
                    pos_pct = (positive / total) * 100
                    neg_pct = (negative / total) * 100
                    
                    if pos_pct > 60:
                        insights.append({
                            "type": "success",
                            "icon": "fas fa-smile",
                            "title": "Positive Content Dominance",
                            "description": f"{pos_pct:.1f}% of content shows positive sentiment.",
                            "recommendation": "High positive sentiment indicates healthy community engagement.",
                            "priority": "low"
                        })
                    elif neg_pct > 40:
                        insights.append({
                            "type": "warning",
                            "icon": "fas fa-frown",
                            "title": "High Negative Sentiment",
                            "description": f"{neg_pct:.1f}% of content shows negative sentiment.",
                            "recommendation": "Consider content moderation strategies to improve community tone.",
                            "priority": "high"
                        })
                    else:
                        insights.append({
                            "type": "info",
                            "icon": "fas fa-balance-scale",
                            "title": "Balanced Sentiment Distribution",
                            "description": f"Sentiment distribution: {pos_pct:.1f}% positive, {neg_pct:.1f}% negative.",
                            "recommendation": "Sentiment balance suggests natural, authentic conversations.",
                            "priority": "low"
                        })
            
            # Language complexity insights
            avg_complexity = content_results.get('average_complexity', 0)
            if avg_complexity > 0:
                if avg_complexity > 15:
                    insights.append({
                        "type": "info",
                        "icon": "fas fa-graduation-cap",
                        "title": "High Language Complexity",
                        "description": f"Average complexity score: {avg_complexity:.1f}",
                        "recommendation": "Complex language may indicate educated audience or technical discussions.",
                        "priority": "medium"
                    })
                elif avg_complexity < 8:
                    insights.append({
                        "type": "info",
                        "icon": "fas fa-comments",
                        "title": "Simple Language Usage",
                        "description": f"Average complexity score: {avg_complexity:.1f}",
                        "recommendation": "Simple language suggests broad accessibility and casual communication.",
                        "priority": "medium"
                    })
            
            # Gratification patterns
            gratification_scores = content_results.get('gratification_analysis', {})
            if gratification_scores:
                entertainment = gratification_scores.get('entertainment_seeking', 0)
                information = gratification_scores.get('information_seeking', 0)
                social = gratification_scores.get('social_interaction', 0)
                
                max_grat = max(entertainment, information, social)
                if max_grat == entertainment and entertainment > 0.6:
                    insights.append({
                        "type": "info",
                        "icon": "fas fa-play-circle",
                        "title": "Entertainment-Focused Content",
                        "description": f"High entertainment seeking score: {entertainment:.2f}",
                        "recommendation": "Content serves primarily entertainment purposes. Consider engagement strategies.",
                        "priority": "medium"
                    })
                elif max_grat == information and information > 0.6:
                    insights.append({
                        "type": "success",
                        "icon": "fas fa-info-circle",
                        "title": "Information-Rich Content",
                        "description": f"High information seeking score: {information:.2f}",
                        "recommendation": "Content provides valuable information. Maintain quality standards.",
                        "priority": "low"
                    })
                elif max_grat == social and social > 0.6:
                    insights.append({
                        "type": "info",
                        "icon": "fas fa-users",
                        "title": "Social Interaction Focus",
                        "description": f"High social interaction score: {social:.2f}",
                        "recommendation": "Content facilitates social connections. Encourage community building.",
                        "priority": "medium"
                    })
            
        except Exception as e:
            self.logger.error(f"Error generating content analysis insights: {e}")
            insights.append({
                "type": "danger",
                "icon": "fas fa-exclamation-triangle",
                "title": "Insight Generation Error",
                "description": f"Error analyzing content results: {str(e)}",
                "recommendation": "Check content analysis data format and try again.",
                "priority": "high"
            })
        
        return insights
    
    def generate_behavioral_profiling_insights(self, behavioral_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate insights for behavioral profiling results.
        
        Args:
            behavioral_results: Dictionary containing behavioral profiling results
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        if not behavioral_results:
            return insights
        
        try:
            # User engagement patterns
            engagement_metrics = behavioral_results.get('engagement_metrics', {})
            if engagement_metrics:
                avg_posts = engagement_metrics.get('average_posts_per_user', 0)
                avg_interactions = engagement_metrics.get('average_interactions_per_user', 0)
                
                if avg_posts > 10:
                    insights.append({
                        "type": "success",
                        "icon": "fas fa-chart-line",
                        "title": "High User Activity",
                        "description": f"Users average {avg_posts:.1f} posts each.",
                        "recommendation": "Strong user engagement. Focus on retention strategies.",
                        "priority": "low"
                    })
                elif avg_posts < 3:
                    insights.append({
                        "type": "warning",
                        "icon": "fas fa-chart-line",
                        "title": "Low User Activity",
                        "description": f"Users average only {avg_posts:.1f} posts each.",
                        "recommendation": "Consider engagement campaigns to increase user participation.",
                        "priority": "high"
                    })
                
                if avg_interactions > avg_posts * 2:
                    insights.append({
                        "type": "success",
                        "icon": "fas fa-heart",
                        "title": "High Interaction Rate",
                        "description": f"Users interact {avg_interactions/avg_posts:.1f}x more than they post.",
                        "recommendation": "Strong community engagement. Users are actively participating.",
                        "priority": "low"
                    })
            
            # Behavioral clusters
            clusters = behavioral_results.get('user_clusters', {})
            if clusters:
                cluster_count = len(clusters)
                if cluster_count > 5:
                    insights.append({
                        "type": "info",
                        "icon": "fas fa-users",
                        "title": "Diverse User Segments",
                        "description": f"Identified {cluster_count} distinct behavioral clusters.",
                        "recommendation": "Consider targeted strategies for different user segments.",
                        "priority": "medium"
                    })
                elif cluster_count < 3:
                    insights.append({
                        "type": "warning",
                        "icon": "fas fa-users",
                        "title": "Limited User Diversity",
                        "description": f"Only {cluster_count} behavioral clusters identified.",
                        "recommendation": "User base may be homogeneous. Consider diversification strategies.",
                        "priority": "medium"
                    })
            
            # Temporal patterns
            temporal_patterns = behavioral_results.get('temporal_patterns', {})
            if temporal_patterns:
                peak_hours = temporal_patterns.get('peak_activity_hours', [])
                if peak_hours:
                    insights.append({
                        "type": "info",
                        "icon": "fas fa-clock",
                        "title": "Activity Peak Times",
                        "description": f"Peak activity during hours: {', '.join(map(str, peak_hours))}",
                        "recommendation": "Schedule important content during peak activity times.",
                        "priority": "medium"
                    })
            
            # Risk behavior indicators
            risk_indicators = behavioral_results.get('risk_indicators', {})
            if risk_indicators:
                high_risk_users = risk_indicators.get('high_risk_user_count', 0)
                total_users = behavioral_results.get('total_users', 1)
                risk_percentage = (high_risk_users / total_users) * 100
                
                if risk_percentage > 15:
                    insights.append({
                        "type": "danger",
                        "icon": "fas fa-exclamation-triangle",
                        "title": "High Risk User Percentage",
                        "description": f"{risk_percentage:.1f}% of users show high-risk behavioral patterns.",
                        "recommendation": "Implement targeted intervention strategies for high-risk users.",
                        "priority": "critical"
                    })
                elif risk_percentage > 5:
                    insights.append({
                        "type": "warning",
                        "icon": "fas fa-shield-alt",
                        "title": "Moderate Risk Indicators",
                        "description": f"{risk_percentage:.1f}% of users show moderate risk patterns.",
                        "recommendation": "Monitor risk indicators and consider preventive measures.",
                        "priority": "high"
                    })
                else:
                    insights.append({
                        "type": "success",
                        "icon": "fas fa-shield-alt",
                        "title": "Low Risk Environment",
                        "description": f"Only {risk_percentage:.1f}% of users show risk patterns.",
                        "recommendation": "Maintain current safety measures and monitoring.",
                        "priority": "low"
                    })
            
        except Exception as e:
            self.logger.error(f"Error generating behavioral profiling insights: {e}")
            insights.append({
                "type": "danger",
                "icon": "fas fa-exclamation-triangle",
                "title": "Insight Generation Error",
                "description": f"Error analyzing behavioral results: {str(e)}",
                "recommendation": "Check behavioral profiling data format and try again.",
                "priority": "high"
            })
        
        return insights
    
    def generate_zero_shot_insights(self, zero_shot_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate insights from zero-shot classification results."""
        insights = []
        
        try:
            if not zero_shot_results:
                insights.append({
                    "type": "warning",
                    "title": "No Zero-Shot Results",
                    "description": "No zero-shot classification results available for analysis.",
                    "recommendation": "Run zero-shot classification first to generate insights.",
                    "priority": "medium"
                })
                return insights
            
            # Classification distribution insights
            if 'classification_distribution' in zero_shot_results:
                dist = zero_shot_results['classification_distribution']
                
                # Find dominant classification
                if dist:
                    dominant_class = max(dist.items(), key=lambda x: x[1])
                    insights.append({
                        "type": "info",
                        "title": "Dominant Classification",
                        "description": f"Most content classified as '{dominant_class[0]}' ({dominant_class[1]:.1f}%)",
                        "recommendation": f"Focus analysis on {dominant_class[0]} content patterns.",
                        "priority": "high"
                    })
                
                # Check for balanced distribution
                if len(dist) > 1:
                    values = list(dist.values())
                    max_val, min_val = max(values), min(values)
                    if max_val - min_val < 20:  # Less than 20% difference
                        insights.append({
                            "type": "warning",
                            "title": "Balanced Classification Distribution",
                            "description": "Content is relatively evenly distributed across classifications.",
                            "recommendation": "Consider refining classification criteria or examining edge cases.",
                            "priority": "medium"
                        })
            
            # Confidence analysis
            if 'average_confidence' in zero_shot_results:
                avg_conf = zero_shot_results['average_confidence']
                
                if avg_conf > 0.8:
                    insights.append({
                        "type": "success",
                        "title": "High Classification Confidence",
                        "description": f"Average confidence score is {avg_conf:.2f}, indicating reliable classifications.",
                        "recommendation": "Results are trustworthy for decision making.",
                        "priority": "low"
                    })
                elif avg_conf < 0.6:
                    insights.append({
                        "type": "warning",
                        "title": "Low Classification Confidence",
                        "description": f"Average confidence score is {avg_conf:.2f}, indicating uncertain classifications.",
                        "recommendation": "Consider manual review of low-confidence samples or model fine-tuning.",
                        "priority": "high"
                    })
            
            # Confidence distribution insights
            if 'confidence_distribution' in zero_shot_results:
                conf_dist = zero_shot_results['confidence_distribution']
                
                high_conf = conf_dist.get('high', 0)
                low_conf = conf_dist.get('low', 0)
                
                if low_conf > 30:  # More than 30% low confidence
                    insights.append({
                        "type": "warning",
                        "title": "High Proportion of Low-Confidence Classifications",
                        "description": f"{low_conf:.1f}% of classifications have low confidence.",
                        "recommendation": "Review classification criteria and consider additional training data.",
                        "priority": "medium"
                    })
                
                if high_conf > 70:  # More than 70% high confidence
                    insights.append({
                        "type": "success",
                        "title": "Strong Classification Performance",
                        "description": f"{high_conf:.1f}% of classifications have high confidence.",
                        "recommendation": "Model performs well on this dataset.",
                        "priority": "low"
                    })
            
            # Sample analysis
            if 'total_samples_classified' in zero_shot_results:
                total = zero_shot_results['total_samples_classified']
                
                if total < 100:
                    insights.append({
                        "type": "info",
                        "title": "Small Sample Size",
                        "description": f"Only {total} samples classified.",
                        "recommendation": "Consider larger sample size for more robust insights.",
                        "priority": "medium"
                    })
                elif total > 10000:
                    insights.append({
                        "type": "success",
                        "title": "Large-Scale Analysis",
                        "description": f"{total:,} samples successfully classified.",
                        "recommendation": "Results are statistically significant.",
                        "priority": "low"
                    })
            
            # Model performance insights
            if 'model_used' in zero_shot_results:
                model = zero_shot_results['model_used']
                insights.append({
                    "type": "info",
                    "title": "Classification Model",
                    "description": f"Used {model} for zero-shot classification.",
                    "recommendation": "Consider comparing with other models for validation.",
                    "priority": "low"
                })
            
        except Exception as e:
            insights.append({
                "type": "error",
                "title": "Error Generating Zero-Shot Insights",
                "description": f"Failed to analyze zero-shot results: {str(e)}",
                "recommendation": "Check zero-shot results data format and try again.",
                "priority": "high"
            })
        
        return insights
    
    def generate_theoretical_insights(self, df, theoretical_features_df=None) -> Dict[str, float]:
        """Generate theoretical framework insights (RAT/RCT metrics)."""
        try:
            insights = {
                'rat_risk_perception': 0.0,
                'rat_benefit_perception': 0.0,
                'rct_coping_appraisal': 0.0,
                'rct_threat_appraisal': 0.0
            }
            
            if theoretical_features_df is not None and not theoretical_features_df.empty:
                # Extract RAT insights
                if 'rat_perceived_risk' in theoretical_features_df.columns:
                    insights['rat_risk_perception'] = float(theoretical_features_df['rat_perceived_risk'].mean())
                elif 'rat_overall_risk_score' in theoretical_features_df.columns:
                    insights['rat_risk_perception'] = float(theoretical_features_df['rat_overall_risk_score'].mean())
                
                if 'rat_perceived_benefit' in theoretical_features_df.columns:
                    insights['rat_benefit_perception'] = float(theoretical_features_df['rat_perceived_benefit'].mean())
                elif 'rat_suitable_target_score' in theoretical_features_df.columns:
                    insights['rat_benefit_perception'] = float(theoretical_features_df['rat_suitable_target_score'].mean())
                
                # Extract RCT insights
                if 'rct_coping_appraisal' in theoretical_features_df.columns:
                    insights['rct_coping_appraisal'] = float(theoretical_features_df['rct_coping_appraisal'].mean())
                elif 'rct_decision_rationality_score' in theoretical_features_df.columns:
                    insights['rct_coping_appraisal'] = float(theoretical_features_df['rct_decision_rationality_score'].mean())
                
                if 'rct_threat_appraisal' in theoretical_features_df.columns:
                    insights['rct_threat_appraisal'] = float(theoretical_features_df['rct_threat_appraisal'].mean())
                elif 'rct_overall_score' in theoretical_features_df.columns:
                    insights['rct_threat_appraisal'] = float(theoretical_features_df['rct_overall_score'].mean())
            
            # Ensure values are in valid range [0, 1]
            for key in insights:
                insights[key] = max(0.0, min(1.0, insights[key]))
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating theoretical insights: {e}")
            return {
                'rat_risk_perception': 0.0,
                'rat_benefit_perception': 0.0,
                'rct_coping_appraisal': 0.0,
                'rct_threat_appraisal': 0.0
            }