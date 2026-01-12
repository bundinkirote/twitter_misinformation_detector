# Twitter Misinformation Detection System

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-Academic-orange.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)](README.md)

##  Overview

A comprehensive machine learning system for detecting misinformation in Twitter data, incorporating both textual and behavioral features based on three theoretical frameworks: **Routine Activity Theory (RAT)**, **Rational Choice Theory (RCT)**, and **Uses and Gratifications Theory (UGT)**. The system provides a web interface for dataset upload, processing, feature extraction, model training, and evaluation.

### Key Features

-  **Multi-Modal Analysis**: Text, behavioral, and network features
- **Theoretical Framework Integration**: RAT, RCT, and UGT implementations
-  **Web Interface**: User-friendly Flask-based dashboard
-  **Advanced Visualizations**: Interactive charts and network diagrams
-  **AI-Powered Insights**: Intelligent recommendations at each step
-  **Explainable AI**: SHAP-based model interpretability
-  **Multilingual Support**: Language detection and analysis
-  **Performance Optimization**: Automated hyperparameter tuning

##  Quick Start

### Prerequisites

- Python 3.11+ (compatible with 3.13.2)
- 8GB RAM minimum (16GB+ recommended)
- 5GB free disk space
- Internet connection (for downloading transformer models)

### Installation

1. **Clone or extract the project**
   ```bash
   cd project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # For macOS users:
   # pip install -r requirements_macos.txt
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

6. **Access the web interface**
   Open your browser and navigate to: `http://localhost:5000`

##  Project Structure

```
twitter_misinformation_detection/
  main.py                     # Main Flask application
  config.json                 # Application configuration
  requirements.txt            # Python dependencies
  requirements_macos.txt      # macOS-specific dependencies
  setup_macos.sh             # macOS setup script
 
  src/                        # Core modules
     data_processor.py       # Data preprocessing and cleaning
     data_collector.py       # Data collection and integration
     feature_extractor.py    # Feature engineering
     model_trainer.py        # ML model training
     model_evaluator.py      # Model evaluation and metrics
     model_manager.py        # Model loading and management
     model_compatibility.py  # Model version compatibility
     hyperparameter_optimizer.py # Automated tuning
     network_analyzer.py     # Social network analysis
     language_detector.py    # Language detection
     sentiment_analyzer.py   # Sentiment analysis
     zero_shot_labeling.py   # Zero-shot classification
     theoretical_frameworks.py # RAT/RCT/UGT implementations
     shap_explainer.py       # Model explainability
     insights_generator.py   # AI insights system
     fact_check_validator.py # Fact-checking integration
     visualization_generator.py # Chart generation
     interaction_logger.py   # User interaction logging
     auto_scraper_trigger.py # Automated data scraping (future)
     local_model_manager.py  # Local transformer model caching
     smart_model_manager.py  # Smart model downloading
     ensemble_builder.py     # Ensemble model building
     prediction_service.py   # Prediction service API
     utils/                  # Utility functions
         file_manager.py     # File operations
 
  templates/                  # HTML templates
     base.html              # Base template
     index.html             # Main dashboard
     upload_dataset.html    # Dataset upload
     dataset_overview.html  # Dataset statistics
     feature_extraction.html # Feature engineering
     training_pipeline.html # Model training
     training_results.html  # Training results
     network_analysis.html  # Network visualization
     language_detection.html # Language analysis
     sentiment_analysis.html # Sentiment analysis
     zero_shot_labeling.html # Zero-shot classification
     explainability.html    # Model explanations
     predict.html           # Live predictions
 
  static/                     # Static assets
     css/                   # Stylesheets
     js/                    # JavaScript files
     images/                # Images and icons
     plots/                 # Generated plots
     visualizations/        # Visualization outputs
     network_visualizations/ # Network diagrams
 
  datasets/                   # Dataset storage
  models/                     # Trained model storage
  local_models/              # Downloaded transformer models
  logs/                      # Application logs
  tests/                     # Test scripts and documentation
  reports/                   # Generated reports
  visualizations/            # Output visualizations
```

##  Usage Guide

### 1. Dataset Upload and Processing

1. **Navigate to Upload Page**
   - Click "Upload Dataset" from the main dashboard
   - Supported formats: CSV, Excel (.xlsx)

2. **Required Data Format**
   ```csv
   text,LABEL,user_id,retweet_count,favorite_count
   "Sample tweet text",0,user123,5,10
   "Another tweet",1,user456,2,3
   ```
   - `text`: Tweet content (required)
   - `LABEL`: 0 = legitimate, 1 = misinformation (optional for zero-shot)
   - Additional columns for behavioral features (optional)

3. **Automatic Processing**
   - Data cleaning and preprocessing
   - Column mapping and validation
   - Statistical analysis and insights

### 2. Feature Extraction

The system extracts multiple types of features:

#### Text Features
- **TF-IDF Vectors**: Term frequency analysis
- **Sentiment Scores**: Emotional content analysis
- **Linguistic Features**: Readability, complexity metrics
- **Transformer Embeddings**: BERT-based semantic representations

#### Behavioral Features
- **Engagement Metrics**: Likes, retweets, replies
- **User Activity**: Posting patterns, account age
- **Network Position**: Centrality measures
- **Temporal Patterns**: Time-based activity analysis

#### Theoretical Framework Features
- **RAT Features**: Motivated offenders, suitable targets, guardianship
- **RCT Features**: Cost-benefit analysis, decision-making patterns

### 3. Model Training and Evaluation

#### Supported Algorithms
- **Logistic Regression**: Linear classification
- **Naive Bayes**: Probabilistic classification
- **Support Vector Machine (SVM)**: Kernel-based classification
- **Random Forest**: Ensemble method
- **Gradient Boosting**: Advanced ensemble
- **Neural Networks**: Deep learning (MLPClassifier)

#### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Balanced performance measure
- **ROC-AUC**: Discrimination ability
- **Confusion Matrix**: Detailed error analysis

### 4. Advanced Features

#### Zero-Shot Classification
- Classify unlabeled data using pre-trained models
- Custom label definitions
- Confidence scoring

#### Network Analysis
- User interaction networks
- Community detection
- Influence propagation analysis
- Centrality measures

#### Explainable AI
- SHAP value analysis
- Feature importance ranking
- Decision boundary visualization
- Local explanations for individual predictions

##  AI Insights System

The system provides intelligent recommendations at each step:

### Dataset Analysis
- **Quality Assessment**: Data completeness, balance, outliers
- **Feature Recommendations**: Optimal feature combinations
- **Processing Suggestions**: Data cleaning strategies

### Model Performance
- **Algorithm Selection**: Best models for your data
- **Hyperparameter Guidance**: Optimization strategies
- **Performance Interpretation**: Strengths and weaknesses

### Network Insights
- **Community Structure**: User group identification
- **Influence Patterns**: Key spreaders and receivers
- **Propagation Analysis**: Information flow patterns

##  Performance Benchmarks

### Typical Performance Metrics
- **Accuracy**: 85-95%
- **F1-Score**: 0.82-0.95
- **ROC-AUC**: 0.87-0.96
- **Processing Speed**: 1000+ tweets/minute

### Optimization Features
- **Hyperparameter Tuning**: Grid search and random search
- **Feature Selection**: Automated feature importance
- **Model Ensemble**: Combining multiple algorithms
- **Cross-Validation**: Robust performance estimation

##  Research Applications

### Academic Research
- Misinformation detection studies
- Social media behavior analysis
- Network theory applications
- Computational social science

### Practical Applications
- Content moderation systems
- Fact-checking automation
- Social media monitoring
- Crisis communication analysis

### Theoretical Contributions
- **Routine Activity Theory**: Digital crime prevention through identification of motivated offenders, suitable targets, and capable guardians
- **Rational Choice Theory**: Decision-making analysis of cost-benefit tradeoffs in misinformation propagation
- **Uses and Gratifications Theory**: Understanding user motivations (information seeking, entertainment, social interaction, identity affirmation, surveillance, escapism) for consuming and sharing misinformation
- **Multi-theoretical Integration**: Combined framework approach for comprehensive misinformation detection

##  Technical Specifications

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, Linux
- **Python**: 3.11+ (tested with 3.13.2)
- **Memory**: 8GB RAM minimum, 16GB+ recommended
- **Storage**: 5GB free space minimum
- **Network**: Internet connection required for model downloads

### Dependencies
- **Web Framework**: Flask 2.3.3, Waitress 2.1.2
- **Data Processing**: pandas 2.1.1, numpy 1.24.3
- **Machine Learning**: scikit-learn 1.3.0, transformers 4.33.2
- **Visualization**: matplotlib 3.7.2, seaborn 0.12.2, plotly 5.16.1
- **Network Analysis**: networkx 3.1
- **NLP**: nltk 3.8.1, sentence-transformers 2.2.2

### Dependencies
- **Web Framework**: Flask 2.3.3, Waitress 2.1.2
- **Data Processing**: pandas 2.1.1, numpy 1.24.3
- **Machine Learning**: scikit-learn 1.3.0, imbalanced-learn 0.13.0, transformers 4.33.2
- **Visualization**: matplotlib 3.7.2, seaborn 0.12.2, plotly 5.16.1
- **Network Analysis**: networkx 3.1, community 1.0.0b1
- **NLP**: nltk 3.8.1, sentence-transformers 2.2.2, langdetect 1.0.9
- **Explainability**: shap 0.42.1
- **Model Persistence**: joblib 1.3.2

### Configuration Files

#### config.json
```json
{
  "app_name": "Twitter Misinformation Detection System",
  "version": "1.0.0",
  "max_file_size": "16MB",
  "supported_formats": [".csv", ".xlsx"],
  "feature_extraction": {
    "use_tfidf": true,
    "use_sentiment": true,
    "use_behavioral": true,
    "use_theoretical_frameworks": true
  },
  "model_training": {
    "test_size": 0.2,
    "cross_validation_folds": 5,
    "random_state": 42
  }
}
```

#### Environment Variables
Create a `.env` file for sensitive configuration:
```
FLASK_ENV=development
FLASK_DEBUG=0
SECRET_KEY=your-secret-key-here
TRANSFORMER_CACHE=./local_models
LOG_LEVEL=INFO
```

##  Monitoring and Logging

### Application Logs
- **Location**: `logs/app.log`
- **Levels**: INFO, WARNING, ERROR
- **Format**: Timestamp, level, message
- **Note**: Uses StreamHandler and FileHandler (not rotating due to single-user design)

### Performance Monitoring (CPU-only)
- Processing time tracking per operation
- Memory usage monitoring
- Model performance metrics (accuracy, F1, ROC-AUC)
- Dataset statistics and insights
- Processing speed: Approximately 100-500 samples/second depending on feature complexity

##  Security and Privacy

### Data Protection
- Local data processing (no external data sharing)
- Secure file handling
- Input validation and sanitization
- Session management

### Model Security
- Local model storage
- Secure model loading
- Input validation for predictions

##  Deployment

### Development Mode
```bash
python main.py
```

### Production Deployment
For production deployment, consider:
- Using a production WSGI server (Gunicorn, uWSGI)
- Setting up reverse proxy (Nginx, Apache)
- Configuring SSL/TLS certificates
- Setting up monitoring and logging

### Docker Deployment (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "main.py"]
```

##  Support and Troubleshooting

### Common Issues

1. **Model Download Failures**
   - Check internet connection
   - Verify disk space availability
   - Check firewall settings

2. **Memory Issues**
   - Reduce dataset size for testing
   - Increase system RAM
   - Use feature selection to reduce dimensionality

3. **Performance Issues**
   - Enable GPU acceleration (if available)
   - Use smaller transformer models
   - Implement batch processing

### Getting Help
1. Check the AI insights for recommendations
2. Review application logs in `logs/app.log`
3. Verify all dependencies are correctly installed
4. Check the test results for system validation

##  Academic Citation

If you use this system in academic research, please cite:

```bibtex
@software{twitter_misinformation_detection,
  title={Twitter Misinformation Detection System: A Multi-Theoretical Framework Approach},
  author={[Doreen Nkirote]},
  year={2025},
  url={https://github.com/bundinkirote/twitter_misinformation_detector},
  note={Machine Learning System for Social Media Analysis}
}
```

##  License

This project is licensed for academic and research use. Please ensure appropriate citation in publications and research papers.

##  Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

##  Changelog

### Version 1.0.0 (Current)
- Initial release with full functionality
- Multi-theoretical framework integration
- Web interface with AI insights
- Comprehensive testing suite
- Production-ready deployment

##  Documentation

### Complete Documentation Suite
This project includes comprehensive documentation covering all aspects of the system:

- **[ Documentation Index](docs/README.md)** - Complete documentation overview and navigation
- **[ Installation Guide](docs/INSTALLATION_GUIDE.md)** - Detailed setup instructions for all platforms
- **[ User Guide](docs/USER_GUIDE.md)** - Complete user manual with step-by-step instructions
- **[ API Documentation](docs/API_DOCUMENTATION.md)** - Full API reference with examples
- **[ Architecture Guide](docs/ARCHITECTURE.md)** - System architecture and design patterns
- **[ Theoretical Framework](docs/THEORETICAL_FRAMEWORK.md)** - RAT and RCT academic foundations
- **[ Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions

### Quick Links
- **New Users**: Start with [Installation Guide](docs/INSTALLATION_GUIDE.md) → [User Guide](docs/USER_GUIDE.md)
- **Developers**: Check [Architecture Guide](docs/ARCHITECTURE.md) → [API Documentation](docs/API_DOCUMENTATION.md)
- **Researchers**: Read [Theoretical Framework](docs/THEORETICAL_FRAMEWORK.md) → [User Guide](docs/USER_GUIDE.md)
- **Issues**: See [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for solutions

##  Advanced Components

### Fact-Check Validator (src/fact_check_validator.py)
**Purpose**: Validates claims against external fact-checking sources and known corpus

**Key Features**:
- Local fact-check corpus management with TF-IDF similarity matching
- External source integration (PesaCheck, AfricaCheck, Wikipedia)
- Confidence scoring and verdict determination
- Multi-source validation and consensus building

**Integration Points**:
- Loads trained corpus from `results/` directory
- Integrates with file_manager for data persistence
- Used in prediction service for claim validation
- Generates fact-check validation reports

**Configuration**:
- Corpus similarity threshold: 0.3 (adjustable)
- External source timeout: 10 seconds
- Supports verdicts: TRUE, FALSE, PARTIALLY, UNKNOWN
- Fact-checking sources: PesaCheck (Kenya), AfricaCheck (Africa), Wikipedia (global)

### Auto Scraper Trigger (src/auto_scraper_trigger.py)
**Purpose**: Automatically triggers scraping of fact-checking sources based on content analysis

**Key Features**:
- Content relevance analysis for misinformation detection
- Author credibility assessment
- Engagement pattern analysis for viral content detection
- Automated scraping pipeline triggering

**Intelligent Triggers**:
- **Politics Keywords**: "ruto", "raila", "parliament", "government", "election"
- **Economy Keywords**: "finance bill", "tax", "budget", "inflation"
- **Health Keywords**: "covid", "vaccine", "disease"
- **Corruption Keywords**: "scandal", "fraud", "embezzlement"

**Integration Points**:
- Works with DataCollector for corpus building
- Analyzes tweet metadata and engagement metrics
- Triggers fact-check validation pipeline
- Provides recommendations for source selection

### Data Collector (src/data_collector.py)
**Purpose**: Collects fact-checked data from multiple sources to build verification corpus

**Supported Sources**:
1. **PesaCheck** (https://pesacheck.org) - Kenyan fact-checking
2. **AfricaCheck** (https://africacheck.org) - African fact-checking network
3. **Wikipedia** (https://en.wikipedia.org) - General knowledge base
4. **Nation** (https://nation.co.ke) - Disabled by default due to rate limiting

**Data Collection Process**:
- Web scraping with BeautifulSoup for HTML parsing
- Article extraction: title, claim, verdict, content, URL, date
- Verdict normalization: TRUE, FALSE, PARTIALLY, UNKNOWN
- Rate limiting: 2-second delays between requests to avoid blocking

**Corpus Building Features**:
- Topic-based article collection
- Duplicate detection and removal
- Data quality validation
- Corpus deduplication and updates
- Automated data persistence to JSON/CSV

**Configuration**:
```python
# Maximum articles per source
max_articles_per_source = 50

# Request delay for rate limiting
request_delay = 2  # seconds

# Supported verdicts
verdicts = {
    'true': 'TRUE',
    'false': 'FALSE', 
    'mixed': 'PARTIALLY',
    'unknown': 'UNKNOWN'
}
```

**Usage in Pipeline**:
- Called automatically by AutoScraperTrigger when high-risk content detected
- Builds training corpus for fact-check validator
- Updates knowledge base with new fact-checks
- Supports researcher manual topic-based corpus building

### Model Management Components

#### Smart Model Manager (src/smart_model_manager.py)
**Purpose**: Intelligent transformer model management with automatic downloading and local caching

**Features**:
- Automatic model downloading from Hugging Face Hub
- Local model caching with version control
- Smart path resolution (local → cache → remote)
- Efficient storage management
- Support for sentence-transformers, transformers, and zero-shot models

**Supported Model Types**:
- General transformers (BERT, DistilBERT, RoBERTa)
- Sentence transformers (for embeddings)
- Zero-shot classifiers (for classification without fine-tuning)

**Key Methods**:
- `get_model_path()`: Get path to model (downloads if needed)
- `load_sentence_transformer()`: Load sentence embedding models
- `load_transformers_model()`: Load transformers with tokenizer
- `get_local_models_info()`: List locally cached models

#### Local Model Manager (src/local_model_manager.py)
**Purpose**: Manages transformer models stored locally within project structure

**Features**:
- Model registry tracking
- Version control and metadata management
- Model portability (create portable packages)
- Cache optimization
- Model removal and cleanup

**Key Functions**:
- Download and cache transformer models
- Create portable model packages
- Track model versions and compatibility
- Export/import model packages
- Manage local storage efficiently

#### Model Compatibility Manager (src/model_compatibility.py)
**Purpose**: Ensures compatibility across different sklearn and library versions

**Features**:
- Safe model loading with multiple fallback strategies
- Version mismatch detection
- Feature dimension adaptation (padding/truncation)
- Pickle vs. joblib format handling
- Model metadata validation

**Fallback Strategies**:
1. Primary: joblib loading with pickle protocol detection
2. Secondary: Pickle-only loading
3. Tertiary: Feature dimension adaptation
4. Final: Manual format detection and parsing

**Integration Points**:
- Used by ModelTrainer when saving models
- Used by ModelEvaluator when loading for evaluation
- Used by PredictionService for production predictions

---

**Ready to detect misinformation? Start with `python main.py`!** 

For complete documentation, visit the **[docs/](docs/)** directory.