# Troubleshooting Guide

## Overview

This guide provides solutions to common issues you might encounter while using the Twitter Misinformation Detection System. Issues are organized by category with step-by-step solutions and prevention tips.

##  Quick Diagnostic Checklist

Before diving into specific issues, run through this quick checklist:

- [ ] Python 3.11+ is installed and accessible
- [ ] Virtual environment is activated
- [ ] All dependencies are installed (`pip list` shows required packages)
- [ ] Internet connection is available (for model downloads)
- [ ] Sufficient disk space (5GB+ free)
- [ ] Sufficient RAM (8GB+ available)
- [ ] Application logs are accessible (`logs/app.log`)

##  Installation Issues

### Issue: Python Version Compatibility

**Symptoms:**
- Error messages about unsupported Python version
- Import errors for modern Python features
- Package installation failures

**Solutions:**

1. **Check Python Version:**
   ```bash
   python --version
   # Should show 3.11.0 or higher
   ```

2. **Install Correct Python Version:**
   ```bash
   # Windows - Download from python.org
   # macOS
   brew install python@3.11
   # Ubuntu/Debian
   sudo apt install python3.11 python3.11-venv
   ```

3. **Use Specific Python Version:**
   ```bash
   python3.11 -m venv .venv
   # Instead of just 'python -m venv .venv'
   ```

**Prevention:**
- Always verify Python version before installation
- Use version-specific commands when multiple Python versions are installed

### Issue: Virtual Environment Problems

**Symptoms:**
- "Command not found" errors
- Packages installing globally instead of in virtual environment
- Permission errors during package installation

**Solutions:**

1. **Recreate Virtual Environment:**
   ```bash
   # Remove existing environment
   rm -rf .venv  # Linux/macOS
   rmdir /s .venv  # Windows

   # Create new environment
   python -m venv .venv
   
   # Activate environment
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

2. **Verify Environment Activation:**
   ```bash
   # Check if environment is active
   which python  # Should point to .venv/bin/python
   echo $VIRTUAL_ENV  # Should show path to .venv
   ```

3. **Fix Activation Script Issues:**
   ```bash
   # Windows PowerShell execution policy
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

**Prevention:**
- Always activate virtual environment before installing packages
- Use `pip list` to verify packages are installed in the correct environment

### Issue: Package Installation Failures

**Symptoms:**
- "No module named" errors
- Package compilation failures
- Network timeout errors during installation

**Solutions:**

1. **Upgrade pip and setuptools:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Install with verbose output:**
   ```bash
   pip install -v package_name
   # Shows detailed installation process
   ```

3. **Use alternative installation methods:**
   ```bash
   # Install from requirements file with no cache
   pip install --no-cache-dir -r requirements.txt
   
   # Install with user flag if permission issues
   pip install --user -r requirements.txt
   ```

4. **Handle specific package issues:**
   ```bash
   # For packages requiring compilation
   pip install --no-binary=package_name package_name
   
   # For network issues
   pip install --timeout 1000 -r requirements.txt
   ```

**Prevention:**
- Keep pip updated
- Use virtual environments to avoid conflicts
- Check system requirements before installation

##  Dataset Issues

### Issue: File Upload Failures

**Symptoms:**
- "File not supported" errors
- Upload process hangs or times out
- Corrupted file errors

**Solutions:**

1. **Verify File Format:**
   ```python
   import pandas as pd
   
   # Test file reading
   try:
       df = pd.read_csv('your_file.csv')
       print("CSV file is valid")
   except Exception as e:
       print(f"CSV error: {e}")
   
   try:
       df = pd.read_excel('your_file.xlsx')
       print("Excel file is valid")
   except Exception as e:
       print(f"Excel error: {e}")
   ```

2. **Check File Size:**
   ```bash
   # Check file size (should be < 16MB)
   ls -lh your_file.csv  # Linux/macOS
   dir your_file.csv     # Windows
   ```

3. **Fix File Encoding Issues:**
   ```python
   # Try different encodings
   encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
   
   for encoding in encodings:
       try:
           df = pd.read_csv('your_file.csv', encoding=encoding)
           print(f"Success with encoding: {encoding}")
           break
       except Exception as e:
           print(f"Failed with {encoding}: {e}")
   ```

4. **Handle Special Characters:**
   ```python
   # Clean file before upload
   df = pd.read_csv('your_file.csv', encoding='utf-8')
   df = df.applymap(lambda x: x.encode('utf-8', 'ignore').decode('utf-8') if isinstance(x, str) else x)
   df.to_csv('cleaned_file.csv', index=False, encoding='utf-8')
   ```

**Prevention:**
- Use UTF-8 encoding for all files
- Keep file sizes under 16MB
- Validate files before upload

### Issue: Column Mapping Problems

**Symptoms:**
- "Required column not found" errors
- Incorrect feature extraction
- Missing data warnings

**Solutions:**

1. **Check Required Columns:**
   ```python
   import pandas as pd
   
   df = pd.read_csv('your_file.csv')
   print("Available columns:", df.columns.tolist())
   
   # Required columns
   required = ['text', 'LABEL']  # or your specific column names
   missing = [col for col in required if col not in df.columns]
   print("Missing columns:", missing)
   ```

2. **Rename Columns:**
   ```python
   # Map your columns to expected names
   column_mapping = {
       'tweet_text': 'text',
       'label': 'LABEL',
       'user': 'user_id'
   }
   
   df = df.rename(columns=column_mapping)
   df.to_csv('mapped_file.csv', index=False)
   ```

3. **Handle Case Sensitivity:**
   ```python
   # Convert column names to uppercase
   df.columns = df.columns.str.upper()
   
   # Or create case-insensitive mapping
   df_lower = df.copy()
   df_lower.columns = df_lower.columns.str.lower()
   ```

**Prevention:**
- Use standard column names (text, LABEL, user_id, etc.)
- Check column names before upload
- Maintain consistent naming conventions

### Issue: Data Quality Problems

**Symptoms:**
- High missing value warnings
- Poor model performance
- Skewed class distributions

**Solutions:**

1. **Analyze Data Quality:**
   ```python
   import pandas as pd
   
   df = pd.read_csv('your_file.csv')
   
   # Check missing values
   print("Missing values per column:")
   print(df.isnull().sum())
   
   # Check class distribution
   if 'LABEL' in df.columns:
       print("Class distribution:")
       print(df['LABEL'].value_counts())
   
   # Check text quality
   if 'text' in df.columns:
       print("Text statistics:")
       print(f"Empty texts: {df['text'].isnull().sum()}")
       print(f"Average text length: {df['text'].str.len().mean():.2f}")
   ```

2. **Clean Data:**
   ```python
   # Remove rows with missing text
   df = df.dropna(subset=['text'])
   
   # Remove empty or very short texts
   df = df[df['text'].str.len() > 10]
   
   # Handle missing labels for supervised learning
   if 'LABEL' in df.columns:
       df = df.dropna(subset=['LABEL'])
   
   # Save cleaned data
   df.to_csv('cleaned_data.csv', index=False)
   ```

3. **Balance Dataset:**
   ```python
   from sklearn.utils import resample
   
   # Separate classes
   df_majority = df[df['LABEL'] == 0]
   df_minority = df[df['LABEL'] == 1]
   
   # Downsample majority class
   df_majority_downsampled = resample(df_majority, 
                                    replace=False,
                                    n_samples=len(df_minority),
                                    random_state=42)
   
   # Combine classes
   df_balanced = pd.concat([df_majority_downsampled, df_minority])
   ```

**Prevention:**
- Validate data quality before upload
- Maintain balanced datasets when possible
- Document data collection and cleaning procedures

##  Feature Extraction Issues

### Issue: Memory Errors During Feature Extraction

**Symptoms:**
- "MemoryError" or "Out of memory" messages
- System freezing during feature extraction
- Process killed by system

**Solutions:**

1. **Reduce Dataset Size:**
   ```python
   # Process data in chunks
   chunk_size = 1000
   df = pd.read_csv('large_dataset.csv')
   
   for i in range(0, len(df), chunk_size):
       chunk = df.iloc[i:i+chunk_size]
       # Process chunk
       features = extract_features(chunk)
       # Save chunk results
       features.to_csv(f'features_chunk_{i}.csv', index=False)
   ```

2. **Optimize Feature Extraction:**
   ```python
   # Use smaller vocabulary for TF-IDF
   from sklearn.feature_extraction.text import TfidfVectorizer
   
   vectorizer = TfidfVectorizer(
       max_features=1000,  # Reduce from default
       max_df=0.95,
       min_df=2
   )
   ```

3. **Enable Garbage Collection:**
   ```python
   import gc
   
   # Force garbage collection
   gc.collect()
   
   # Monitor memory usage
   import psutil
   process = psutil.Process()
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

4. **Use Efficient Data Types:**
   ```python
   # Optimize data types
   df['user_id'] = df['user_id'].astype('category')
   df['LABEL'] = df['LABEL'].astype('int8')
   
   # Use sparse matrices for features
   from scipy.sparse import csr_matrix
   features_sparse = csr_matrix(features)
   ```

**Prevention:**
- Monitor system memory usage
- Process large datasets in chunks
- Use efficient data structures

### Issue: Model Download Failures

**Symptoms:**
- Network timeout errors
- "Model not found" errors
- Incomplete model downloads

**Solutions:**

1. **Check Internet Connection:**
   ```bash
   # Test connectivity
   ping huggingface.co
   curl -I https://huggingface.co
   ```

2. **Manual Model Download:**
   ```python
   from transformers import AutoTokenizer, AutoModel
   
   # Download with retry logic
   import time
   
   for attempt in range(3):
       try:
           tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
           model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
           print("Model downloaded successfully")
           break
       except Exception as e:
           print(f"Attempt {attempt + 1} failed: {e}")
           time.sleep(10)
   ```

3. **Configure Cache Directory:**
   ```python
   import os
   
   # Set custom cache directory
   os.environ['TRANSFORMERS_CACHE'] = './local_models'
   os.environ['HF_HOME'] = './local_models'
   ```

4. **Use Offline Mode:**
   ```python
   # Use local models only
   os.environ['TRANSFORMERS_OFFLINE'] = '1'
   ```

**Prevention:**
- Ensure stable internet connection
- Pre-download models when possible
- Use local model storage

##  Model Training Issues

### Issue: Training Failures

**Symptoms:**
- Training process crashes
- "Convergence warning" messages
- Poor model performance

**Solutions:**

1. **Check Data Preparation:**
   ```python
   # Verify features and labels
   print("Feature shape:", X.shape)
   print("Label shape:", y.shape)
   print("Label distribution:", y.value_counts())
   
   # Check for NaN values
   print("NaN in features:", X.isnull().sum().sum())
   print("NaN in labels:", y.isnull().sum())
   ```

2. **Adjust Model Parameters:**
   ```python
   from sklearn.linear_model import LogisticRegression
   
   # Increase max iterations
   model = LogisticRegression(max_iter=1000)
   
   # Use different solver
   model = LogisticRegression(solver='liblinear')
   
   # Adjust regularization
   model = LogisticRegression(C=0.1)
   ```

3. **Handle Class Imbalance:**
   ```python
   # Use class weights
   model = LogisticRegression(class_weight='balanced')
   
   # Or use SMOTE
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X, y)
   ```

4. **Scale Features:**
   ```python
   from sklearn.preprocessing import StandardScaler
   
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

**Prevention:**
- Validate data before training
- Use appropriate model parameters
- Monitor training progress

### Issue: Poor Model Performance

**Symptoms:**
- Low accuracy scores
- High bias or variance
- Inconsistent results across runs

**Solutions:**

1. **Analyze Performance Metrics:**
   ```python
   from sklearn.metrics import classification_report, confusion_matrix
   
   # Detailed performance analysis
   print(classification_report(y_true, y_pred))
   print(confusion_matrix(y_true, y_pred))
   
   # Check per-class performance
   from sklearn.metrics import precision_recall_fscore_support
   precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
   ```

2. **Feature Analysis:**
   ```python
   # Check feature importance
   if hasattr(model, 'feature_importances_'):
       importance = model.feature_importances_
       feature_names = X.columns
       
       # Sort by importance
       indices = importance.argsort()[::-1]
       print("Top 10 features:")
       for i in range(10):
           print(f"{feature_names[indices[i]]}: {importance[indices[i]]:.4f}")
   ```

3. **Cross-Validation Analysis:**
   ```python
   from sklearn.model_selection import cross_val_score
   
   # Check consistency across folds
   cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
   print(f"CV scores: {cv_scores}")
   print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
   ```

4. **Hyperparameter Tuning:**
   ```python
   from sklearn.model_selection import GridSearchCV
   
   # Grid search for best parameters
   param_grid = {
       'C': [0.1, 1, 10],
       'max_iter': [100, 500, 1000]
   }
   
   grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
   grid_search.fit(X, y)
   print("Best parameters:", grid_search.best_params_)
   ```

**Prevention:**
- Use appropriate evaluation metrics
- Perform thorough feature engineering
- Validate model assumptions

##  Web Interface Issues

### Issue: Application Won't Start

**Symptoms:**
- "Port already in use" errors
- Import errors on startup
- Flask application crashes

**Solutions:**

1. **Check Port Availability:**
   ```bash
   # Check if port 5000 is in use
   netstat -an | grep 5000  # Linux/macOS
   netstat -an | findstr 5000  # Windows
   
   # Kill process using port
   lsof -ti:5000 | xargs kill -9  # macOS/Linux
   ```

2. **Use Different Port:**
   ```python
   # In main.py, change port
   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5001, debug=True)
   ```

3. **Check Dependencies:**
   ```python
   # Test imports
   try:
       import flask
       import pandas
       import sklearn
       print("All dependencies available")
   except ImportError as e:
       print(f"Missing dependency: {e}")
   ```

4. **Review Application Logs:**
   ```bash
   # Check application logs
   tail -f logs/app.log
   
   # Check for specific errors
   grep -i error logs/app.log
   ```

**Prevention:**
- Use unique ports for development
- Verify all dependencies before starting
- Monitor application logs

### Issue: File Upload Problems

**Symptoms:**
- Upload button not working
- Files not processing after upload
- Timeout errors during upload

**Solutions:**

1. **Check File Size Limits:**
   ```python
   # In main.py, adjust file size limit
   app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB
   ```

2. **Verify File Permissions:**
   ```bash
   # Check directory permissions
   ls -la datasets/
   
   # Fix permissions if needed
   chmod 755 datasets/
   ```

3. **Test Upload Manually:**
   ```python
   # Test file processing outside web interface
   from src.data_processor import DataProcessor
   
   processor = DataProcessor()
   result = processor.process_dataset('test_file.csv', 'test_dataset')
   print("Processing result:", result)
   ```

4. **Check Browser Console:**
   - Open browser developer tools (F12)
   - Check for JavaScript errors
   - Verify AJAX requests are completing

**Prevention:**
- Test uploads with small files first
- Monitor server logs during uploads
- Validate file formats before upload

##  Performance Issues

### Issue: Slow Processing

**Symptoms:**
- Long processing times
- System becomes unresponsive
- High CPU or memory usage

**Solutions:**

1. **Profile Performance:**
   ```python
   import time
   import cProfile
   
   # Time specific operations
   start_time = time.time()
   result = your_function()
   end_time = time.time()
   print(f"Processing time: {end_time - start_time:.2f} seconds")
   
   # Profile code
   cProfile.run('your_function()')
   ```

2. **Optimize Data Processing:**
   ```python
   # Use vectorized operations
   df['new_column'] = df['column1'] + df['column2']  # Fast
   # Instead of: df['new_column'] = df.apply(lambda x: x['column1'] + x['column2'], axis=1)  # Slow
   
   # Use efficient data types
   df['category_col'] = df['category_col'].astype('category')
   ```

3. **Enable Parallel Processing:**
   ```python
   from joblib import Parallel, delayed
   
   # Parallel feature extraction
   results = Parallel(n_jobs=-1)(
       delayed(extract_features)(chunk) 
       for chunk in data_chunks
   )
   ```

4. **Monitor System Resources:**
   ```python
   import psutil
   
   # Monitor CPU and memory
   print(f"CPU usage: {psutil.cpu_percent()}%")
   print(f"Memory usage: {psutil.virtual_memory().percent}%")
   ```

**Prevention:**
- Use efficient algorithms and data structures
- Monitor resource usage regularly
- Optimize code based on profiling results

##  Security and Permission Issues

### Issue: Permission Denied Errors

**Symptoms:**
- Cannot create directories
- Cannot save files
- Cannot access models

**Solutions:**

1. **Check File Permissions:**
   ```bash
   # Check current permissions
   ls -la
   
   # Fix directory permissions
   chmod 755 datasets/ models/ logs/
   
   # Fix file permissions
   chmod 644 *.py *.json
   ```

2. **Run with Appropriate Privileges:**
   ```bash
   # On Windows, run as administrator if needed
   # On Linux/macOS, use sudo only if necessary
   sudo python main.py  # Use sparingly
   ```

3. **Change Ownership:**
   ```bash
   # Change ownership to current user
   chown -R $USER:$USER .
   ```

**Prevention:**
- Use appropriate file permissions
- Avoid running as root/administrator unless necessary
- Set up proper user permissions

##  Logging and Debugging

### Issue: Missing or Unclear Error Messages

**Symptoms:**
- Generic error messages
- No stack traces
- Missing log files

**Solutions:**

1. **Enable Debug Mode:**
   ```python
   # In main.py
   app.run(debug=True)
   
   # Or set environment variable
   export FLASK_DEBUG=1
   ```

2. **Increase Logging Level:**
   ```python
   import logging
   
   # Set to DEBUG level
   logging.basicConfig(level=logging.DEBUG)
   
   # Add more detailed logging
   logger = logging.getLogger(__name__)
   logger.debug("Detailed debug information")
   ```

3. **Check Log Files:**
   ```bash
   # View recent logs
   tail -f logs/app.log
   
   # Search for specific errors
   grep -i "error\|exception\|traceback" logs/app.log
   ```

4. **Add Custom Logging:**
   ```python
   import logging
   
   def debug_function(data):
       logger = logging.getLogger(__name__)
       logger.info(f"Processing {len(data)} records")
       
       try:
           result = process_data(data)
           logger.info(f"Successfully processed {len(result)} records")
           return result
       except Exception as e:
           logger.error(f"Processing failed: {e}", exc_info=True)
           raise
   ```

**Prevention:**
- Use appropriate logging levels
- Include context in log messages
- Regularly review log files

##  Emergency Recovery

### Issue: System Completely Broken

**Symptoms:**
- Nothing works
- Multiple error messages
- Cannot start application

**Emergency Recovery Steps:**

1. **Complete Reinstallation:**
   ```bash
   # Backup important data
   cp -r datasets/ datasets_backup/
   cp -r models/ models_backup/
   
   # Remove virtual environment
   rm -rf .venv
   
   # Recreate environment
   python -m venv .venv
   source .venv/bin/activate
   
   # Reinstall dependencies
   pip install -r requirements.txt
   
   # Restore data
   cp -r datasets_backup/ datasets/
   cp -r models_backup/ models/
   ```

2. **Reset to Default Configuration:**
   ```bash
   # Backup current config
   cp config.json config.json.backup
   
   # Reset to default
   git checkout config.json  # If using git
   # Or manually recreate config.json
   ```

3. **Clear All Caches:**
   ```bash
   # Clear Python cache
   find . -type d -name "__pycache__" -exec rm -rf {} +
   find . -name "*.pyc" -delete
   
   # Clear model cache
   rm -rf local_models/
   
   # Clear temporary files
   rm -rf /tmp/transformers_cache/
   ```

4. **Start Fresh:**
   ```bash
   # Test with minimal example
   python -c "
   from src.data_processor import DataProcessor
   processor = DataProcessor()
   print('Basic functionality works')
   "
   ```

##  Getting Additional Help

### When to Seek Help
- Error persists after trying documented solutions
- System behavior is inconsistent or unpredictable
- Performance issues cannot be resolved
- Security concerns arise

### How to Report Issues
1. **Gather Information:**
   - Error messages and stack traces
   - System information (OS, Python version)
   - Steps to reproduce the issue
   - Log files (last 50 lines)

2. **Create Minimal Example:**
   ```python
   # Minimal code that reproduces the issue
   import pandas as pd
   from src.data_processor import DataProcessor
   
   # Your minimal example here
   ```

3. **Check Existing Issues:**
   - Search documentation for similar problems
   - Check GitHub issues (if applicable)
   - Review community forums

### Support Channels
- **Documentation**: Check all relevant documentation sections
- **Logs**: Review application logs for detailed error information
- **Community**: Engage with user community for peer support
- **Issues**: Report bugs with detailed reproduction steps

---

**Remember: Most issues can be resolved by carefully following the solutions in this guide. Take your time and work through the steps systematically.** 