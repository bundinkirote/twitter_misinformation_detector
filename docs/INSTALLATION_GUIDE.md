# Installation Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.11+ (tested with Python 3.13.2)
- **RAM**: 8GB minimum
- **Storage**: 5GB free disk space
- **Internet**: Required for downloading transformer models

### Recommended Requirements
- **RAM**: 16GB or more
- **Storage**: 10GB+ free disk space
- **CPU**: Multi-core processor (4+ cores)
- **Note**: GPU/CUDA acceleration is not currently implemented. Processing runs on CPU.

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Python Installation
Ensure Python 3.11+ is installed on your system:

**Windows:**
1. Download Python from [python.org](https://python.org)
2. Run the installer and check "Add Python to PATH"
3. Verify installation:
   ```cmd
   python --version
   ```

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.11

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip
```

#### Step 2: Project Setup
1. **Navigate to project directory:**
   ```bash
   cd c:\Users\proff\Documents\ML2\new
   ```

2. **Create virtual environment:**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Upgrade pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

#### Step 3: Install Dependencies
```bash
# Standard installation
pip install -r requirements.txt

# For macOS users (if you encounter issues)
pip install -r requirements_macos.txt
```

#### Step 4: Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

#### Step 5: Verify Installation
```bash
python -c "import flask, pandas, sklearn, nltk, transformers; print('All dependencies installed successfully!')"
```

### Method 2: macOS Automated Setup

For macOS users, use the provided setup script:

```bash
chmod +x setup_macos.sh
./setup_macos.sh
```

This script will:
- Install Python dependencies
- Download NLTK data
- Create necessary directories
- Verify the installation

### Method 3: Docker Installation (Advanced)

#### Prerequisites
- Docker installed on your system
- Docker Compose (optional)

#### Docker Setup
1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Download NLTK data
   RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"

   # Copy application code
   COPY . .

   # Create necessary directories
   RUN mkdir -p logs models datasets static/plots static/visualizations

   # Expose port
   EXPOSE 5000

   # Run application
   CMD ["python", "main.py"]
   ```

2. **Build and run:**
   ```bash
   docker build -t misinformation-detector .
   docker run -p 5000:5000 -v $(pwd)/datasets:/app/datasets misinformation-detector
   ```

## Platform-Specific Instructions

### Windows Installation

#### Using Command Prompt
```cmd
# Navigate to project directory
cd c:\Users\proff\Documents\ML2\new

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

#### Using PowerShell
```powershell
# Navigate to project directory
Set-Location "c:\Users\proff\Documents\ML2\new"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# If execution policy error occurs:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt
```

#### Common Windows Issues
1. **Long Path Names**: Enable long path support in Windows
2. **Antivirus Interference**: Add project folder to antivirus exclusions
3. **Permission Issues**: Run as administrator if needed

### macOS Installation

#### Using Terminal
```bash
# Navigate to project directory
cd /path/to/project

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements_macos.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

#### macOS-Specific Dependencies
The `requirements_macos.txt` includes platform-specific versions:
- Optimized NumPy and SciPy builds
- Metal Performance Shaders support
- macOS-compatible PyTorch builds

#### Common macOS Issues
1. **Xcode Command Line Tools**: Install with `xcode-select --install`
2. **Homebrew Dependencies**: Some packages may require Homebrew
3. **M1/M2 Compatibility**: Use native ARM64 builds when available

### Linux Installation

#### Ubuntu/Debian
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install python3.11 python3.11-venv python3.11-dev python3-pip build-essential

# Navigate to project directory
cd /path/to/project

# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### CentOS/RHEL/Fedora
```bash
# Install Python and development tools
sudo dnf install python3.11 python3.11-venv python3.11-devel gcc gcc-c++ make

# Follow same steps as Ubuntu
```

#### Common Linux Issues
1. **Missing Development Headers**: Install `python3-dev` or `python3-devel`
2. **Compiler Issues**: Install `build-essential` or equivalent
3. **Permission Issues**: Use `sudo` only for system packages

## GPU Support

GPU/CUDA acceleration is NOT currently implemented in this system. All processing runs on CPU using PyTorch. The following information is provided for reference if GPU support is added in future versions:

### NVIDIA GPU (Future Implementation)
For future CUDA support:
```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verification
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Dependency Details

### Core Dependencies
```
Flask==2.3.3              # Web framework
pandas==2.1.1             # Data manipulation
numpy==1.24.3             # Numerical computing
scikit-learn==1.3.0       # Machine learning
nltk==3.8.1               # Natural language processing
transformers==4.33.2      # Transformer models
torch==2.0.1              # Deep learning framework
```

### Visualization Dependencies
```
matplotlib==3.7.2         # Plotting library
seaborn==0.12.2           # Statistical visualization
plotly==5.16.1            # Interactive plots
```

### Network Analysis
```
networkx==3.1             # Network analysis
community==1.0.0b1        # Community detection
```

### Optional Dependencies
```
numba==0.57.1             # JIT compilation for performance
shap==0.42.1              # Model explainability
```

## Post-Installation Setup

### 1. Create Directory Structure
The application will create necessary directories automatically, but you can create them manually:

```bash
mkdir -p logs models datasets static/plots static/visualizations local_models
```

### 2. Configuration
Edit `config.json` to customize settings:
```json
{
  "app_name": "Twitter Misinformation Detection System",
  "version": "1.0.0",
  "debug": false,
  "max_file_size": "16MB",
  "supported_formats": [".csv", ".xlsx"]
}
```

### 3. Test Installation
Run the test suite to verify everything is working:
```bash
python -m pytest tests/ -v
```

### 4. First Run
Start the application:
```bash
python main.py
```

Visit `http://localhost:5000` to access the web interface.

## Troubleshooting

### Common Installation Issues

#### 1. Python Version Issues
```bash
# Check Python version
python --version

# If wrong version, use specific version
python3.11 -m venv .venv
```

#### 2. Pip Installation Failures
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Use --no-cache-dir for clean install
pip install --no-cache-dir -r requirements.txt

# Install with verbose output for debugging
pip install -v package_name
```

#### 3. Virtual Environment Issues
```bash
# Deactivate current environment
deactivate

# Remove and recreate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

#### 4. NLTK Download Issues
```python
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
```

#### 5. Transformer Model Download Issues
```python
# Set cache directory
import os
os.environ['TRANSFORMERS_CACHE'] = './local_models'

# Download models manually
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
```

### Memory Issues
If you encounter memory issues:

1. **Reduce batch size** in configuration
2. **Use smaller models** (e.g., DistilBERT instead of BERT)
3. **Increase virtual memory** (swap space)
4. **Process data in chunks**

### Network Issues
For network-related problems:

1. **Check firewall settings**
2. **Configure proxy settings** if behind corporate firewall
3. **Use offline model downloads** if internet is limited

### Performance Optimization

#### 1. Enable Multiprocessing
```python
# In config.json
{
  "multiprocessing": true,
  "n_jobs": -1
}
```

#### 2. Use Faster BLAS Libraries
```bash
# Install Intel MKL
pip install mkl

# Or OpenBLAS
pip install openblas
```

#### 3. Enable JIT Compilation
```bash
pip install numba
```

### Logging and Monitoring

The system includes comprehensive logging for debugging and monitoring:

**Log Configuration**:
- **Log Level**: INFO (adjustable via LOG_LEVEL environment variable)
- **Log File Location**: `logs/app.log`
- **Log Format**: Timestamp, level, module, and message
- **Handlers**: StreamHandler (console) and FileHandler (file)
- **Note**: Uses fixed file handler (not rotating) suitable for single-user research deployment

**Interaction Logging**:
- **Location**: `logs/interactions/` directory
- **Format**: JSONL (one JSON record per line)
- **Daily Files**: `interactions_YYYYMMDD.jsonl`
- **Includes**: User sessions, predictions, model selections, visualizations

After installation, verify these components:

- [ ] Python 3.11+ installed and accessible
- [ ] Virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] NLTK data downloaded successfully
- [ ] Application starts without errors
- [ ] Web interface accessible at localhost:5000
- [ ] Can upload and process a sample dataset
- [ ] Models can be trained successfully
- [ ] Visualizations generate correctly

## Getting Help

If you encounter issues:

1. **Check the logs**: Look in `logs/app.log` for error messages
2. **Run diagnostics**: Use the built-in system diagnostics
3. **Test components**: Run individual test files
4. **Check dependencies**: Verify all packages are correctly installed
5. **Review documentation**: Check API documentation for usage examples

## Next Steps

After successful installation:

1. **Read the User Guide**: Understand how to use the system
2. **Try the Tutorial**: Follow the step-by-step tutorial
3. **Explore Examples**: Check the examples directory
4. **Review API Documentation**: For programmatic access
5. **Join the Community**: Connect with other users and developers

---

**Installation complete! Ready to detect misinformation!** 