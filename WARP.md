# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a Japanese neural network research project for B-H (magnetic hysteresis loop) modeling and prediction using fully connected deep neural networks. The system predicts magnetic hysteresis behavior at arbitrary flux density amplitudes using PyTorch-based neural networks.

## Development Commands

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
py -m pytest

# Run specific test module
py -m pytest tests/test_1_nn.py

# Run tests with verbose output
py -m pytest -v

# Run tests for Training Data Folder specifically
py -m pytest "1.Training Data Folder/tests/"
```

### Running the Main Neural Network Script
```bash
# Navigate to the project root
cd "C:\Users\RM-2503-1\Desktop\M1\3_研究\NN_perf"

# Run the main NN script from project root
python "src/1. NN.py"
```

### Data Visualization
```bash
# Check specific hysteresis loop data plots
python plot_check.py
```

## Architecture Overview

### Core Components

**Neural Network Architecture (`src/1. NN.py`)**
- Fully connected feedforward neural network using PyTorch
- Configurable hidden layers and activation functions via `config/1. NN.ini`
- Supports ReLU, Tanh, and Sigmoid activation functions
- Custom RMSE loss function implementation alongside standard MSE
- Model persistence with state dict, scalers, and metadata saving

**Configuration System**
- INI-based configuration in `config/1. NN.ini`
- Controls training parameters (learning rate, epochs, batch size)
- Architecture settings (hidden layers, activation functions)
- Data ranges for training and regression
- Material properties and frequency settings

**Data Pipeline Architecture**
The system processes magnetic measurement data through several stages:
1. Raw hysteresis loop data in `1.Training Data Folder/assets`
2. Akima spline interpolation data from `2.Normal Magnetization Curve Extraction Folder`
3. Combined training data with automatic data cleaning and validation
4. Standardized inputs/outputs using sklearn StandardScaler

### Key Design Patterns

**Model Management**
- Automatic model saving/loading with configuration validation
- Smart model reuse when settings match previous training runs
- Separate storage of model weights, scalers, and metadata
- Training can be bypassed by setting `PERFORM_TRAINING = False`

**Error Handling & Data Validation**
- Comprehensive NaN/Inf detection and cleaning in training data
- Akima data filtering based on target regression amplitudes
- Gradient clipping to prevent exploding gradients
- Loss validation during training to detect NaN losses

**Results Generation**
- RMSE calculation against reference data
- Excel output with embedded charts using openpyxl
- Matplotlib plots with Japanese font support (japanize-matplotlib)
- Comparison sheets for each amplitude prediction

### Data Flow

1. **Training Data Assembly**: Combines hysteresis loop data from multiple amplitudes with Akima interpolation points
2. **Data Preprocessing**: StandardScaler normalization of both inputs (amplitude, B-field) and outputs (H-field)  
3. **Model Training**: PyTorch DataLoader with configurable batch size and Adam optimizer
4. **Prediction Phase**: Model generates predictions across specified amplitude ranges
5. **Validation & Output**: RMSE comparison against reference data with Excel and plot generation

### File Structure Logic

- `src/` - Main Python scripts
- `config/` - INI configuration files (git-ignored for environment-specific settings)
- `tests/` - pytest unit tests with comprehensive fixtures in `conftest.py`
- `1.Training Data Folder/` - Training data preparation (has own README)
- `3.Answer/` - Output directory for results and trained models
- `plot_check.py` - Utility for visualizing specific hysteresis data

### Testing Infrastructure

The test suite uses pytest with sophisticated module importing via `conftest.py`:
- Dynamic import of numbered Python files (e.g., "1. NN.py")
- Mocking of GUI libraries (matplotlib, japanize-matplotlib) to prevent window spawning
- Parameterized tests for activation functions and model architectures
- Mock configuration files for isolated testing

### Configuration Management

Settings are centralized in INI files but the main script validates consistency between saved models and current configuration. This prevents accidental model misuse when parameters change.

Key configurable aspects:
- Neural network architecture (layers, nodes, activation)
- Training hyperparameters (learning rate, epochs, batch size)
- Data amplitude ranges for training and regression
- Material properties and frequency settings

## Important Notes

- File paths contain Japanese characters and spaces - always use full quoted paths
- The system expects specific Excel file naming conventions for hysteresis data
- Model training can be computationally intensive (default: 100,000 epochs)
- Results include both visual plots and detailed Excel reports with embedded charts
- Configuration changes require model retraining for safety