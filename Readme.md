# Credit Card Fraud Detection

A comprehensive machine learning project implementing both **supervised classification** and **unsupervised anomaly detection** approaches to identify fraudulent credit card transactions.

## Project Overview

This project tackles the challenge of credit card fraud detection using a dataset of **284,807 transactions**, employing two distinct methodologies:

1. **Supervised Classification**: Leveraging labeled data to train classification models
2. **Unsupervised Anomaly Detection**: Identifying fraudulent patterns without relying on labels

The dual-approach strategy allows for robust fraud detection that can work both with and without labeled data, making it adaptable to real-world scenarios.

## Dataset

- **Total Transactions**: 284,807
- **Features**: 30 (Time, V1-V28, Amount)
- **Target**: Class (0 = Legitimate, 1 = Fraudulent)
- **Note**: Features V1-V28 are PCA-transformed for confidentiality

### Data Split Strategy

**Temporal Split (80-20)**: To simulate real-world scenarios where models are trained on historical data and tested on future transactions:
- Training Set: 227,846 transactions (first 80%)
- Validation Set: 56,961 transactions (last 20%)

This temporal approach prevents data leakage and provides a realistic evaluation of model performance.

## Approach 1: Supervised Classification

### Models Implemented

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **XGBoost Classifier**

### Preprocessing Pipeline

```python
1. Temporal data split (80-20)
2. Feature scaling using StandardScaler
3. Class balancing techniques applied
```

### Model Performance Summary

**Note**: The results below represent the **best-performing runs** for each model. Multiple experiments with various hyperparameters were conducted, and all runs are tracked in MLflow for complete reproducibility.

| Model | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test ROC-AUC |
|-------|----------------|---------------|----------------|-------------|--------------|
| Logistic Regression | 97.96% | 98.44% | 7.07% | 89.33% | 93.89% |
| Decision Tree | 98.50% | 98.70% | 7.44% | 77.33% | 88.03% |
| **Random Forest** | **99.96%** | **99.96%** | **92.06%** | **77.33%** | **88.66%** |
| XGBoost | 99.997% | 99.94% | 88.89% | 64.00% | 81.99% |

**Best Model**: Random Forest Classifier achieved the highest test accuracy (99.96%) with excellent precision (92.06%).

*All hyperparameter tuning experiments, including intermediate runs and alternative configurations, are logged in MLflow for full transparency and experiment tracking.*

### Key Features - Classification

- Balanced class weights to handle imbalanced data
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, ROC-AUC)
- MLflow experiment tracking for all model runs
- Temporal validation to prevent data leakage

##  Approach 2: Unsupervised Anomaly Detection

### Model Implemented

**Isolation Forest**: An ensemble-based anomaly detection algorithm that isolates anomalies instead of profiling normal data points.

### Contamination Parameter Tuning

Multiple experiments were conducted with varying contamination parameters to optimize fraud detection. **The table below shows the best results from systematic experimentation**:

| Version | Contamination | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test ROC-AUC |
|---------|---------------|----------------|---------------|----------------|-------------|--------------|
| V1 | 0.002 | 99.74% | 99.74% | 1.35% | 1.33% | 50.60% |
| V2 | 0.006 | 99.41% | 99.43% | 5.67% | 21.33% | 60.43% |
| V3 | 0.010 | 99.04% | 99.15% | 7.28% | 46.67% | 72.94% |
| V4 | 0.015 | 98.56% | 98.69% | 5.67% | 57.33% | 78.04% |
| V5 | 0.0175 | 98.32% | 98.44% | 4.98% | 60.00% | 79.25% |
| **V6** | **0.020** | **98.08%** | **98.23%** | **4.57%** | **62.67%** | **80.47%** |

**Best Configuration**: Contamination = 0.020, achieving the highest test ROC-AUC (80.47%) and recall (62.67%).

*Additional experiments with different hyperparameters (n_estimators, max_samples, etc.) are available in MLflow tracking.*

### Key Features - Anomaly Detection

- No reliance on labeled training data
- Effective for scenarios with limited fraud examples
- Adjustable contamination parameter for sensitivity tuning
- Works well with imbalanced datasets

##  Technologies Used

- **Python 3.11**
- **scikit-learn**: Model implementation and preprocessing
- **XGBoost**: Gradient boosting framework
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **MLflow**: Experiment tracking and model registry
- **joblib**: Model serialization

## Project Structure

```
Credit-Card-Fraud-Detection/
├── Data/
│   └── creditcard.csv
├── Models/
│   ├── rf.pkl              # Random Forest model
│   └── iso.pkl             # Isolation Forest model
├── mlruns/                 # MLflow experiment tracking
├── CC_Classification.ipynb # Supervised learning notebook
├── CC_Unsupervised.ipynb  # Anomaly detection notebook
└── README.md
```

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn xgboost mlflow joblib
```

### Running Classification Models

```python
# Load the notebook
jupyter notebook CC_Classification.ipynb

# The notebook includes:
# - Data loading and preprocessing
# - Model training with multiple algorithms
# - Performance evaluation
# - MLflow tracking
```

### Running Anomaly Detection

```python
# Load the notebook
jupyter notebook CC_Unsupervised.ipynb

# The notebook includes:
# - Data preprocessing
# - Isolation Forest implementation
# - Contamination parameter tuning
# - Performance metrics
```

## Experiment Tracking

All experiments are tracked using **MLflow** for complete transparency and reproducibility:

- **Classification Experiments**: `Credit Card Fraud Detection ==> Classification`
  - Multiple runs per model with different hyperparameters
  - Includes Logistic Regression, Decision Tree, Random Forest, and XGBoost variants
  - All metrics (accuracy, precision, recall, ROC-AUC) logged for train and test sets
  
- **Unsupervised Experiments**: `Credit Card Fraud Detection ==> Unsupervised`
  - 6+ Isolation Forest configurations with varying contamination parameters
  - Systematic exploration of hyperparameter space
  - Complete parameter and metric logging

**Note**: The performance tables in this README show only the **best results** from each approach. To explore all experimental runs, intermediate results, and alternative configurations:

```bash
mlflow ui
```

This will launch the MLflow UI where you can:
- Compare all experiment runs side-by-side
- View hyperparameter configurations
- Analyze metric evolution across experiments
- Access model artifacts and logs

## Key Learnings

1. **Class Imbalance Handling**: Balanced class weights significantly improve minority class detection
2. **Temporal Validation**: Essential for time-series financial data to prevent data leakage
3. **Model Trade-offs**: 
   - Random Forest offers best precision for classification
   - Isolation Forest provides label-independent fraud detection
4. **Hyperparameter Tuning**: Contamination parameter critically affects anomaly detection performance

##  Future Enhancements

- Implement ensemble methods combining both approaches
- Add SMOTE or other oversampling techniques
- Deploy models using Flask/FastAPI
- Real-time fraud detection pipeline
- Cost-sensitive learning to minimize false negatives
- Deep learning approaches (Autoencoders, Neural Networks)

##  Performance Comparison

### Supervised vs Unsupervised

| Metric | Random Forest (Supervised) | Isolation Forest (Unsupervised) |
|--------|----------------------------|----------------------------------|
| Test Accuracy | 99.96% | 98.23% |
| Test Precision | 92.06% | 4.57% |
| Test Recall | 77.33% | 62.67% |
| Test ROC-AUC | 88.66% | 80.47% |

**Insight**: Supervised learning excels with labeled data, while unsupervised provides reasonable detection without labels.

##  Model Persistence

Trained models are saved for deployment:

```python
# Load Random Forest model
from joblib import load
rf_model = load('Models/rf.pkl')

# Load Isolation Forest model
iso_model = load('Models/iso.pkl')
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Created as part of a comprehensive machine learning portfolio project.

## Acknowledgments

- Dataset source: Credit card transactions dataset (Kaggle)
- MLflow for experiment tracking capabilities
- scikit-learn community for robust ML tools

---

**Note**: This project demonstrates both supervised and unsupervised approaches to fraud detection, showcasing the versatility of machine learning in handling different data scenarios and business requirements.