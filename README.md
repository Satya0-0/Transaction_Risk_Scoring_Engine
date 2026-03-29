# Transaction Risk Scoring Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://risk-engine.streamlit.app/)

A machine learning-based transaction risk assessment system that leverages LightGBM to calculate fraud probability scores and expected values, enabling strategic transaction pooling for optimal fraud prevention resource allocation.

---
## 🔗 [Launch Live Interactive Demo](https://risk-engine.streamlit.app/)
---


## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running Predictions](#running-predictions)
- [Configuration](#configuration)
- [Key Components](#key-components)
- [Data Requirements](#data-requirements)
- [Risk Scoring Methodology](#risk-scoring-methodology)
- [Performance Metrics](#performance-metrics)

## Overview

This system implements an end-to-end transaction risk scoring workflow designed to optimize fraud investigation resource allocation. Rather than focusing solely on classification accuracy, the engine prioritizes actionable risk assessment through expected value calculations and strategic transaction pooling.

### Key Capabilities

- **Intelligent Data Processing**: Merges transaction and identity datasets with memory-optimized handling
- **Feature Engineering**: Extracts temporal patterns and selects high-impact features
- **Probabilistic Risk Scoring**: Generates fraud probability scores using LightGBM
- **Expected Value Analysis**: Calculates investigation cost-benefit for each transaction
- **Strategic Pooling**: Categorizes transactions into risk-based action buckets
- **Production-Ready Pipeline**: Serializes complete preprocessing and prediction workflow

## Project Structure

```
FDS/
├── config.py                      # System configuration and parameters
├── data/
│   ├── raw/                       # Source datasets
│   │   ├── train_transaction.csv
│   │   ├── train_identity.csv
│   │   ├── test_transaction.csv
│   │   └── test_identity.csv
│   └── processed/                 # Generated processed data
├── models/
│   └── lgbm_pipeline_v1.joblib   # Serialized model pipeline
├── notebooks/                     # Exploratory analysis notebooks
├── src/
│   ├── app.py                     # Inference interface
│   ├── train.py                   # Model training orchestration
│   └── util/
│       ├── buckets_validation.py  # Pooling logic and validation
│       ├── data_loader.py         # Data ingestion and merging
│       ├── data_split.py          # Train/validation splitting
│       ├── feature_engineering.py # Custom sklearn transformers
│       ├── memory_reduction.py    # Memory optimization utilities
│       ├── model_save.py          # Model serialization
│       ├── random_test_record.py  # Test record sampling
│       └── reproducibility.py     # Seed management
└── README.md
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd FDS
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. Install required dependencies:
```bash
pip install pandas numpy scikit-learn lightgbm joblib
```

## Usage

### Training the Model

Execute the training pipeline to build and validate the risk scoring model:

```bash
python -m src.train
```

The training process performs the following operations:

1. Loads and merges transaction and identity datasets
2. Splits data into training (80%) and validation (20%) sets
3. Applies feature engineering transformations
4. Trains LightGBM classifier on balanced data
5. Generates risk scores and expected values for validation set
6. Pools transactions into strategic buckets
7. Calculates and displays validation metrics
8. Conditionally saves pipeline if performance thresholds are met:
   - Precision@1000 > 0.8 for Pool 0
   - Lift > 20 for Pool 0

### Running Predictions

Generate risk assessments for test transactions:

```bash
python -m src.app
```

Sample output:
```json
Testing for record N of the test set
{
  "Transaction Risk Score": 0.87,
  "Expected Value": 342.50,
  "Pool": "P0"
}
```

## Configuration

System parameters are managed through `config.py`:

```python
CONFIG = {
    'seed': 42,                          # Reproducibility seed
    'split_point': 0.8,                  # Train/validation ratio
    'Inspection_Cost': 50,               # Cost per manual investigation ($)
    'lgb_params': {
        'objective': 'binary',           # Binary classification task
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'min_data_in_leaf': 100,
        'is_unbalance': True,            # Handles class imbalance
        'n_estimators': 100,
        'verbosity': -1
    }
}
```

Modify these parameters to adjust model behavior and business logic.

## Key Components

### Data Pipeline (`src/util/data_loader.py`)
Handles data ingestion, merging transaction and identity datasets on common keys, and applies memory optimization techniques to manage large-scale data efficiently.

### Feature Engineering (`src/util/feature_engineering.py`)
Implements custom sklearn transformers:
- **ExtractMonth**: Derives temporal features from transaction timestamps
- **SelectFeatures50**: Selects top 50 features by importance
- **TypeConverter**: Optimizes memory through intelligent dtype conversion

### Model Architecture
The pipeline integrates:
1. Temporal feature extraction
2. Feature importance-based selection
3. Memory-optimized type conversion
4. LightGBM gradient boosting classifier

### Validation & Pooling (`src/util/buckets_validation.py`)
Core business logic implementation:
- **calculate_EV**: Computes expected value incorporating risk score, transaction amount, and investigation cost
- **pooling**: Assigns transactions to risk-based action pools
- **validation_metrics**: Generates performance metrics stratified by pool

### Model Persistence (`src/util/model_save.py`)
Manages serialization and deserialization of trained pipelines using joblib, ensuring reproducible inference.

## Data Requirements

### Input Datasets

**Transaction Data** (`train_transaction.csv`, `test_transaction.csv`):
- Single row per transaction
- Features include transaction amount, merchant information, device data, and temporal attributes

**Identity Data** (`train_identity.csv`, `test_identity.csv`):
- Single row per transaction
- Features include customer identity verification signals and behavioral indicators

Datasets merge on shared transaction identifiers to create the complete feature space.

### Output Format

The system generates three key outputs per transaction:
- **Risk Score**: Fraud probability (range: 0-1)
- **Expected Value**: Cost-benefit metric for investigation decision
- **Pool Assignment**: Strategic bucket (P0, P1, P2) for action prioritization

## Risk Scoring Methodology

### Model Performance

The LightGBM classifier achieves balanced performance with:
- **Recall**: 55% (identifies over half of fraudulent transactions)
- **Precision**: 38% (optimized for F1-score balance)

While the model misses approximately 45% of fraud cases, the threshold selection prioritizes the balance between recall and precision to support the expected value framework rather than maximize a single metric.

### Expected Value Calculation

The system calculates investigation cost-benefit using:

```
EV = (Risk_Score × Transaction_Amount) - Investigation_Cost
```

Where:
- **Risk_Score**: Model-generated fraud probability (0-1)
- **Transaction_Amount**: Monetary value at risk
- **Investigation_Cost**: Manual review cost per transaction (default: $50)

### Transaction Pooling Strategy

Transactions are assigned to action pools based on expected value and risk thresholds:

**Pool 0 (P0) - High Priority Investigation**
- Criteria: `EV ≥ 15 × IC` OR `Risk_Score ≥ 0.9`
- Rationale: High-value potential losses or extremely suspicious patterns warrant immediate investigation
- Action: Prioritize for manual review, ordered by transaction value

**Pool 1 (P1) - Moderate Priority**
- Criteria: `(5 × IC ≤ EV < 15 × IC)` OR `(0.75 ≤ Risk_Score < 0.9)`
- Rationale: Moderate risk-reward profile justifies selective investigation
- Action: Review based on available resources

**Pool 2 (P2) - Low Priority**
- Criteria: `EV < 5 × IC`
- Rationale: Investigation cost exceeds expected value protection
- Action: Monitor passively; no immediate investigation

## Performance Metrics

### Validation Results by Pool

The model demonstrates strong discrimination in high-priority pools:

| Pool | Precision@1000 | Recall@1000 | Lift |
|------|----------------|-------------|------|
| P0   | 0.875          | 0.215       | 25.74|
| P1   | 0.252          | 0.062       | 7.41 |
| P2   | 0.119          | 0.029       | 3.50 |

**Key Insights:**
- Pool 0 achieves 87.5% precision in its top 1000 transactions, delivering 25.7× lift over baseline fraud rate
- The system successfully concentrates high-value fraud cases in the priority investigation queue
- Meets production deployment thresholds (Precision@1000 > 0.8, Lift > 20 for P0)

### Model Configuration

The LightGBM implementation uses:
- **Objective**: Binary classification (fraud detection)
- **Optimization Metrics**: AUC-ROC and binary log loss
- **Algorithm**: Gradient Boosting Decision Trees (GBDT)
- **Class Imbalance**: Handled via `is_unbalance=True`
- **Ensemble Size**: 100 boosting iterations
- **Leaf Constraints**: Minimum 100 samples per leaf for generalization

---

**Note**: This system prioritizes strategic resource allocation over raw classification performance. The pooling methodology enables financial institutions to maximize fraud prevention ROI by focusing investigation resources on transactions with the highest expected value protection.
