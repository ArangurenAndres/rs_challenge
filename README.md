# RS Challenge - Robust Sparse Recommendation System

A comprehensive recommendation system specifically designed for extremely sparse datasets with advanced fallback strategies and ensemble methods.

## 🎯 Project Overview

This project implements a robust recommendation system capable of handling highly sparse interaction data (>99.9% sparsity). It combines multiple recommendation techniques in an ensemble approach to provide reliable recommendations even for cold-start users and items with minimal interaction data.

## ✨ Key Features

### Core Algorithms
- **Matrix Factorization**: Implicit ALS and BPR models for collaborative filtering
- **Dimensionality Reduction**: TruncatedSVD for handling sparse matrices
- **Content-Based Filtering**: TF-IDF vectorization of item metadata
- **Co-occurrence Analysis**: Item-to-item relationships from user behavior
- **Ensemble Learning**: Weighted combination of multiple recommendation strategies

### Advanced Capabilities
- **Temporal Modeling**: Time-decay weighting for user interactions
- **Bias Correction**: User and item bias terms for better score calibration
- **Cold Start Handling**: Sophisticated fallback strategies for new users/items
- **Diversity Optimization**: Category-based diversification of recommendations
- **Robust Error Handling**: Graceful degradation with multiple fallback layers

### Data Processing
- **Metadata Integration**: Efficient processing of item features (category, price, quality)
- **Sparse Matrix Optimization**: Memory-efficient handling of large sparse datasets
- **Popularity-Based Baselines**: Multiple popularity scoring methods
- **Feature Engineering**: Advanced temporal and behavioral feature extraction

## 📁 Project Structure

```
rs_challenge/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── .gitignore                                  # Git ignore patterns
├── robust_sparse_recommender.py               # Main recommendation system
├── data_analysis.py                           # Data exploration and analysis
├── train.csv                                  # Training interaction data
├── test.csv                                   # Test user data
├── item_meta.csv                              # Item metadata
├── robust_sparse_recommendation_submission.csv # Generated predictions
└── venv/                                      # Virtual environment (ignored)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rs_challenge
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Data Analysis
Run comprehensive data analysis to understand your dataset:
```bash
python data_analysis.py
```

This will provide:
- Dataset statistics and sparsity analysis
- User behavior patterns
- Item popularity distribution
- Cold start analysis
- Temporal patterns (if timestamps available)
- Modeling recommendations based on data characteristics

### Training and Prediction
Generate recommendations using the robust system:
```bash
python robust_sparse_recommender.py
```

This will:
1. Load training and test data
2. Process item metadata (if available)
3. Train multiple recommendation models
4. Generate ensemble predictions
5. Create submission file with top-10 recommendations per user

### Custom Usage
```python
from robust_sparse_recommender import RobustSparseRecommendationSystem
import pandas as pd

# Load your data
train_df = pd.read_csv('train.csv')
item_meta_df = pd.read_csv('item_meta.csv')

# Initialize system
rec_sys = RobustSparseRecommendationSystem(n_factors=150, alpha=60)

# Train the model
rec_sys.fit(train_df, item_meta_df)

# Get recommendations for a user
recommendations = rec_sys.recommend(user_id=123, top_k=10)
print(f"Top 10 recommendations: {recommendations}")
```

## 🛠️ Key Components

### RobustSparseRecommendationSystem Class

**Core Methods:**
- `fit(train_df, item_meta_df)`: Train all models with robust error handling
- `recommend(user_id, top_k)`: Generate top-k recommendations for a user
- `get_ensemble_recommendations()`: Combine multiple model outputs

**Specialized Recommendation Methods:**
- `get_robust_model_recommendations()`: ALS/BPR collaborative filtering
- `get_svd_recommendations()`: Matrix factorization approach
- `get_cooccurrence_recommendations()`: Item-to-item similarity
- `get_content_recommendations()`: Content-based filtering
- `get_fallback_recommendations()`: Popularity and metadata-based fallbacks

### Data Processing Features
- **Temporal Weighting**: Exponential decay for older interactions
- **Metadata Processing**: Category, price range, and quality analysis
- **Co-occurrence Matrix**: User-item interaction patterns
- **Bias Computation**: User and item bias correction

## 📈 Performance Optimizations

### Memory Efficiency
- Sparse matrix operations using scipy.sparse
- Efficient metadata processing with batch loading
- Memory-conscious model training

### Computational Speed
- Parallel processing where applicable
- Optimized matrix operations
- Early stopping for sparse scenarios

### Robustness
- Multiple fallback layers for failed predictions
- Graceful handling of missing data
- Adaptive parameters based on data characteristics

## 🎛️ Configuration

### Model Parameters
```python
RobustSparseRecommendationSystem(
    n_factors=150,        # Number of latent factors
    alpha=60              # Confidence parameter for implicit feedback
)
```

### Ensemble Weights
The system automatically adjusts weights based on data characteristics, but default weights are:
- ALS Model: 0.25
- BPR Model: 0.25  
- SVD Model: 0.20
- Co-occurrence: 0.15
- Content-based: 0.10
- Fallback: 0.05

## 📋 Data Format

### Required Files
- **train.csv**: User-item interactions (`user_id`, `item_id`, optional `timestamp`)
- **test.csv**: Test users requiring recommendations (`user_id`)
- **item_meta.csv**: Item metadata (optional but recommended)

### Expected Columns
**Training Data:**
- `user_id`: Unique user identifier
- `item_id`: Unique item identifier  
- `timestamp`: Interaction timestamp (optional, in milliseconds)

**Item Metadata:**
- `item_id`: Unique item identifier
- `main_category`: Item category
- `price`: Item price
- `average_rating`: Item quality rating

## 🚨 Troubleshooting

### Common Issues
1. **Memory errors**: Reduce `n_factors` or process metadata in smaller batches
2. **No recommendations**: System automatically falls back to popularity-based recommendations
3. **Poor performance**: Ensure sufficient training data and consider metadata integration

---

**Note**: This system is specifically optimized for extremely sparse datasets (>99% sparsity) and includes extensive fallback mechanisms to ensure robust performance in challenging scenarios.