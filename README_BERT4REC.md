# BERT4Rec Implementation for Sequential Recommendation

This repository contains two comprehensive BERT4Rec implementations for sequential recommendation systems:

1. **`run_bert4rec.py`** - Basic BERT4Rec with hybrid features
2. **`run_bert4rec_advanced.py`** - Advanced BERT4Rec with temporal modeling and contrastive learning

## 🌟 Features

### Data Preprocessing & Quality
- **Missing Value Handling**: Robust imputation strategies for timestamps and metadata
- **Duplicate Detection**: Advanced duplicate removal with temporal considerations
- **Outlier Detection**: Bot detection and extreme interaction filtering
- **Data Consistency**: Cross-validation between interaction and metadata
- **Temporal Feature Engineering**: Rich time-based features extraction

### Model Architecture

#### Basic BERT4Rec (`run_bert4rec.py`)
- Standard BERT encoder with masked language modeling
- Item and position embeddings
- Hybrid collaborative filtering components
- Content-based features from item metadata
- Multi-task learning objectives

#### Advanced BERT4Rec (`run_bert4rec_advanced.py`)
- **Temporal Modeling**: 
  - Cyclic time features (hour, day of week)
  - Time gap embeddings between interactions
  - Temporal attention mechanisms
- **Contrastive Learning**: Self-supervised representation learning
- **Multi-task Learning**: MLM + Next Item Prediction + Contrastive Loss
- **Enhanced Hybrid Features**:
  - User embeddings
  - Category embeddings
  - Content fusion layers
- **Advanced Training**:
  - Cosine annealing with warm restarts
  - Label smoothing
  - Gradient clipping

### Key Technical Innovations

#### 1. Data Quality Processing
```python
# Advanced duplicate handling with temporal constraints
train_df['time_diff'] = train_df.groupby(['user_id', 'item_id'])['timestamp'].diff()
close_duplicates = (train_df['time_diff'] < pd.Timedelta(hours=1))
train_df = train_df[~close_duplicates]
```

#### 2. Temporal Feature Engineering
```python
# Cyclic encoding for temporal patterns
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
```

#### 3. Contrastive Learning
```python
class ContrastiveLoss(nn.Module):
    def forward(self, anchor, positive, negatives):
        # InfoNCE loss for better representation learning
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        return F.cross_entropy(logits, labels)
```

#### 4. Multi-task Architecture
```python
# Multiple prediction heads for different objectives
self.mlm_head = nn.Linear(hidden_size, vocab_size)           # Masked LM
self.next_item_head = nn.Linear(hidden_size, vocab_size)     # Next item prediction
self.contrastive_head = nn.Linear(hidden_size, hidden_size)  # Contrastive learning
```

## 📊 Model Configurations

### Basic BERT4Rec
```python
config = {
    'vocab_size': 50000,      # Item vocabulary size
    'hidden_size': 128,       # Hidden dimension
    'num_layers': 2,          # BERT layers
    'num_attention_heads': 4, # Attention heads
    'max_seq_len': 50,        # Sequence length
    'dropout_prob': 0.1,      # Dropout rate
    'learning_rate': 1e-4,    # Learning rate
    'batch_size': 32,         # Batch size
    'num_epochs': 5           # Training epochs
}
```

### Advanced BERT4Rec
```python
config = {
    'vocab_size': 50000,
    'hidden_size': 256,           # Larger hidden dimension
    'num_layers': 4,              # More BERT layers
    'num_attention_heads': 8,     # More attention heads
    'max_seq_len': 50,
    'dropout_prob': 0.1,
    'learning_rate': 1e-4,
    'batch_size': 16,             # Smaller batch for memory
    'num_epochs': 10,             # More epochs
    'contrastive_temperature': 0.1,  # Contrastive learning
    'mlm_weight': 1.0,            # Loss weights
    'contrastive_weight': 0.1
}
```

## 🚀 Usage

### Installation
```bash
pip install -r requirements_bert4rec.txt
```

### Basic BERT4Rec
```bash
python run_bert4rec.py
```

### Advanced BERT4Rec
```bash
python run_bert4rec_advanced.py
```

## 📁 Data Format

### Expected Input Files
- `train.csv`: Training interactions (item_id, user_id, timestamp)
- `test.csv`: Test interactions (item_id, user_id, timestamp)  
- `item_meta.csv`: Item metadata (item_id, main_category, average_rating, price, etc.)
- `sample_submission.csv`: Submission format (ID, user_id, item_id)

### Output Files
- `bert4rec_submission.csv` / `advanced_bert4rec_submission.csv`: Predictions
- `bert4rec_model.pth` / `advanced_bert4rec_model.pth`: Trained models
- `bert4rec_preprocessor.pkl`: Data preprocessor
- `bert4rec_config.json`: Model configuration

## 🔧 Advanced Features Explained

### 1. Temporal Modeling
The advanced version captures temporal patterns in user behavior:
- **Time of day effects**: Shopping patterns vary by hour
- **Day of week patterns**: Weekend vs weekday behavior
- **Interaction intervals**: Time gaps between purchases
- **Seasonal trends**: Monthly/seasonal preferences

### 2. Hybrid Architecture
Combines multiple recommendation approaches:
- **Collaborative Filtering**: User-item interaction patterns
- **Content-Based**: Item metadata and categories
- **Sequential**: Temporal ordering of interactions
- **Deep Learning**: Neural representation learning

### 3. Loss Functions
Multiple training objectives:
- **Masked Language Modeling**: BERT-style pretraining
- **Next Item Prediction**: Sequential recommendation
- **Contrastive Learning**: Better representation quality
- **Multi-task Fusion**: Balanced combination

### 4. Cold Start Handling
Strategies for new users/items:
- **Popular Item Fallback**: Most frequently interacted items
- **Content-Based Similarity**: Metadata-driven recommendations
- **User Embedding**: Learned user representations
- **Category-Based**: Similar category recommendations

## 📈 Performance Optimization

### Memory Efficiency
- Gradient accumulation for large batches
- Mixed precision training (optional)
- Efficient data loading with PyTorch DataLoader
- Sparse matrix operations for user-item interactions

### Training Techniques
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Gradient Clipping**: Prevents exploding gradients
- **Label Smoothing**: Regularization for better generalization
- **Early Stopping**: Prevents overfitting

### Inference Optimization
- Batch prediction for multiple users
- Caching of item embeddings
- Top-k efficient selection
- Temperature scaling for prediction confidence

## 🎯 Hyperparameter Tuning

### Key Parameters to Tune
1. **Model Architecture**:
   - `hidden_size`: 128, 256, 512
   - `num_layers`: 2, 4, 6
   - `num_attention_heads`: 4, 8, 12

2. **Training**:
   - `learning_rate`: 1e-5 to 1e-3
   - `batch_size`: 8, 16, 32, 64
   - `dropout_prob`: 0.1, 0.2, 0.3

3. **Loss Weights**:
   - `mlm_weight`: 0.5 to 2.0
   - `contrastive_weight`: 0.05 to 0.5

4. **Sequence**:
   - `max_seq_len`: 20, 50, 100
   - `mask_prob`: 0.10 to 0.20

## 🔍 Evaluation Metrics

The models are evaluated on standard recommendation metrics:
- **Recall@10**: Percentage of relevant items in top-10
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **Hit Rate@10**: Fraction of users with at least one relevant item
- **MRR**: Mean Reciprocal Rank of the first relevant item

## 🛠️ Customization

### Adding New Features
1. **Temporal Features**: Extend `extract_temporal_features()`
2. **Content Features**: Modify `create_enhanced_content_features()`
3. **Loss Functions**: Add new objectives to the trainer
4. **Architectures**: Implement new model components

### Adapting to New Domains
1. Update vocabulary size and special tokens
2. Modify content feature extraction for domain-specific metadata
3. Adjust sequence length based on user behavior patterns
4. Fine-tune hyperparameters for domain characteristics

## 📚 References

1. Sun, F., et al. "BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer." CIKM 2019.
2. Kang, W., et al. "Self-attentive sequential recommendation." ICDM 2018.
3. He, R., et al. "Temporal collaborative filtering with bayesian probabilistic tensor factorization." SDM 2010.

## 🤝 Contributing

Feel free to contribute by:
- Adding new architectures or features
- Improving data preprocessing
- Optimizing training procedures
- Adding evaluation metrics
- Fixing bugs or improving documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 