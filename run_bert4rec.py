import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel, BertTokenizer
import pickle
import warnings
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class DataPreprocessor:
    """Handle missing values, timestamps, duplicates, and data cleaning"""
    
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.stats = {}
        
    def clean_and_prepare_data(self, train_df, test_df, item_meta_df):
        """Comprehensive data cleaning and preparation"""
        print("=== Data Cleaning and Preparation ===")
        
        # 1. Handle missing values
        print("1. Handling missing values...")
        train_df = train_df.dropna(subset=['user_id', 'item_id'])
        test_df = test_df.dropna(subset=['user_id', 'item_id'])
        
        # Fill missing timestamps with a default value
        train_df['timestamp'] = train_df['timestamp'].fillna(0)
        test_df['timestamp'] = test_df['timestamp'].fillna(0)
        
        # 2. Handle duplicates (keep the latest interaction)
        print("2. Handling duplicates...")
        print(f"Train duplicates before: {train_df.duplicated(subset=['user_id', 'item_id']).sum()}")
        train_df = train_df.sort_values('timestamp').drop_duplicates(
            subset=['user_id', 'item_id'], keep='last'
        )
        print(f"Train size after dedup: {len(train_df)}")
        
        # 3. Convert timestamps to proper format
        print("3. Processing timestamps...")
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], unit='ms', errors='coerce')
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], unit='ms', errors='coerce')
        
        # Handle any remaining invalid timestamps
        train_df['timestamp'] = train_df['timestamp'].fillna(pd.Timestamp.now())
        test_df['timestamp'] = test_df['timestamp'].fillna(pd.Timestamp.now())
        
        # 4. Filter users and items with minimum interactions
        print("4. Filtering by minimum interactions...")
        user_counts = train_df['user_id'].value_counts()
        item_counts = train_df['item_id'].value_counts()
        
        # Keep users with at least 5 interactions
        min_user_interactions = 5
        valid_users = user_counts[user_counts >= min_user_interactions].index
        train_df = train_df[train_df['user_id'].isin(valid_users)]
        
        # Keep items with at least 3 interactions
        min_item_interactions = 3
        valid_items = item_counts[item_counts >= min_item_interactions].index
        train_df = train_df[train_df['item_id'].isin(valid_items)]
        
        print(f"Final train size: {len(train_df)}")
        print(f"Unique users: {train_df['user_id'].nunique()}")
        print(f"Unique items: {train_df['item_id'].nunique()}")
        
        # 5. Encode users and items
        print("5. Encoding users and items...")
        all_users = list(set(train_df['user_id'].unique()) | set(test_df['user_id'].unique()))
        all_items = train_df['item_id'].unique()
        
        # Convert to string to handle mixed types
        all_users = [str(u) for u in all_users]
        all_items = [str(i) for i in all_items]
        train_df['user_id'] = train_df['user_id'].astype(str)
        train_df['item_id'] = train_df['item_id'].astype(str)
        test_df['user_id'] = test_df['user_id'].astype(str)
        test_df['item_id'] = test_df['item_id'].astype(str)
        
        # Add special tokens
        special_tokens = ['[PAD]', '[MASK]', '[UNK]']
        all_items_with_special = list(special_tokens) + list(all_items)
        
        self.user_encoder.fit(all_users)
        self.item_encoder.fit(all_items_with_special)
        
        train_df['user_encoded'] = self.user_encoder.transform(train_df['user_id'])
        train_df['item_encoded'] = self.item_encoder.transform(train_df['item_id'])
        
        # Handle test users that might not be in train
        test_user_mask = test_df['user_id'].isin(all_users)
        test_df = test_df[test_user_mask]
        if len(test_df) > 0:
            test_df['user_encoded'] = self.user_encoder.transform(test_df['user_id'])
        
        # 6. Store statistics
        self.stats = {
            'n_users': len(all_users),
            'n_items': len(all_items),
            'n_items_with_special': len(all_items_with_special),
            'pad_token': self.item_encoder.transform(['[PAD]'])[0],
            'mask_token': self.item_encoder.transform(['[MASK]'])[0],
            'unk_token': self.item_encoder.transform(['[UNK]'])[0],
            'vocab_size': len(all_items_with_special),
            'avg_sequence_length': train_df.groupby('user_id').size().mean()
        }
        
        print(f"Vocabulary size: {self.stats['vocab_size']}")
        print(f"PAD token: {self.stats['pad_token']}")
        print(f"MASK token: {self.stats['mask_token']}")
        print(f"UNK token: {self.stats['unk_token']}")
        
        return train_df, test_df, item_meta_df
    
    def create_sequences(self, train_df, max_seq_len=50):
        """Create user interaction sequences sorted by timestamp"""
        print("Creating user interaction sequences...")
        
        user_sequences = {}
        for user_id in tqdm(train_df['user_encoded'].unique()):
            user_data = train_df[train_df['user_encoded'] == user_id].sort_values('timestamp')
            sequence = user_data['item_encoded'].tolist()
            user_sequences[user_id] = sequence
        
        # Truncate or pad sequences
        processed_sequences = {}
        for user_id, seq in user_sequences.items():
            if len(seq) > max_seq_len:
                # Take the most recent interactions
                seq = seq[-max_seq_len:]
            processed_sequences[user_id] = seq
            
        return processed_sequences

class BERT4RecDataset(Dataset):
    """Dataset for BERT4Rec with masked language modeling"""
    
    def __init__(self, user_sequences, stats, max_seq_len=50, mask_prob=0.15):
        self.user_sequences = user_sequences
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.pad_token = stats['pad_token']
        self.mask_token = stats['mask_token']
        self.vocab_size = stats['vocab_size']
        
        # Prepare data
        self.data = []
        for user_id, sequence in user_sequences.items():
            self.data.append((user_id, sequence))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_id, sequence = self.data[idx]
        
        # Pad sequence
        if len(sequence) < self.max_seq_len:
            sequence = sequence + [self.pad_token] * (self.max_seq_len - len(sequence))
        else:
            sequence = sequence[:self.max_seq_len]
        
        # Create masks for MLM
        input_ids = sequence.copy()
        labels = [-100] * self.max_seq_len  # -100 is ignored in loss computation
        
        for i in range(len(sequence)):
            if sequence[i] != self.pad_token and random.random() < self.mask_prob:
                labels[i] = sequence[i]  # Store original token
                
                # 80% of time replace with [MASK]
                if random.random() < 0.8:
                    input_ids[i] = self.mask_token
                # 10% of time replace with random token
                elif random.random() < 0.5:
                    input_ids[i] = random.randint(3, self.vocab_size - 1)  # Skip special tokens
                # 10% of time keep original
        
        return {
            'user_id': user_id,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor([1 if x != self.pad_token else 0 for x in input_ids], dtype=torch.long)
        }

class BERT4Rec(nn.Module):
    """BERT4Rec model with hybrid components"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Item embedding
        self.item_embedding = nn.Embedding(
            config['vocab_size'], 
            config['hidden_size'], 
            padding_idx=config['pad_token']
        )
        
        # Position embedding
        self.position_embedding = nn.Embedding(config['max_seq_len'], config['hidden_size'])
        
        # BERT encoder
        bert_config = BertConfig(
            vocab_size=config['vocab_size'],
            hidden_size=config['hidden_size'],
            num_hidden_layers=config['num_layers'],
            num_attention_heads=config['num_attention_heads'],
            intermediate_size=config['intermediate_size'],
            hidden_dropout_prob=config['dropout_prob'],
            attention_probs_dropout_prob=config['dropout_prob'],
            max_position_embeddings=config['max_seq_len'],
            pad_token_id=config['pad_token']
        )
        
        self.bert = BertModel(bert_config)
        
        # Prediction head
        self.prediction_head = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        # Hybrid components
        if config['use_hybrid']:
            # User embedding for collaborative filtering component
            self.user_embedding = nn.Embedding(config['n_users'], config['hidden_size'])
            
            # Content-based features (if item metadata available)
            self.content_projection = nn.Linear(config['content_dim'], config['hidden_size'])
            
            # Fusion layer - adjust based on whether content is available
            self.fusion = nn.Linear(config['hidden_size'] * 2, config['hidden_size'])  # sequence + user only
        
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.layer_norm = nn.LayerNorm(config['hidden_size'])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask, user_ids=None, content_features=None):
        batch_size, seq_len = input_ids.size()
        
        # Item embeddings
        item_embeds = self.item_embedding(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combined embeddings
        embeddings = item_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # BERT encoding
        encoder_outputs = self.bert.encoder(
            embeddings,
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
        )
        sequence_output = encoder_outputs.last_hidden_state
        
        # Hybrid components
        if self.config['use_hybrid'] and user_ids is not None:
            # User embeddings
            user_embeds = self.user_embedding(user_ids).unsqueeze(1).expand(-1, seq_len, -1)
            
            # For now, just use sequence + user (content features can be added later)
            fused_features = torch.cat([sequence_output, user_embeds], dim=-1)
            sequence_output = self.fusion(fused_features)
            sequence_output = torch.relu(sequence_output)
        
        # Prediction
        prediction_scores = self.prediction_head(sequence_output)
        
        return prediction_scores

class BERT4RecTrainer:
    """Training and evaluation logic for BERT4Rec"""
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        self.training_history = {'loss': [], 'accuracy': []}
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            user_ids = batch['user_id'].to(self.device) if self.config['use_hybrid'] else None
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, user_ids)
            
            # Calculate loss
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate accuracy for non-masked tokens
            mask = labels != -100
            if mask.sum() > 0:
                predictions = outputs.argmax(dim=-1)
                accuracy = (predictions == labels)[mask].float().mean()
                
                total_accuracy += accuracy.item() * mask.sum().item()
                total_samples += mask.sum().item()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
        
        return avg_loss, avg_accuracy
    
    def train(self, train_dataloader, num_epochs):
        print(f"Training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_dataloader)
            
            self.training_history['loss'].append(train_loss)
            self.training_history['accuracy'].append(train_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    
    def predict_next_items(self, user_sequence, top_k=10, temperature=1.0):
        """Predict next items for a user sequence"""
        self.model.eval()
        
        with torch.no_grad():
            # Prepare input
            input_ids = torch.tensor([user_sequence], dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(
                [[1 if x != self.config['pad_token'] else 0 for x in user_sequence]], 
                dtype=torch.long
            ).to(self.device)
            
            # Get predictions
            outputs = self.model(input_ids, attention_mask)
            
            # Use the last non-padded position
            last_pos = len([x for x in user_sequence if x != self.config['pad_token']]) - 1
            logits = outputs[0, last_pos] / temperature
            
            # Apply softmax
            probs = torch.softmax(logits, dim=-1)
            
            # Get top-k items (excluding special tokens)
            top_probs, top_indices = torch.topk(probs[3:], top_k)  # Skip special tokens
            top_indices = top_indices + 3  # Add offset back
            
            return top_indices.cpu().numpy(), top_probs.cpu().numpy()

def create_hybrid_features(item_meta_df, item_encoder):
    """Create content-based features from item metadata"""
    print("Creating hybrid features from item metadata...")
    
    # Handle missing values in metadata
    item_meta_df = item_meta_df.fillna('')
    
    # Convert item_id to string to match encoding
    item_meta_df['item_id'] = item_meta_df['item_id'].astype(str)
    
    # Category encoding
    category_encoder = LabelEncoder()
    valid_categories = item_meta_df['main_category'].replace('', 'unknown')
    category_features = category_encoder.fit_transform(valid_categories)
    
    # Rating features
    rating_features = item_meta_df['average_rating'].fillna(0).values
    
    # Price features (extract numeric values)
    price_features = []
    for price in item_meta_df['price']:
        try:
            if pd.isna(price) or price == '':
                price_features.append(0.0)
            else:
                # Extract numeric value from price string
                import re
                numbers = re.findall(r'[\d.]+', str(price))
                if numbers:
                    price_features.append(float(numbers[0]))
                else:
                    price_features.append(0.0)
        except:
            price_features.append(0.0)
    
    price_features = np.array(price_features)
    
    # Combine features
    content_features = np.column_stack([
        category_features,
        rating_features,
        price_features / max(price_features.max(), 1)  # Normalize prices
    ])
    
    # Create mapping from encoded item to features
    item_to_features = {}
    for idx, row in item_meta_df.iterrows():
        try:
            item_encoded = item_encoder.transform([row['item_id']])[0]
            item_to_features[item_encoded] = content_features[idx]
        except:
            continue
    
    return item_to_features, content_features.shape[1]

def main():
    print("=== BERT4Rec Recommender System ===")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    item_meta_df = pd.read_csv('item_meta.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Item metadata shape: {item_meta_df.shape}")
    
    # Data preprocessing
    preprocessor = DataPreprocessor()
    train_df, test_df, item_meta_df = preprocessor.clean_and_prepare_data(
        train_df, test_df, item_meta_df
    )
    
    # Create sequences
    user_sequences = preprocessor.create_sequences(train_df, max_seq_len=50)
    
    # Create hybrid features
    item_to_features, content_dim = create_hybrid_features(item_meta_df, preprocessor.item_encoder)
    
    # Model configuration
    config = {
        'vocab_size': int(preprocessor.stats['vocab_size']),
        'hidden_size': 128,
        'num_layers': 2,
        'num_attention_heads': 4,
        'intermediate_size': 256,
        'max_seq_len': 50,
        'dropout_prob': 0.1,
        'pad_token': int(preprocessor.stats['pad_token']),
        'mask_token': int(preprocessor.stats['mask_token']),
        'n_users': int(preprocessor.stats['n_users']),
        'content_dim': int(content_dim),
        'use_hybrid': True,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'label_smoothing': 0.1,
        'batch_size': 32,
        'num_epochs': 5
    }
    
    print(f"Model configuration: {config}")
    
    # Create dataset and dataloader
    dataset = BERT4RecDataset(user_sequences, preprocessor.stats, max_seq_len=config['max_seq_len'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    
    # Initialize model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BERT4Rec(config)
    trainer = BERT4RecTrainer(model, config, device)
    
    # Train the model
    trainer.train(dataloader, config['num_epochs'])
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    
    # Load sample submission to get test users
    try:
        sample_submission = pd.read_csv('sample_submission.csv')
        sample_submission['user_id'] = sample_submission['user_id'].astype(str)
        test_users = sample_submission['user_id'].unique()
    except:
        test_users = test_df['user_id'].unique()[:100]  # Fallback
    
    recommendations = {}
    
    for user_id in tqdm(test_users, desc="Generating recommendations"):
        try:
            if user_id in preprocessor.user_encoder.classes_:
                user_encoded = preprocessor.user_encoder.transform([user_id])[0]
                
                if user_encoded in user_sequences:
                    # Use existing sequence
                    sequence = user_sequences[user_encoded].copy()
                    
                    # Pad sequence if necessary
                    if len(sequence) < config['max_seq_len']:
                        sequence = sequence + [config['pad_token']] * (config['max_seq_len'] - len(sequence))
                    else:
                        sequence = sequence[-config['max_seq_len']:]
                    
                    # Get predictions
                    item_indices, scores = trainer.predict_next_items(sequence, top_k=20)
                    
                    # Convert back to original item IDs
                    recommended_items = []
                    for idx in item_indices:
                        try:
                            item_id = preprocessor.item_encoder.inverse_transform([idx])[0]
                            if item_id not in ['[PAD]', '[MASK]', '[UNK]']:
                                recommended_items.append(item_id)
                            if len(recommended_items) == 10:
                                break
                        except:
                            continue
                    
                    recommendations[user_id] = recommended_items[:10]
                else:
                    # Cold start: use popular items
                    popular_items = train_df['item_id'].value_counts().head(10).index.tolist()
                    recommendations[user_id] = popular_items
            else:
                # Cold start: use popular items
                popular_items = train_df['item_id'].value_counts().head(10).index.tolist()
                recommendations[user_id] = popular_items
        except Exception as e:
            print(f"Error for user {user_id}: {e}")
            popular_items = train_df['item_id'].value_counts().head(10).index.tolist()
            recommendations[user_id] = popular_items
    
    # Create submission
    print("Creating submission file...")
    submission_data = []
    
    try:
        sample_submission = pd.read_csv('sample_submission.csv')
        for idx, row in sample_submission.iterrows():
            user_id = row['user_id']
            recs = recommendations.get(user_id, train_df['item_id'].value_counts().head(10).index.tolist())
            
            # Ensure we have 10 recommendations
            if len(recs) < 10:
                popular_items = train_df['item_id'].value_counts().head(20).index.tolist()
                for item in popular_items:
                    if item not in recs:
                        recs.append(item)
                    if len(recs) == 10:
                        break
            
            submission_data.append({
                'ID': row['ID'],
                'user_id': user_id,
                'item_id': ','.join(map(str, recs[:10]))
            })
    except:
        # Fallback submission format
        for i, user_id in enumerate(test_users):
            recs = recommendations.get(user_id, train_df['item_id'].value_counts().head(10).index.tolist())
            submission_data.append({
                'ID': i,
                'user_id': user_id,
                'item_id': ','.join(map(str, recs[:10]))
            })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('bert4rec_submission.csv', index=False)
    
    print(f"Submission shape: {submission_df.shape}")
    print("Submission saved to 'bert4rec_submission.csv'")
    
    # Save model and preprocessor
    print("Saving model and preprocessor...")
    torch.save(model.state_dict(), 'bert4rec_model.pth')
    
    with open('bert4rec_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    with open('bert4rec_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Training completed!")
    print(f"Final training loss: {trainer.training_history['loss'][-1]:.4f}")
    print(f"Final training accuracy: {trainer.training_history['accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main() 