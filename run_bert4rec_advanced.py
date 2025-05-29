import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel
import pickle
import warnings
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime, timedelta
import random
from tqdm import tqdm
import json
import math

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class AdvancedDataPreprocessor:
    """Enhanced data preprocessor with temporal features and advanced cleaning"""
    
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.stats = {}
        self.temporal_stats = {}
        
    def extract_temporal_features(self, df):
        """Extract rich temporal features from timestamps"""
        print("Extracting temporal features...")
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        
        # Extract temporal components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time since first interaction
        min_time = df['timestamp'].min()
        df['days_since_start'] = (df['timestamp'] - min_time).dt.days
        
        # Interaction frequency patterns
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def handle_data_quality_issues(self, train_df, test_df, item_meta_df):
        """Comprehensive data quality handling"""
        print("=== Advanced Data Quality Processing ===")
        
        # 1. Outlier detection and removal
        print("1. Detecting and handling outliers...")
        
        # Remove users with excessive interactions (potential bots)
        user_interaction_counts = train_df['user_id'].value_counts()
        outlier_threshold = user_interaction_counts.quantile(0.99)
        valid_users = user_interaction_counts[user_interaction_counts <= outlier_threshold].index
        train_df = train_df[train_df['user_id'].isin(valid_users)]
        
        # 2. Handle missing and invalid data
        print("2. Handling missing and invalid data...")
        
        # Remove rows with missing critical fields
        train_df = train_df.dropna(subset=['user_id', 'item_id'])
        test_df = test_df.dropna(subset=['user_id', 'item_id'])
        
        # Fill missing timestamps
        train_df['timestamp'] = train_df['timestamp'].fillna(train_df['timestamp'].median())
        test_df['timestamp'] = test_df['timestamp'].fillna(test_df['timestamp'].median())
        
        # 3. Advanced duplicate handling
        print("3. Advanced duplicate handling...")
        
        # Convert timestamps to datetime first
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], unit='ms', errors='coerce')
        
        # Sort by timestamp and keep only the latest interaction
        train_df = train_df.sort_values(['user_id', 'item_id', 'timestamp'])
        
        # For duplicates, calculate time differences
        train_df['time_diff'] = train_df.groupby(['user_id', 'item_id'])['timestamp'].diff()
        
        # Remove duplicates that are too close in time (< 1 hour)
        close_duplicates = (train_df['time_diff'] < pd.Timedelta(hours=1)) & train_df['time_diff'].notna()
        train_df = train_df[~close_duplicates]
        
        # Remove time_diff column
        train_df = train_df.drop('time_diff', axis=1)
        
        # 4. Data consistency checks
        print("4. Data consistency checks...")
        
        # Ensure item_ids exist in metadata
        valid_items_in_meta = set(item_meta_df['item_id'].unique())
        train_df = train_df[train_df['item_id'].isin(valid_items_in_meta)]
        
        print(f"After quality processing - Train: {len(train_df)}, Test: {len(test_df)}")
        
        return train_df, test_df, item_meta_df
    
    def create_advanced_sequences(self, train_df, max_seq_len=50):
        """Create sequences with advanced temporal and contextual information"""
        print("Creating advanced user sequences...")
        
        # Add temporal features
        train_df = self.extract_temporal_features(train_df)
        
        user_sequences = {}
        user_temporal_features = {}
        user_contexts = {}
        
        for user_id in tqdm(train_df['user_encoded'].unique()):
            user_data = train_df[train_df['user_encoded'] == user_id].sort_values('timestamp')
            
            # Item sequence
            item_sequence = user_data['item_encoded'].tolist()
            
            # Temporal features sequence
            temporal_features = user_data[['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                                         'is_weekend', 'days_since_start']].values.tolist()
            
            # Context features (time gaps between interactions)
            timestamps = user_data['timestamp'].values
            time_gaps = []
            for i in range(len(timestamps)):
                if i == 0:
                    time_gaps.append(0.0)
                else:
                    # Convert numpy timedelta to pandas timedelta and then to hours
                    gap_td = pd.Timedelta(timestamps[i] - timestamps[i-1])
                    gap_hours = gap_td.total_seconds() / 3600
                    time_gaps.append(min(gap_hours, 168))  # Cap at 1 week
            
            # Truncate sequences if too long
            if len(item_sequence) > max_seq_len:
                item_sequence = item_sequence[-max_seq_len:]
                temporal_features = temporal_features[-max_seq_len:]
                time_gaps = time_gaps[-max_seq_len:]
            
            user_sequences[user_id] = item_sequence
            user_temporal_features[user_id] = temporal_features
            user_contexts[user_id] = time_gaps
        
        return user_sequences, user_temporal_features, user_contexts
    
    def clean_and_prepare_data(self, train_df, test_df, item_meta_df):
        """Main data preparation pipeline"""
        print("=== Advanced Data Cleaning and Preparation ===")
        
        # Quality processing
        train_df, test_df, item_meta_df = self.handle_data_quality_issues(
            train_df, test_df, item_meta_df
        )
        
        # Filter by minimum interactions
        print("Filtering by minimum interactions...")
        user_counts = train_df['user_id'].value_counts()
        item_counts = train_df['item_id'].value_counts()
        
        # More balanced filtering
        min_user_interactions = max(3, int(user_counts.quantile(0.1)))
        min_item_interactions = max(2, int(item_counts.quantile(0.05)))
        
        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_items = item_counts[item_counts >= min_item_interactions].index
        
        train_df = train_df[train_df['user_id'].isin(valid_users)]
        train_df = train_df[train_df['item_id'].isin(valid_items)]
        
        print(f"Final train size: {len(train_df)}")
        print(f"Unique users: {train_df['user_id'].nunique()}")
        print(f"Unique items: {train_df['item_id'].nunique()}")
        
        # Encoding
        print("Encoding users and items...")
        all_users = list(set(train_df['user_id'].unique()) | set(test_df['user_id'].unique()))
        all_items = train_df['item_id'].unique()
        
        # Convert to string to handle mixed types
        all_users = [str(u) for u in all_users]
        all_items = [str(i) for i in all_items]
        train_df['user_id'] = train_df['user_id'].astype(str)
        train_df['item_id'] = train_df['item_id'].astype(str)
        test_df['user_id'] = test_df['user_id'].astype(str)
        test_df['item_id'] = test_df['item_id'].astype(str)
        
        # Special tokens
        special_tokens = ['[PAD]', '[MASK]', '[UNK]', '[CLS]', '[SEP]']
        all_items_with_special = list(special_tokens) + list(all_items)
        
        self.user_encoder.fit(all_users)
        self.item_encoder.fit(all_items_with_special)
        
        train_df['user_encoded'] = self.user_encoder.transform(train_df['user_id'])
        train_df['item_encoded'] = self.item_encoder.transform(train_df['item_id'])
        
        # Handle test users
        test_user_mask = test_df['user_id'].isin(all_users)
        test_df = test_df[test_user_mask]
        if len(test_df) > 0:
            test_df['user_encoded'] = self.user_encoder.transform(test_df['user_id'])
        
        # Statistics
        self.stats = {
            'n_users': len(all_users),
            'n_items': len(all_items),
            'vocab_size': len(all_items_with_special),
            'pad_token': self.item_encoder.transform(['[PAD]'])[0],
            'mask_token': self.item_encoder.transform(['[MASK]'])[0],
            'unk_token': self.item_encoder.transform(['[UNK]'])[0],
            'cls_token': self.item_encoder.transform(['[CLS]'])[0],
            'sep_token': self.item_encoder.transform(['[SEP]'])[0],
            'avg_sequence_length': train_df.groupby('user_id').size().mean()
        }
        
        return train_df, test_df, item_meta_df

class ContrastiveLoss(nn.Module):
    """Contrastive learning loss for better representation learning"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positive, negatives):
        """
        anchor: [batch_size, hidden_size]
        positive: [batch_size, hidden_size]  
        negatives: [batch_size, num_negatives, hidden_size]
        """
        batch_size = anchor.size(0)
        
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        # Positive similarities
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        
        # Negative similarities
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature
        
        # Contrastive loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        return F.cross_entropy(logits, labels)

class AdvancedBERT4RecDataset(Dataset):
    """Advanced dataset with temporal features and contrastive learning"""
    
    def __init__(self, user_sequences, user_temporal_features, user_contexts, 
                 stats, max_seq_len=50, mask_prob=0.15, num_negatives=5):
        self.user_sequences = user_sequences
        self.user_temporal_features = user_temporal_features
        self.user_contexts = user_contexts
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.num_negatives = num_negatives
        self.pad_token = stats['pad_token']
        self.mask_token = stats['mask_token']
        self.vocab_size = stats['vocab_size']
        
        # Prepare data
        self.data = []
        for user_id, sequence in user_sequences.items():
            if len(sequence) >= 3:  # Minimum sequence length
                self.data.append(user_id)
        
        # Create item frequency for negative sampling
        all_items = []
        for seq in user_sequences.values():
            all_items.extend(seq)
        self.item_freq = Counter(all_items)
        self.items = list(self.item_freq.keys())
        self.item_probs = np.array([self.item_freq[item] for item in self.items])
        self.item_probs = self.item_probs ** 0.75  # Sublinear sampling
        self.item_probs = self.item_probs / self.item_probs.sum()
    
    def __len__(self):
        return len(self.data)
    
    def sample_negatives(self, user_items, num_negatives):
        """Sample negative items not in user's history"""
        negatives = []
        max_attempts = num_negatives * 10
        attempts = 0
        
        while len(negatives) < num_negatives and attempts < max_attempts:
            item = np.random.choice(self.items, p=self.item_probs)
            if item not in user_items and item >= 5:  # Exclude special tokens
                negatives.append(item)
            attempts += 1
        
        # Fill remaining with random items if needed
        while len(negatives) < num_negatives:
            item = np.random.randint(5, self.vocab_size)
            if item not in user_items:
                negatives.append(item)
        
        return negatives[:num_negatives]
    
    def __getitem__(self, idx):
        user_id = self.data[idx]
        sequence = self.user_sequences[user_id].copy()
        temporal_features = self.user_temporal_features[user_id].copy()
        time_gaps = self.user_contexts[user_id].copy()
        
        # Pad sequences
        seq_len = len(sequence)
        if seq_len < self.max_seq_len:
            padding_len = self.max_seq_len - seq_len
            sequence.extend([self.pad_token] * padding_len)
            temporal_features.extend([[0.0] * 6] * padding_len)  # 6 temporal features
            time_gaps.extend([0.0] * padding_len)
        else:
            sequence = sequence[:self.max_seq_len]
            temporal_features = temporal_features[:self.max_seq_len]
            time_gaps = time_gaps[:self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Create MLM masks
        input_ids = sequence.copy()
        labels = [-100] * self.max_seq_len
        
        # Only mask non-padding tokens
        for i in range(seq_len):
            if sequence[i] != self.pad_token and random.random() < self.mask_prob:
                labels[i] = sequence[i]
                
                if random.random() < 0.8:
                    input_ids[i] = self.mask_token
                elif random.random() < 0.5:
                    input_ids[i] = random.randint(5, self.vocab_size - 1)
        
        # Sample negatives for contrastive learning
        user_items = set(self.user_sequences[user_id])
        negatives = self.sample_negatives(user_items, self.num_negatives)
        
        return {
            'user_id': user_id,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'temporal_features': torch.tensor(temporal_features, dtype=torch.float),
            'time_gaps': torch.tensor(time_gaps, dtype=torch.float),
            'attention_mask': torch.tensor([1 if i < seq_len else 0 for i in range(self.max_seq_len)], dtype=torch.long),
            'negatives': torch.tensor(negatives, dtype=torch.long)
        }

class AdvancedBERT4Rec(nn.Module):
    """Advanced BERT4Rec with temporal modeling and hybrid features"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config['hidden_size']
        
        # Core embeddings
        self.item_embedding = nn.Embedding(
            config['vocab_size'], hidden_size, padding_idx=config['pad_token']
        )
        self.position_embedding = nn.Embedding(config['max_seq_len'], hidden_size)
        
        # Temporal embeddings
        self.temporal_projection = nn.Linear(6, hidden_size // 4)  # 6 temporal features
        self.time_gap_embedding = nn.Embedding(169, hidden_size // 4)  # Up to 168 hours + 1
        
        # User and category embeddings for hybrid approach
        if config['use_hybrid']:
            self.user_embedding = nn.Embedding(config['n_users'], hidden_size)
            self.category_embedding = nn.Embedding(config['n_categories'], hidden_size // 2)
        
        # Enhanced BERT encoder
        bert_config = BertConfig(
            vocab_size=config['vocab_size'],
            hidden_size=hidden_size,
            num_hidden_layers=config['num_layers'],
            num_attention_heads=config['num_attention_heads'],
            intermediate_size=config['intermediate_size'],
            hidden_dropout_prob=config['dropout_prob'],
            attention_probs_dropout_prob=config['dropout_prob'],
            max_position_embeddings=config['max_seq_len'],
            pad_token_id=config['pad_token']
        )
        
        self.bert = BertModel(bert_config)
        
        # Multi-task heads
        self.mlm_head = nn.Linear(hidden_size, config['vocab_size'])
        self.next_item_head = nn.Linear(hidden_size, config['vocab_size'])
        
        # Contrastive learning head
        self.contrastive_head = nn.Linear(hidden_size, hidden_size)
        
        # Fusion layers - temporal features are hidden_size//4 + hidden_size//4 = hidden_size//2
        self.temporal_fusion = nn.Linear(hidden_size // 2, hidden_size)  # Output full hidden size
        if config['use_hybrid']:
            self.hybrid_fusion = nn.Linear(hidden_size * 2, hidden_size)
        
        # Regularization
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.layer_norm = nn.LayerNorm(hidden_size)
        
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
    
    def forward(self, input_ids, attention_mask, temporal_features, time_gaps, 
                user_ids=None, category_ids=None):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Core embeddings
        item_embeds = self.item_embedding(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Temporal embeddings
        temporal_embeds = self.temporal_projection(temporal_features)
        
        # Time gap embeddings (clip to max value)
        time_gaps_clipped = torch.clamp(time_gaps.long(), 0, 168)
        time_gap_embeds = self.time_gap_embedding(time_gaps_clipped)
        
        # Combine temporal features
        temporal_combined = torch.cat([temporal_embeds, time_gap_embeds], dim=-1)
        temporal_combined = self.temporal_fusion(temporal_combined)
        
        # Initial embeddings
        embeddings = item_embeds + position_embeds + temporal_combined
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # BERT encoding
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        encoder_outputs = self.bert.encoder(
            embeddings,
            attention_mask=extended_attention_mask
        )
        sequence_output = encoder_outputs.last_hidden_state
        
        # Hybrid fusion
        if self.config['use_hybrid'] and user_ids is not None:
            user_embeds = self.user_embedding(user_ids)
            user_embeds = user_embeds.unsqueeze(1).expand(-1, seq_len, -1)
            
            hybrid_output = torch.cat([sequence_output, user_embeds], dim=-1)
            sequence_output = self.hybrid_fusion(hybrid_output)
            sequence_output = torch.relu(sequence_output)
        
        # Multi-task outputs
        mlm_logits = self.mlm_head(sequence_output)
        next_item_logits = self.next_item_head(sequence_output)
        contrastive_features = self.contrastive_head(sequence_output)
        
        return {
            'mlm_logits': mlm_logits,
            'next_item_logits': next_item_logits,
            'contrastive_features': contrastive_features,
            'sequence_output': sequence_output
        }

class AdvancedBERT4RecTrainer:
    """Advanced trainer with multiple loss functions and techniques"""
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizers
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=max(1, config['num_epochs'] // 4)
        )
        
        # Loss functions
        self.mlm_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.contrastive_loss = ContrastiveLoss(temperature=config['contrastive_temperature'])
        
        # Loss weights
        self.mlm_weight = config.get('mlm_weight', 1.0)
        self.contrastive_weight = config.get('contrastive_weight', 0.1)
        
        self.training_history = {
            'total_loss': [], 'mlm_loss': [], 'contrastive_loss': [], 'accuracy': []
        }
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_mlm_loss = 0
        total_contrastive_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            temporal_features = batch['temporal_features'].to(self.device)
            time_gaps = batch['time_gaps'].to(self.device)
            user_ids = batch['user_id'].to(self.device) if self.config['use_hybrid'] else None
            negatives = batch['negatives'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids, attention_mask, temporal_features, time_gaps, user_ids
            )
            
            # MLM loss
            mlm_loss = self.mlm_loss(
                outputs['mlm_logits'].view(-1, outputs['mlm_logits'].size(-1)),
                labels.view(-1)
            )
            
            # Contrastive loss
            # Get sequence representations (mean of non-padded positions)
            mask = attention_mask.unsqueeze(-1).float()
            sequence_repr = (outputs['sequence_output'] * mask).sum(dim=1) / mask.sum(dim=1)
            
            # Sample positive and negative items for contrastive learning
            batch_size = input_ids.size(0)
            contrastive_loss = 0
            
            if batch_size > 1:
                # Use other sequences as negatives (simplified contrastive learning)
                anchor = sequence_repr
                positive = torch.roll(sequence_repr, 1, dims=0)  # Shift by 1
                neg_indices = torch.randperm(batch_size)[:min(5, batch_size-1)]
                negative = sequence_repr[neg_indices]
                
                # Compute contrastive loss
                pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / self.config['contrastive_temperature']
                neg_sim = torch.mm(anchor, negative.t()) / self.config['contrastive_temperature']
                
                logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
                targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                contrastive_loss = F.cross_entropy(logits, targets)
            
            # Combined loss
            total_loss_batch = (
                self.mlm_weight * mlm_loss + 
                self.contrastive_weight * contrastive_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate accuracy
            mask = labels != -100
            if mask.sum() > 0:
                predictions = outputs['mlm_logits'].argmax(dim=-1)
                accuracy = (predictions == labels)[mask].float().mean()
                total_accuracy += accuracy.item() * mask.sum().item()
                total_samples += mask.sum().item()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_mlm_loss += mlm_loss.item()
            total_contrastive_loss += contrastive_loss if isinstance(contrastive_loss, float) else contrastive_loss.item()
        
        # Update learning rate
        self.scheduler.step()
        
        # Return averages
        num_batches = len(dataloader)
        return {
            'total_loss': total_loss / num_batches,
            'mlm_loss': total_mlm_loss / num_batches,
            'contrastive_loss': total_contrastive_loss / num_batches,
            'accuracy': total_accuracy / total_samples if total_samples > 0 else 0,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def train(self, train_dataloader, num_epochs):
        print(f"Training advanced BERT4Rec for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            metrics = self.train_epoch(train_dataloader)
            
            # Store history
            for key, value in metrics.items():
                if key in self.training_history:
                    self.training_history[key].append(value)
            
            print(f"Loss: {metrics['total_loss']:.4f}, "
                  f"MLM: {metrics['mlm_loss']:.4f}, "
                  f"Contrastive: {metrics['contrastive_loss']:.4f}, "
                  f"Accuracy: {metrics['accuracy']:.4f}, "
                  f"LR: {metrics['lr']:.6f}")
    
    def predict_next_items(self, user_sequence, temporal_seq, time_gaps_seq, 
                          top_k=10, temperature=1.0):
        """Enhanced prediction with temporal features"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = torch.tensor([user_sequence], dtype=torch.long).to(self.device)
            temporal_features = torch.tensor([temporal_seq], dtype=torch.float).to(self.device)
            time_gaps = torch.tensor([time_gaps_seq], dtype=torch.float).to(self.device)
            attention_mask = torch.tensor(
                [[1 if x != self.config['pad_token'] else 0 for x in user_sequence]], 
                dtype=torch.long
            ).to(self.device)
            
            outputs = self.model(input_ids, attention_mask, temporal_features, time_gaps)
            
            # Use next_item_logits for prediction
            last_pos = len([x for x in user_sequence if x != self.config['pad_token']]) - 1
            logits = outputs['next_item_logits'][0, last_pos] / temperature
            
            # Apply softmax and get top-k
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs[5:], top_k)  # Skip special tokens
            top_indices = top_indices + 5
            
            return top_indices.cpu().numpy(), top_probs.cpu().numpy()

def create_enhanced_content_features(item_meta_df, item_encoder):
    """Create enhanced content features with better handling"""
    print("Creating enhanced content features...")
    
    # Handle missing values
    item_meta_df = item_meta_df.fillna('')
    
    # Convert item_id to string to match encoding
    item_meta_df['item_id'] = item_meta_df['item_id'].astype(str)
    
    # Category features
    category_encoder = LabelEncoder()
    categories = item_meta_df['main_category'].replace('', 'Unknown')
    category_features = category_encoder.fit_transform(categories)
    
    # Rating features (normalized)
    ratings = item_meta_df['average_rating'].fillna(item_meta_df['average_rating'].median())
    rating_features = (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-8)
    
    # Rating count features (log-scaled)
    rating_counts = item_meta_df['rating_number'].fillna(0)
    rating_count_features = np.log1p(rating_counts)
    rating_count_features = rating_count_features / rating_count_features.max()
    
    # Price features (robust extraction and normalization)
    price_features = []
    for price in item_meta_df['price']:
        try:
            if pd.isna(price) or price == '':
                price_features.append(0.0)
            else:
                import re
                # Extract all numbers and take the first one
                numbers = re.findall(r'[\d.]+', str(price).replace(',', ''))
                if numbers:
                    price_val = float(numbers[0])
                    price_features.append(price_val)
                else:
                    price_features.append(0.0)
        except:
            price_features.append(0.0)
    
    price_features = np.array(price_features)
    # Robust normalization (remove outliers)
    price_q99 = np.percentile(price_features[price_features > 0], 99)
    price_features = np.clip(price_features, 0, price_q99)
    price_features = price_features / (price_q99 + 1e-8)
    
    # Combine all features
    content_features = np.column_stack([
        category_features,
        rating_features,
        rating_count_features,
        price_features
    ])
    
    # Create item mapping
    item_to_features = {}
    category_mapping = {}
    
    for idx, row in item_meta_df.iterrows():
        try:
            item_encoded = item_encoder.transform([row['item_id']])[0]
            item_to_features[item_encoded] = content_features[idx]
            category_mapping[item_encoded] = category_features[idx]
        except:
            continue
    
    return item_to_features, content_features.shape[1], category_mapping, len(category_encoder.classes_)

def main():
    print("=== Advanced BERT4Rec Recommender System ===")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    item_meta_df = pd.read_csv('item_meta.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Item metadata shape: {item_meta_df.shape}")
    
    # Advanced preprocessing
    preprocessor = AdvancedDataPreprocessor()
    train_df, test_df, item_meta_df = preprocessor.clean_and_prepare_data(
        train_df, test_df, item_meta_df
    )
    
    # Create advanced sequences
    user_sequences, user_temporal_features, user_contexts = preprocessor.create_advanced_sequences(
        train_df, max_seq_len=50
    )
    
    # Enhanced content features
    item_features, content_dim, category_mapping, n_categories = create_enhanced_content_features(
        item_meta_df, preprocessor.item_encoder
    )
    
    # Advanced model configuration
    config = {
        'vocab_size': int(preprocessor.stats['vocab_size']),
        'hidden_size': 256,
        'num_layers': 4,
        'num_attention_heads': 8,
        'intermediate_size': 512,
        'max_seq_len': 50,
        'dropout_prob': 0.1,
        'pad_token': int(preprocessor.stats['pad_token']),
        'mask_token': int(preprocessor.stats['mask_token']),
        'n_users': int(preprocessor.stats['n_users']),
        'n_categories': int(n_categories),
        'content_dim': int(content_dim),
        'use_hybrid': True,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 16,  # Smaller batch for memory efficiency
        'num_epochs': 3,
        'contrastive_temperature': 0.1,
        'mlm_weight': 1.0,
        'contrastive_weight': 0.1
    }
    
    print(f"Model configuration: {json.dumps(config, indent=2)}")
    
    # Create dataset and dataloader
    dataset = AdvancedBERT4RecDataset(
        user_sequences, user_temporal_features, user_contexts,
        preprocessor.stats, max_seq_len=config['max_seq_len']
    )
    dataloader = DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=True, 
        num_workers=0, pin_memory=True
    )
    
    # Initialize model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AdvancedBERT4Rec(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = AdvancedBERT4RecTrainer(model, config, device)
    
    # Train the model
    trainer.train(dataloader, config['num_epochs'])
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    
    # Load test users
    try:
        sample_submission = pd.read_csv('sample_submission.csv')
        test_users = sample_submission['user_id'].unique()
    except:
        test_users = test_df['user_id'].unique()[:100] if len(test_df) > 0 else []
    
    recommendations = {}
    
    for user_id in tqdm(test_users, desc="Generating recommendations"):
        try:
            if user_id in preprocessor.user_encoder.classes_:
                user_encoded = preprocessor.user_encoder.transform([user_id])[0]
                
                if user_encoded in user_sequences:
                    sequence = user_sequences[user_encoded].copy()
                    temporal_seq = user_temporal_features[user_encoded].copy()
                    time_gaps_seq = user_contexts[user_encoded].copy()
                    
                    # Pad sequences
                    max_len = config['max_seq_len']
                    if len(sequence) < max_len:
                        pad_len = max_len - len(sequence)
                        sequence.extend([config['pad_token']] * pad_len)
                        temporal_seq.extend([[0.0] * 6] * pad_len)
                        time_gaps_seq.extend([0.0] * pad_len)
                    else:
                        sequence = sequence[-max_len:]
                        temporal_seq = temporal_seq[-max_len:]
                        time_gaps_seq = time_gaps_seq[-max_len:]
                    
                    # Get predictions
                    item_indices, scores = trainer.predict_next_items(
                        sequence, temporal_seq, time_gaps_seq, top_k=20
                    )
                    
                    # Convert to original item IDs
                    recommended_items = []
                    for idx in item_indices:
                        try:
                            item_id = preprocessor.item_encoder.inverse_transform([idx])[0]
                            if item_id not in ['[PAD]', '[MASK]', '[UNK]', '[CLS]', '[SEP]']:
                                recommended_items.append(item_id)
                            if len(recommended_items) == 10:
                                break
                        except:
                            continue
                    
                    recommendations[user_id] = recommended_items[:10]
                else:
                    # Cold start
                    popular_items = train_df['item_id'].value_counts().head(10).index.tolist()
                    recommendations[user_id] = popular_items
            else:
                # Cold start
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
            
            # Ensure 10 recommendations
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
        # Fallback
        for i, user_id in enumerate(test_users):
            recs = recommendations.get(user_id, train_df['item_id'].value_counts().head(10).index.tolist())
            submission_data.append({
                'ID': i,
                'user_id': user_id,
                'item_id': ','.join(map(str, recs[:10]))
            })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('advanced_bert4rec_submission.csv', index=False)
    
    print(f"Submission shape: {submission_df.shape}")
    print("Submission saved to 'advanced_bert4rec_submission.csv'")
    
    # Save everything
    print("Saving model and artifacts...")
    torch.save(model.state_dict(), 'advanced_bert4rec_model.pth')
    
    with open('advanced_bert4rec_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    with open('advanced_bert4rec_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Training completed!")
    print(f"Final training metrics:")
    for key, values in trainer.training_history.items():
        if values:
            print(f"  {key}: {values[-1]:.4f}")

if __name__ == "__main__":
    main() 