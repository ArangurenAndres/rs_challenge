import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.neural_network import MLPRegressor
import implicit
from collections import defaultdict, Counter
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import math

warnings.filterwarnings('ignore')

class LightGCNModel(nn.Module):
    """Simplified LightGCN implementation for graph-based collaborative filtering"""
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, adj_matrix, users, items):
        # Get initial embeddings
        user_embs = [self.user_embedding.weight]
        item_embs = [self.item_embedding.weight]
        
        # Graph convolution layers
        for _ in range(self.n_layers):
            # Propagate user embeddings to items
            new_item_emb = torch.sparse.mm(adj_matrix.t(), user_embs[-1])
            # Propagate item embeddings to users  
            new_user_emb = torch.sparse.mm(adj_matrix, item_embs[-1])
            
            user_embs.append(new_user_emb)
            item_embs.append(new_item_emb)
        
        # Average all layer embeddings
        final_user_emb = torch.mean(torch.stack(user_embs), dim=0)
        final_item_emb = torch.mean(torch.stack(item_embs), dim=0)
        
        # Get embeddings for specific users and items
        user_emb = final_user_emb[users]
        item_emb = final_item_emb[items]
        
        # Compute scores
        scores = torch.sum(user_emb * item_emb, dim=1)
        return scores

class NCFModel(nn.Module):
    """Neural Collaborative Filtering Model"""
    
    def __init__(self, n_users, n_items, embedding_dim=64, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        self.mlp_layers = nn.ModuleList()
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            self.mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        self.output_layer = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        
        # Pass through MLP
        for layer in self.mlp_layers:
            x = layer(x)
        
        output = self.output_layer(x)
        return output.squeeze()

class AdvancedRecommendationSystem:
    """Advanced multi-technique recommendation system"""
    
    def __init__(self, embedding_dim=64, n_factors=100):
        self.embedding_dim = embedding_dim
        self.n_factors = n_factors
        
        # Encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Models
        self.als_model = None
        self.svd_model = None
        self.nmf_model = None
        self.lightgcn_model = None
        self.ncf_model = None
        
        # Data structures
        self.interaction_matrix = None
        self.content_features = {}
        self.item_similarity = None
        self.popularity_scores = {}
        self.user_profiles = {}
        
        # Statistics
        self.stats = {}
        
    def build_content_features(self, item_meta_df):
        """Build content-based features from item metadata"""
        print("Building content features...")
        
        if item_meta_df is None or len(item_meta_df) == 0:
            print("No metadata available for content-based filtering")
            return
        
        # Process textual features
        text_features = []
        item_ids = []
        
        for _, row in item_meta_df.iterrows():
            item_id = row['item_id']
            item_ids.append(item_id)
            
            # Combine text fields
            text_parts = []
            feature_dict = {}
            
            for col in ['main_category', 'title', 'features', 'description', 'categories']:
                if col in item_meta_df.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]).lower())
                    feature_dict[col] = str(row[col]).lower()
            
            # Numeric features
            for col in ['average_rating', 'rating_number', 'price']:
                if col in item_meta_df.columns and pd.notna(row[col]):
                    feature_dict[col] = row[col]
            
            text_features.append(' '.join(text_parts))
            self.content_features[item_id] = feature_dict
        
        # Build TF-IDF similarity matrix
        if text_features:
            try:
                tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
                tfidf_matrix = tfidf.fit_transform(text_features)
                self.content_similarity = cosine_similarity(tfidf_matrix)
                self.content_item_ids = item_ids
                print(f"Content similarity matrix built: {self.content_similarity.shape}")
            except Exception as e:
                print(f"Error building content features: {e}")
                self.content_similarity = None
    
    def fit_collaborative_models(self, train_df):
        """Fit collaborative filtering models"""
        print("Fitting collaborative filtering models...")
        
        # Encode users and items
        train_df = train_df.copy()
        train_df['user_idx'] = self.user_encoder.fit_transform(train_df['user_id'])
        train_df['item_idx'] = self.item_encoder.fit_transform(train_df['item_id'])
        
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        print(f"Matrix dimensions: {n_users} users x {n_items} items")
        
        # Create interaction matrix
        rows = train_df['user_idx'].values
        cols = train_df['item_idx'].values
        data = np.ones(len(train_df))
        
        self.interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        # Fit ALS model
        print("Training ALS model...")
        self.als_model = implicit.als.AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=0.01,
            iterations=15,
            calculate_training_loss=True,
            random_state=42
        )
        self.als_model.fit(self.interaction_matrix.T.tocsr())
        
        # Fit SVD model
        print("Training SVD model...")
        self.svd_model = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.user_factors_svd = self.svd_model.fit_transform(self.interaction_matrix)
        self.item_factors_svd = self.svd_model.components_.T
        
        # Normalize factors
        self.user_factors_svd = normalize(self.user_factors_svd, axis=1)
        self.item_factors_svd = normalize(self.item_factors_svd, axis=1)
        
        # Fit NMF model
        print("Training NMF model...")
        self.nmf_model = NMF(n_components=self.n_factors, random_state=42, max_iter=100)
        self.user_factors_nmf = self.nmf_model.fit_transform(self.interaction_matrix)
        self.item_factors_nmf = self.nmf_model.components_.T
        
        # Compute popularity scores
        self.popularity_scores = train_df['item_id'].value_counts().to_dict()
        self.top_popular_items = list(self.popularity_scores.keys())[:100]
        
        print("Collaborative filtering models fitted successfully!")
    
    def fit_neural_models(self, train_df, epochs=50, lr=0.001):
        """Fit neural models (LightGCN and NCF)"""
        print("Training neural models...")
        
        if not torch.cuda.is_available():
            device = torch.device('cpu')
            print("Using CPU for neural models")
        else:
            device = torch.device('cuda')
            print("Using GPU for neural models")
        
        train_df = train_df.copy()
        train_df['user_idx'] = self.user_encoder.transform(train_df['user_id'])
        train_df['item_idx'] = self.item_encoder.transform(train_df['item_id'])
        
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        # Prepare training data
        users = torch.LongTensor(train_df['user_idx'].values)
        items = torch.LongTensor(train_df['item_idx'].values)
        ratings = torch.FloatTensor(np.ones(len(train_df)))
        
        # Create adjacency matrix for LightGCN
        rows = train_df['user_idx'].values
        cols = train_df['item_idx'].values
        data = np.ones(len(train_df))
        adj_matrix = torch.sparse_coo_tensor(
            torch.LongTensor([rows, cols + n_users]),
            torch.FloatTensor(data),
            (n_users + n_items, n_users + n_items)
        ).to(device)
        
        # Initialize models
        self.lightgcn_model = LightGCNModel(n_users, n_items, self.embedding_dim).to(device)
        self.ncf_model = NCFModel(n_users, n_items, self.embedding_dim).to(device)
        
        # Train LightGCN (simplified training)
        print("Training LightGCN...")
        optimizer_gcn = torch.optim.Adam(self.lightgcn_model.parameters(), lr=lr)
        
        for epoch in range(min(epochs, 20)):  # Reduced epochs for efficiency
            self.lightgcn_model.train()
            
            # Sample batch
            batch_size = min(1024, len(users))
            idx = torch.randperm(len(users))[:batch_size]
            batch_users = users[idx].to(device)
            batch_items = items[idx].to(device)
            batch_ratings = ratings[idx].to(device)
            
            optimizer_gcn.zero_grad()
            pred = self.lightgcn_model(adj_matrix, batch_users, batch_items)
            loss = F.mse_loss(pred, batch_ratings)
            loss.backward()
            optimizer_gcn.step()
            
            if epoch % 5 == 0:
                print(f"LightGCN Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Train NCF
        print("Training NCF...")
        optimizer_ncf = torch.optim.Adam(self.ncf_model.parameters(), lr=lr)
        
        for epoch in range(min(epochs, 20)):  # Reduced epochs for efficiency
            self.ncf_model.train()
            
            # Sample batch with negative sampling
            batch_size = min(512, len(users))
            idx = torch.randperm(len(users))[:batch_size]
            batch_users = users[idx].to(device)
            batch_items = items[idx].to(device)
            
            # Positive samples
            pos_ratings = torch.ones(batch_size).to(device)
            
            # Negative samples
            neg_items = torch.randint(0, n_items, (batch_size,)).to(device)
            neg_ratings = torch.zeros(batch_size).to(device)
            
            # Combine positive and negative
            all_users = torch.cat([batch_users, batch_users])
            all_items = torch.cat([batch_items, neg_items])
            all_ratings = torch.cat([pos_ratings, neg_ratings])
            
            optimizer_ncf.zero_grad()
            pred = self.ncf_model(all_users, all_items)
            loss = F.binary_cross_entropy_with_logits(pred, all_ratings)
            loss.backward()
            optimizer_ncf.step()
            
            if epoch % 5 == 0:
                print(f"NCF Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Move models to CPU for inference
        self.lightgcn_model = self.lightgcn_model.cpu()
        self.ncf_model = self.ncf_model.cpu()
        
        print("Neural models training completed!")
    
    def fit(self, train_df, item_meta_df=None, use_neural=True):
        """Fit all models"""
        print("Starting comprehensive model training...")
        
        # Build content features
        if item_meta_df is not None:
            self.build_content_features(item_meta_df)
        
        # Fit collaborative models
        self.fit_collaborative_models(train_df)
        
        # Fit neural models (optional)
        if use_neural:
            try:
                self.fit_neural_models(train_df)
            except Exception as e:
                print(f"Neural model training failed: {e}")
                print("Continuing with classical models only...")
        
        print("All models fitted successfully!")
    
    def get_als_recommendations(self, user_id, top_k=20):
        """Get recommendations from ALS model"""
        if user_id not in self.user_encoder.classes_:
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        
        try:
            item_indices, scores = self.als_model.recommend(
                user_idx, 
                self.interaction_matrix[user_idx],
                N=top_k,
                filter_already_liked_items=True
            )
            
            item_ids = [self.item_encoder.classes_[idx] for idx in item_indices 
                       if idx < len(self.item_encoder.classes_)]
            
            return list(zip(item_ids, scores))
        except:
            return []
    
    def get_svd_recommendations(self, user_id, top_k=20):
        """Get recommendations from SVD model"""
        if user_id not in self.user_encoder.classes_:
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        
        # Get user's interacted items
        interacted_items = set(self.interaction_matrix[user_idx].indices)
        
        # Compute scores for all items
        scores = self.user_factors_svd[user_idx] @ self.item_factors_svd.T
        
        # Get top items (excluding already interacted)
        item_scores = []
        for item_idx, score in enumerate(scores):
            if item_idx not in interacted_items and item_idx < len(self.item_encoder.classes_):
                item_id = self.item_encoder.classes_[item_idx]
                item_scores.append((item_id, score))
        
        # Sort by score and return top k
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:top_k]
    
    def get_content_recommendations(self, user_id, top_k=20):
        """Get content-based recommendations"""
        if self.content_similarity is None:
            return []
        
        if user_id not in self.user_encoder.classes_:
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        interacted_items = self.interaction_matrix[user_idx].indices
        
        # Find content features for user's interacted items
        user_item_ids = [self.item_encoder.classes_[idx] for idx in interacted_items 
                        if idx < len(self.item_encoder.classes_)]
        
        if not user_item_ids:
            return []
        
        # Get content-based scores
        content_scores = defaultdict(float)
        
        for item_id in user_item_ids:
            if item_id in self.content_item_ids:
                item_idx = self.content_item_ids.index(item_id)
                similarities = self.content_similarity[item_idx]
                
                for candidate_idx, sim_score in enumerate(similarities):
                    candidate_id = self.content_item_ids[candidate_idx]
                    if candidate_id != item_id and candidate_id not in user_item_ids:
                        content_scores[candidate_id] += sim_score
        
        # Sort and return top k
        recommendations = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
    
    def get_ensemble_recommendations(self, user_id, top_k=10):
        """Get ensemble recommendations combining all models"""
        
        # Get recommendations from different models
        als_recs = dict(self.get_als_recommendations(user_id, top_k * 2))
        svd_recs = dict(self.get_svd_recommendations(user_id, top_k * 2))
        content_recs = dict(self.get_content_recommendations(user_id, top_k * 2))
        
        # Combine scores with weights
        final_scores = defaultdict(float)
        
        # ALS weight: 0.4
        for item_id, score in als_recs.items():
            final_scores[item_id] += 0.4 * score
        
        # SVD weight: 0.3
        for item_id, score in svd_recs.items():
            final_scores[item_id] += 0.3 * score
        
        # Content weight: 0.2
        for item_id, score in content_recs.items():
            final_scores[item_id] += 0.2 * score
        
        # Popularity boost: 0.1
        for item_id in self.top_popular_items[:50]:
            if item_id in final_scores:
                final_scores[item_id] += 0.1 * (self.popularity_scores.get(item_id, 0) / 
                                               max(self.popularity_scores.values()))
        
        # Sort and return top k
        recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        rec_items = [item_id for item_id, score in recommendations[:top_k]]
        
        # Fill with popular items if needed
        if len(rec_items) < top_k:
            for item in self.top_popular_items:
                if item not in rec_items:
                    rec_items.append(item)
                if len(rec_items) == top_k:
                    break
        
        return rec_items[:top_k]
    
    def recommend(self, user_id, top_k=10):
        """Main recommendation function"""
        return self.get_ensemble_recommendations(user_id, top_k)

def main():
    """Main function to train and evaluate the recommendation system"""
    print("=" * 70)
    print("ADVANCED RECOMMENDATION SYSTEM")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Load item metadata (sample for efficiency)
    try:
        item_meta_df = pd.read_csv('item_meta.csv', nrows=10000)  # Load sample
        print(f"Item metadata loaded: {item_meta_df.shape}")
    except:
        item_meta_df = None
        print("Item metadata not available")
    
    # Initialize and train the recommendation system
    rec_sys = AdvancedRecommendationSystem(embedding_dim=64, n_factors=100)
    rec_sys.fit(train_df, item_meta_df, use_neural=False)  # Set to True for neural models
    
    # Generate predictions for test users
    print("\nGenerating predictions...")
    
    # For submission format compatibility, we need to read the existing submission structure
    try:
        # Try to read existing submission format
        sample_files = ['submission_v3.csv']
        sample_submission = None
        
        for file in sample_files:
            try:
                sample_submission = pd.read_csv(file)
                print(f"Using submission format from {file}")
                break
            except:
                continue
        
        if sample_submission is None:
            # Create sample submission format
            test_users = test_df['user_id'].unique()
            sample_submission = pd.DataFrame({
                'ID': range(len(test_users)),
                'user_id': test_users,
                'item_id': ''
            })
    except:
        test_users = test_df['user_id'].unique()
        sample_submission = pd.DataFrame({
            'ID': range(len(test_users)),
            'user_id': test_users, 
            'item_id': ''
        })
    
    # Generate predictions
    predictions = {}
    test_users = sample_submission['user_id'].unique()
    
    for i, user_id in enumerate(tqdm(test_users, desc="Generating recommendations")):
        recommendations = rec_sys.recommend(user_id, top_k=10)
        predictions[user_id] = recommendations
    
    # Create submission using the existing format from run_hybrid_ensemble.py
    print("\nCreating submission file...")
    submission_data = []
    
    # Create submission in the exact format
    for idx, row in sample_submission.iterrows():
        user_id = row['user_id']
        
        # Get recommendations for this user
        if user_id in predictions:
            recs = predictions[user_id]
        else:
            # This shouldn't happen, but just in case
            recs = rec_sys.recommend(user_id, top_k=10)
            if len(recs) < 10:
                for item in rec_sys.top_popular_items:
                    if item not in recs:
                        recs.append(item)
                    if len(recs) == 10:
                        break
        
        # Create the row matching sample submission format
        submission_data.append({
            'ID': row['ID'],  # Use the ID from sample submission
            'user_id': user_id,
            'item_id': ','.join(map(str, recs[:10]))
        })
    
    submission_df = pd.DataFrame(submission_data)
    
    # Ensure columns are in the same order as sample submission
    submission_df = submission_df[sample_submission.columns]
    
    # Verify submission format
    print(f"\nSubmission shape: {submission_df.shape}")
    print("Submission sample:")
    print(submission_df.head())
    
    # Save submission
    submission_file = 'advanced_recommendation_submission.csv'
    submission_df.to_csv(submission_file, index=False)
    print(f"\nSubmission saved to: {submission_file}")
    
    print("\nAdvanced recommendation system completed successfully!")

if __name__ == "__main__":
    main() 