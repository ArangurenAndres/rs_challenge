import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, linalg
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, normalize
import implicit
from collections import defaultdict, Counter
import warnings
from tqdm import tqdm
import random
import math
from datetime import datetime

warnings.filterwarnings('ignore')

class MemoryEfficientRecommendationSystem:
    """Memory-efficient recommendation system optimized for extremely sparse data"""
    
    def __init__(self, n_factors=200, alpha=50):
        self.n_factors = n_factors
        self.alpha = alpha
        
        # Encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Models
        self.als_model = None
        self.bpr_model = None
        self.lmf_model = None  # Logistic Matrix Factorization
        self.svd_model = None
        
        # Data structures
        self.interaction_matrix = None
        self.content_vectors = None
        self.popularity_scores = {}
        self.item_features = {}
        self.temporal_weights = {}
        
        # User and item profiles
        self.user_bias = {}
        self.item_bias = {}
        self.global_mean = 0
        
    def preprocess_metadata_efficiently(self, item_meta_df):
        """Efficiently process metadata without creating large similarity matrices"""
        print("Processing item metadata efficiently...")
        
        if item_meta_df is None or len(item_meta_df) == 0:
            print("No metadata available")
            return
        
        # Process features more efficiently
        for _, row in tqdm(item_meta_df.iterrows(), desc="Processing metadata", total=len(item_meta_df)):
            item_id = row['item_id']
            feature_dict = {}
            
            # Extract key features
            if 'main_category' in item_meta_df.columns and pd.notna(row['main_category']):
                feature_dict['category'] = str(row['main_category']).lower().strip()
                
            if 'price' in item_meta_df.columns and pd.notna(row['price']):
                try:
                    price = float(row['price'])
                    if price < 15:
                        feature_dict['price_range'] = 'low'
                    elif price < 50:
                        feature_dict['price_range'] = 'medium'
                    else:
                        feature_dict['price_range'] = 'high'
                except:
                    feature_dict['price_range'] = 'unknown'
            
            if 'average_rating' in item_meta_df.columns and pd.notna(row['average_rating']):
                try:
                    rating = float(row['average_rating'])
                    if rating >= 4.0:
                        feature_dict['quality'] = 'high'
                    elif rating >= 3.0:
                        feature_dict['quality'] = 'medium'
                    else:
                        feature_dict['quality'] = 'low'
                except:
                    feature_dict['quality'] = 'unknown'
            
            self.item_features[item_id] = feature_dict
        
        print(f"Processed features for {len(self.item_features)} items")
    
    def build_temporal_features(self, train_df):
        """Build temporal features for time-aware recommendations"""
        print("Building temporal features...")
        
        if 'timestamp' not in train_df.columns:
            print("No timestamp data available")
            return
        
        try:
            # Convert timestamps
            train_df = train_df.copy()
            train_df['datetime'] = pd.to_datetime(train_df['timestamp'], unit='ms', errors='coerce')
            
            if train_df['datetime'].isna().all():
                print("Invalid timestamp data")
                return
            
            # Calculate recency weights
            max_time = train_df['datetime'].max()
            train_df['days_ago'] = (max_time - train_df['datetime']).dt.days
            
            # Use a stronger decay for very old interactions
            train_df['temporal_weight'] = np.exp(-0.001 * train_df['days_ago'])
            
            # Store temporal weights
            for _, row in train_df.iterrows():
                key = (row['user_id'], row['item_id'])
                self.temporal_weights[key] = row['temporal_weight']
            
            print(f"Built temporal features for {len(self.temporal_weights)} interactions")
            
        except Exception as e:
            print(f"Error building temporal features: {e}")
    
    def compute_bias_terms(self, train_df):
        """Compute user and item bias terms"""
        print("Computing bias terms...")
        
        # Global mean (overall average rating/interaction strength)
        if self.temporal_weights:
            weights = [self.temporal_weights.get((row['user_id'], row['item_id']), 1.0) 
                      for _, row in train_df.iterrows()]
            self.global_mean = np.mean(weights)
        else:
            self.global_mean = 1.0
        
        # User bias (deviation from global mean)
        user_interactions = train_df['user_id'].value_counts()
        user_avg_strength = {}
        
        for user_id in user_interactions.index:
            user_data = train_df[train_df['user_id'] == user_id]
            if self.temporal_weights:
                weights = [self.temporal_weights.get((user_id, row['item_id']), 1.0) 
                          for _, row in user_data.iterrows()]
                avg_strength = np.mean(weights)
            else:
                avg_strength = 1.0
            
            user_avg_strength[user_id] = avg_strength
            self.user_bias[user_id] = avg_strength - self.global_mean
        
        # Item bias (deviation from global mean)
        item_interactions = train_df['item_id'].value_counts()
        for item_id in item_interactions.index:
            item_data = train_df[train_df['item_id'] == item_id]
            if self.temporal_weights:
                weights = [self.temporal_weights.get((row['user_id'], item_id), 1.0) 
                          for _, row in item_data.iterrows()]
                avg_strength = np.mean(weights)
            else:
                avg_strength = 1.0
            
            self.item_bias[item_id] = avg_strength - self.global_mean
        
        print(f"Computed bias terms for {len(self.user_bias)} users and {len(self.item_bias)} items")
    
    def fit_advanced_models(self, train_df):
        """Fit multiple collaborative filtering models efficiently"""
        print("Fitting advanced collaborative filtering models...")
        
        # Encode users and items
        train_df = train_df.copy()
        train_df['user_idx'] = self.user_encoder.fit_transform(train_df['user_id'])
        train_df['item_idx'] = self.item_encoder.fit_transform(train_df['item_id'])
        
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        print(f"Matrix dimensions: {n_users} users x {n_items} items")
        
        # Create weighted interaction matrix
        rows = train_df['user_idx'].values
        cols = train_df['item_idx'].values
        
        # Enhanced weighting strategy
        data = []
        for _, row in train_df.iterrows():
            weight = 1.0
            
            # Temporal weight
            if self.temporal_weights:
                temporal_key = (row['user_id'], row['item_id'])
                weight *= self.temporal_weights.get(temporal_key, 1.0)
            
            # Frequency weight (multiple interactions get higher weight)
            user_item_count = len(train_df[(train_df['user_id'] == row['user_id']) & 
                                          (train_df['item_id'] == row['item_id'])])
            weight *= (1.0 + 0.1 * (user_item_count - 1))  # Slight boost for repeat interactions
            
            data.append(weight)
        
        data = np.array(data)
        self.interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        # Model 1: Enhanced ALS
        print("Training enhanced ALS model...")
        self.als_model = implicit.als.AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=0.0005,  # Very low regularization for sparse data
            iterations=30,
            alpha=self.alpha,
            calculate_training_loss=True,
            random_state=42
        )
        self.als_model.fit(self.interaction_matrix.T.tocsr())
        
        # Model 2: BPR with more iterations
        print("Training BPR model...")
        self.bpr_model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.n_factors,
            regularization=0.001,
            iterations=100,  # More iterations for better learning
            learning_rate=0.005,  # Lower learning rate for stability
            random_state=42
        )
        self.bpr_model.fit(self.interaction_matrix.T.tocsr())
        
        # Model 3: Logistic Matrix Factorization (good for implicit feedback)
        print("Training Logistic Matrix Factorization...")
        self.lmf_model = implicit.lmf.LogisticMatrixFactorization(
            factors=self.n_factors,
            regularization=0.001,
            iterations=30,
            learning_rate=0.01,
            random_state=42
        )
        self.lmf_model.fit(self.interaction_matrix.T.tocsr())
        
        # Model 4: Enhanced SVD
        print("Training enhanced SVD model...")
        self.svd_model = TruncatedSVD(n_components=min(self.n_factors, min(n_users, n_items)-1), 
                                     random_state=42)
        self.user_factors_svd = self.svd_model.fit_transform(self.interaction_matrix)
        self.item_factors_svd = self.svd_model.components_.T
        
        # Normalize factors
        self.user_factors_svd = normalize(self.user_factors_svd, axis=1)
        self.item_factors_svd = normalize(self.item_factors_svd, axis=1)
        
        # Enhanced popularity scores with recency weighting
        self.popularity_scores = {}
        for item_id in train_df['item_id'].unique():
            item_data = train_df[train_df['item_id'] == item_id]
            
            # Base popularity
            base_pop = len(item_data)
            
            # Recency-weighted popularity
            if self.temporal_weights:
                weighted_pop = sum([self.temporal_weights.get((row['user_id'], item_id), 1.0) 
                                  for _, row in item_data.iterrows()])
                popularity = 0.7 * base_pop + 0.3 * weighted_pop
            else:
                popularity = base_pop
            
            self.popularity_scores[item_id] = popularity
        
        # Get diversified top popular items
        popular_items = sorted(self.popularity_scores.items(), key=lambda x: x[1], reverse=True)
        self.top_popular_items = self._diversify_popular_items([item[0] for item in popular_items[:500]])[:150]
        
        print("Advanced collaborative filtering models fitted successfully!")
    
    def _diversify_popular_items(self, item_list, max_per_category=15):
        """Diversify popular items by category"""
        if not self.item_features:
            return item_list[:150]
        
        diversified = []
        category_counts = defaultdict(int)
        
        # First pass: items with category info
        for item_id in item_list:
            if len(diversified) >= 150:
                break
                
            if item_id in self.item_features and 'category' in self.item_features[item_id]:
                category = self.item_features[item_id]['category']
                if category_counts[category] < max_per_category:
                    diversified.append(item_id)
                    category_counts[category] += 1
        
        # Second pass: remaining items without category restriction
        for item_id in item_list:
            if len(diversified) >= 150:
                break
            if item_id not in diversified:
                diversified.append(item_id)
        
        return diversified[:150]
    
    def get_model_recommendations(self, model, user_id, top_k=50):
        """Generic function to get recommendations from any implicit model"""
        if user_id not in self.user_encoder.classes_:
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        
        try:
            item_indices, scores = model.recommend(
                user_idx,
                self.interaction_matrix[user_idx],
                N=top_k,
                filter_already_liked_items=True
            )
            
            recommendations = []
            for idx, score in zip(item_indices, scores):
                if idx < len(self.item_encoder.classes_):
                    item_id = self.item_encoder.classes_[idx]
                    
                    # Apply bias correction
                    corrected_score = score + self.item_bias.get(item_id, 0)
                    recommendations.append((item_id, corrected_score))
            
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations from model: {e}")
            return []
    
    def get_svd_recommendations(self, user_id, top_k=50):
        """SVD-based recommendations with bias correction"""
        if user_id not in self.user_encoder.classes_:
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        user_items = set(self.interaction_matrix[user_idx].indices)
        
        # Compute scores
        user_vector = self.user_factors_svd[user_idx]
        scores = user_vector @ self.item_factors_svd.T
        
        # Apply bias correction
        user_bias_val = self.user_bias.get(user_id, 0)
        
        recommendations = []
        for item_idx, score in enumerate(scores):
            if item_idx not in user_items and item_idx < len(self.item_encoder.classes_):
                item_id = self.item_encoder.classes_[item_idx]
                
                # Bias correction
                corrected_score = score + user_bias_val + self.item_bias.get(item_id, 0)
                recommendations.append((item_id, corrected_score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
    
    def get_content_recommendations(self, user_id, top_k=30):
        """Content-based recommendations using item features"""
        if not self.item_features or user_id not in self.user_encoder.classes_:
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        user_items = self.interaction_matrix[user_idx].indices
        
        # Get user's interacted items
        user_item_ids = [self.item_encoder.classes_[idx] for idx in user_items 
                        if idx < len(self.item_encoder.classes_)]
        
        if not user_item_ids:
            return []
        
        # Build user profile from interacted items
        user_categories = defaultdict(int)
        user_price_ranges = defaultdict(int)
        user_qualities = defaultdict(int)
        
        for item_id in user_item_ids:
            if item_id in self.item_features:
                features = self.item_features[item_id]
                
                if 'category' in features:
                    user_categories[features['category']] += 1
                if 'price_range' in features:
                    user_price_ranges[features['price_range']] += 1
                if 'quality' in features:
                    user_qualities[features['quality']] += 1
        
        # Normalize preferences
        total_interactions = len(user_item_ids)
        if total_interactions == 0:
            return []
        
        for category in user_categories:
            user_categories[category] /= total_interactions
        for price_range in user_price_ranges:
            user_price_ranges[price_range] /= total_interactions
        for quality in user_qualities:
            user_qualities[quality] /= total_interactions
        
        # Score candidate items
        content_scores = []
        for item_id, features in self.item_features.items():
            if item_id not in user_item_ids:
                score = 0.0
                
                # Category similarity
                if 'category' in features and features['category'] in user_categories:
                    score += 0.5 * user_categories[features['category']]
                
                # Price range similarity
                if 'price_range' in features and features['price_range'] in user_price_ranges:
                    score += 0.3 * user_price_ranges[features['price_range']]
                
                # Quality similarity
                if 'quality' in features and features['quality'] in user_qualities:
                    score += 0.2 * user_qualities[features['quality']]
                
                if score > 0:
                    content_scores.append((item_id, score))
        
        content_scores.sort(key=lambda x: x[1], reverse=True)
        return content_scores[:top_k]
    
    def get_advanced_ensemble_recommendations(self, user_id, top_k=10):
        """Advanced ensemble with dynamic weighting"""
        
        # Get recommendations from all models
        als_recs = dict(self.get_model_recommendations(self.als_model, user_id, top_k * 4))
        bpr_recs = dict(self.get_model_recommendations(self.bpr_model, user_id, top_k * 4))
        lmf_recs = dict(self.get_model_recommendations(self.lmf_model, user_id, top_k * 4))
        svd_recs = dict(self.get_svd_recommendations(user_id, top_k * 4))
        content_recs = dict(self.get_content_recommendations(user_id, top_k * 3))
        
        # Determine user activity level
        if user_id in self.user_encoder.classes_:
            user_idx = self.user_encoder.transform([user_id])[0]
            n_interactions = len(self.interaction_matrix[user_idx].indices)
        else:
            n_interactions = 0
        
        # Dynamic weights based on user activity and model performance
        if n_interactions <= 1:
            weights = {
                'als': 0.20,
                'bpr': 0.20,
                'lmf': 0.15,
                'svd': 0.15,
                'content': 0.20,
                'popularity': 0.10
            }
        elif n_interactions <= 3:
            weights = {
                'als': 0.25,
                'bpr': 0.25,
                'lmf': 0.20,
                'svd': 0.15,
                'content': 0.10,
                'popularity': 0.05
            }
        elif n_interactions <= 10:
            weights = {
                'als': 0.30,
                'bpr': 0.30,
                'lmf': 0.25,
                'svd': 0.10,
                'content': 0.03,
                'popularity': 0.02
            }
        else:
            weights = {
                'als': 0.35,
                'bpr': 0.35,
                'lmf': 0.20,
                'svd': 0.08,
                'content': 0.01,
                'popularity': 0.01
            }
        
        # Normalize scores for fair combination
        def normalize_scores(score_dict, clip_quantile=0.95):
            if not score_dict:
                return {}
            
            scores = list(score_dict.values())
            if len(scores) <= 1:
                return {k: 1.0 for k in score_dict}
            
            # Clip extreme values
            clip_value = np.quantile(scores, clip_quantile)
            clipped_scores = {k: min(v, clip_value) for k, v in score_dict.items()}
            
            # Min-max normalization
            min_score = min(clipped_scores.values())
            max_score = max(clipped_scores.values())
            
            if max_score == min_score:
                return {k: 1.0 for k in clipped_scores}
            
            return {k: (v - min_score) / (max_score - min_score) 
                   for k, v in clipped_scores.items()}
        
        # Normalize all model scores
        als_recs = normalize_scores(als_recs)
        bpr_recs = normalize_scores(bpr_recs)
        lmf_recs = normalize_scores(lmf_recs)
        svd_recs = normalize_scores(svd_recs)
        content_recs = normalize_scores(content_recs)
        
        # Combine scores
        final_scores = defaultdict(float)
        
        for item_id, score in als_recs.items():
            final_scores[item_id] += weights['als'] * score
        
        for item_id, score in bpr_recs.items():
            final_scores[item_id] += weights['bpr'] * score
        
        for item_id, score in lmf_recs.items():
            final_scores[item_id] += weights['lmf'] * score
        
        for item_id, score in svd_recs.items():
            final_scores[item_id] += weights['svd'] * score
        
        for item_id, score in content_recs.items():
            final_scores[item_id] += weights['content'] * score
        
        # Add popularity boost (with diversification)
        max_pop = max(self.popularity_scores.values()) if self.popularity_scores else 1
        for item_id in self.top_popular_items[:100]:
            if item_id in final_scores:
                pop_score = self.popularity_scores.get(item_id, 0) / max_pop
                final_scores[item_id] += weights['popularity'] * pop_score
        
        # Sort and return recommendations
        recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        rec_items = [item_id for item_id, score in recommendations[:top_k * 2]]
        
        # Final diversification
        rec_items = self._diversify_recommendations(rec_items, top_k)
        
        # Fill with popular items if needed
        while len(rec_items) < top_k:
            for item in self.top_popular_items:
                if item not in rec_items:
                    rec_items.append(item)
                    break
            else:
                break  # No more popular items available
        
        return rec_items[:top_k]
    
    def _diversify_recommendations(self, item_list, target_count):
        """Final diversification step"""
        if not self.item_features:
            return item_list[:target_count]
        
        diversified = []
        category_counts = defaultdict(int)
        max_per_category = max(2, target_count // 5)  # At most 20% from same category
        
        # First pass: respect category limits
        for item_id in item_list:
            if len(diversified) >= target_count:
                break
                
            if item_id in self.item_features and 'category' in self.item_features[item_id]:
                category = self.item_features[item_id]['category']
                if category_counts[category] < max_per_category:
                    diversified.append(item_id)
                    category_counts[category] += 1
            else:
                diversified.append(item_id)
        
        # Second pass: fill remaining slots
        for item_id in item_list:
            if len(diversified) >= target_count:
                break
            if item_id not in diversified:
                diversified.append(item_id)
        
        return diversified[:target_count]
    
    def fit(self, train_df, item_meta_df=None):
        """Fit all models"""
        print("Starting memory-efficient recommendation system training...")
        
        # Build temporal features
        self.build_temporal_features(train_df)
        
        # Compute bias terms
        self.compute_bias_terms(train_df)
        
        # Process metadata efficiently
        if item_meta_df is not None:
            self.preprocess_metadata_efficiently(item_meta_df)
        
        # Fit collaborative models
        self.fit_advanced_models(train_df)
        
        print("Memory-efficient recommendation system training completed!")
    
    def recommend(self, user_id, top_k=10):
        """Main recommendation function"""
        return self.get_advanced_ensemble_recommendations(user_id, top_k)

def main():
    """Main function"""
    print("=" * 70)
    print("MEMORY-EFFICIENT IMPROVED RECOMMENDATION SYSTEM")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Load metadata efficiently (larger sample)
    try:
        item_meta_df = pd.read_csv('item_meta.csv', nrows=100000)
        print(f"Item metadata loaded: {item_meta_df.shape}")
    except:
        item_meta_df = None
        print("Item metadata not available")
    
    # Initialize and train the system
    rec_sys = MemoryEfficientRecommendationSystem(n_factors=200, alpha=50)
    rec_sys.fit(train_df, item_meta_df)
    
    # Generate predictions
    print("\nGenerating predictions...")
    
    # Use existing submission format
    try:
        sample_submission = pd.read_csv('submission_v3.csv')
        print(f"Using existing submission format: {sample_submission.shape}")
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
    
    for user_id in tqdm(test_users, desc="Generating optimized recommendations"):
        recommendations = rec_sys.recommend(user_id, top_k=10)
        predictions[user_id] = recommendations
    
    # Create submission using the format from run_hybrid_ensemble.py
    print("\nCreating submission file...")
    submission_data = []
    
    for idx, row in sample_submission.iterrows():
        user_id = row['user_id']
        
        if user_id in predictions:
            recs = predictions[user_id]
        else:
            recs = rec_sys.recommend(user_id, top_k=10)
            if len(recs) < 10:
                for item in rec_sys.top_popular_items:
                    if item not in recs:
                        recs.append(item)
                    if len(recs) == 10:
                        break
        
        submission_data.append({
            'ID': row['ID'],
            'user_id': user_id,
            'item_id': ','.join(map(str, recs[:10]))
        })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df = submission_df[sample_submission.columns]
    
    print(f"\nSubmission shape: {submission_df.shape}")
    print("Submission sample:")
    print(submission_df.head())
    
    # Save submission
    submission_file = 'memory_efficient_recommendation_submission.csv'
    submission_df.to_csv(submission_file, index=False)
    print(f"\nSubmission saved to: {submission_file}")
    
    print("\nMemory-efficient recommendation system completed!")
    print("Key optimizations:")
    print("- Avoided large similarity matrix computation")
    print("- Enhanced matrix factorization (200 factors)")
    print("- Multiple collaborative filtering models (ALS, BPR, LMF)")
    print("- Bias correction for better accuracy")
    print("- Temporal weighting for recency")
    print("- Advanced ensemble with dynamic weights")
    print("- Memory-efficient metadata processing")
    print("- Smart diversification strategies")

if __name__ == "__main__":
    main() 