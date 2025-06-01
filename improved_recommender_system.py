import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
import implicit
from collections import defaultdict, Counter
import warnings
from tqdm import tqdm
import random
import math
from datetime import datetime

warnings.filterwarnings('ignore')

class ImprovedRecommendationSystem:
    """Improved recommendation system optimized for extremely sparse data"""
    
    def __init__(self, n_factors=150, alpha=40):
        self.n_factors = n_factors
        self.alpha = alpha
        
        # Encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Models
        self.als_model = None
        self.bpr_model = None
        self.svd_model = None
        self.item_knn_model = None
        
        # Data structures
        self.interaction_matrix = None
        self.item_similarity_matrix = None
        self.content_similarity = None
        self.popularity_scores = {}
        self.item_features = {}
        self.temporal_weights = {}
        
        # Advanced features
        self.category_popularity = {}
        self.price_bins = {}
        self.rating_bins = {}
        
    def preprocess_metadata(self, item_meta_df):
        """Enhanced metadata preprocessing"""
        print("Processing item metadata comprehensively...")
        
        if item_meta_df is None or len(item_meta_df) == 0:
            print("No metadata available")
            return
        
        # Load more metadata for better coverage
        try:
            if len(item_meta_df) < 50000:  # If we loaded only sample, load more
                item_meta_df = pd.read_csv('item_meta.csv', nrows=50000)
                print(f"Loaded extended metadata: {item_meta_df.shape}")
        except:
            pass
        
        text_features = []
        item_ids = []
        
        for _, row in item_meta_df.iterrows():
            item_id = row['item_id']
            item_ids.append(item_id)
            
            # Enhanced text processing
            text_parts = []
            feature_dict = {}
            
            # Text features with higher weight on important fields
            if 'main_category' in item_meta_df.columns and pd.notna(row['main_category']):
                category = str(row['main_category']).lower().strip()
                text_parts.extend([category] * 3)  # Higher weight
                feature_dict['category'] = category
                
            if 'title' in item_meta_df.columns and pd.notna(row['title']):
                title = str(row['title']).lower().strip()
                text_parts.extend([title] * 2)  # Medium weight
                feature_dict['title'] = title
                
            for col in ['features', 'description', 'categories']:
                if col in item_meta_df.columns and pd.notna(row[col]):
                    text = str(row[col]).lower().strip()
                    text_parts.append(text)
                    feature_dict[col] = text
            
            # Numeric features with binning
            if 'price' in item_meta_df.columns and pd.notna(row['price']):
                try:
                    price = float(row['price'])
                    if price < 10:
                        price_bin = 'very_low'
                    elif price < 25:
                        price_bin = 'low'
                    elif price < 50:
                        price_bin = 'medium'
                    elif price < 100:
                        price_bin = 'high'
                    else:
                        price_bin = 'very_high'
                    
                    feature_dict['price_bin'] = price_bin
                    text_parts.append(price_bin)
                except:
                    pass
            
            if 'average_rating' in item_meta_df.columns and pd.notna(row['average_rating']):
                try:
                    rating = float(row['average_rating'])
                    if rating >= 4.5:
                        rating_bin = 'excellent'
                    elif rating >= 4.0:
                        rating_bin = 'very_good'
                    elif rating >= 3.5:
                        rating_bin = 'good'
                    elif rating >= 3.0:
                        rating_bin = 'average'
                    else:
                        rating_bin = 'poor'
                    
                    feature_dict['rating_bin'] = rating_bin
                    text_parts.append(rating_bin)
                except:
                    pass
            
            text_features.append(' '.join(text_parts))
            self.item_features[item_id] = feature_dict
        
        # Build enhanced TF-IDF similarity
        if text_features:
            try:
                # Use character-level n-grams for better similarity
                tfidf = TfidfVectorizer(
                    max_features=10000,
                    stop_words='english',
                    ngram_range=(1, 3),
                    analyzer='word',
                    min_df=2,
                    max_df=0.95
                )
                tfidf_matrix = tfidf.fit_transform(text_features)
                self.content_similarity = cosine_similarity(tfidf_matrix)
                self.content_item_ids = item_ids
                print(f"Enhanced content similarity matrix: {self.content_similarity.shape}")
                
                # Build category popularity
                for item_id in item_ids:
                    if item_id in self.item_features and 'category' in self.item_features[item_id]:
                        category = self.item_features[item_id]['category']
                        if category not in self.category_popularity:
                            self.category_popularity[category] = 0
                
            except Exception as e:
                print(f"Error building content features: {e}")
                self.content_similarity = None
    
    def build_temporal_features(self, train_df):
        """Build temporal features for time-aware recommendations"""
        print("Building temporal features...")
        
        if 'timestamp' not in train_df.columns:
            return
        
        try:
            # Convert timestamps
            train_df['datetime'] = pd.to_datetime(train_df['timestamp'], unit='ms', errors='coerce')
            
            if train_df['datetime'].isna().all():
                return
            
            # Calculate recency weights (more recent = higher weight)
            max_time = train_df['datetime'].max()
            train_df['days_ago'] = (max_time - train_df['datetime']).dt.days
            
            # Exponential decay: weight = exp(-lambda * days_ago)
            lambda_decay = 0.01  # Adjust this for faster/slower decay
            train_df['temporal_weight'] = np.exp(-lambda_decay * train_df['days_ago'])
            
            # Store temporal weights per interaction
            for _, row in train_df.iterrows():
                key = (row['user_id'], row['item_id'])
                self.temporal_weights[key] = row['temporal_weight']
            
            print(f"Temporal features built for {len(self.temporal_weights)} interactions")
            
        except Exception as e:
            print(f"Error building temporal features: {e}")
    
    def fit_advanced_models(self, train_df):
        """Fit multiple advanced collaborative filtering models"""
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
        
        # Use temporal weights if available
        if self.temporal_weights:
            weights = []
            for _, row in train_df.iterrows():
                key = (row['user_id'], row['item_id'])
                weight = self.temporal_weights.get(key, 1.0)
                weights.append(weight)
            data = np.array(weights)
        else:
            data = np.ones(len(train_df))
        
        self.interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        # Model 1: Enhanced ALS with higher factors
        print("Training enhanced ALS model...")
        self.als_model = implicit.als.AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=0.001,  # Lower regularization for sparse data
            iterations=25,  # More iterations
            alpha=self.alpha,
            calculate_training_loss=True,
            random_state=42
        )
        self.als_model.fit(self.interaction_matrix.T.tocsr())
        
        # Model 2: BPR (Bayesian Personalized Ranking) - better for implicit feedback
        print("Training BPR model...")
        self.bpr_model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.n_factors,
            regularization=0.001,
            iterations=50,
            learning_rate=0.01,
            random_state=42
        )
        self.bpr_model.fit(self.interaction_matrix.T.tocsr())
        
        # Model 3: Enhanced SVD with more components
        print("Training enhanced SVD model...")
        self.svd_model = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.user_factors_svd = self.svd_model.fit_transform(self.interaction_matrix)
        self.item_factors_svd = self.svd_model.components_.T
        
        # Normalize factors
        self.user_factors_svd = normalize(self.user_factors_svd, axis=1)
        self.item_factors_svd = normalize(self.item_factors_svd, axis=1)
        
        # Model 4: Item-Item KNN (works well for sparse data)
        print("Building item-item similarity model...")
        # Transpose for item-item similarity
        item_matrix = self.interaction_matrix.T.tocsr()
        
        # Use cosine similarity for item-item
        from sklearn.metrics.pairwise import cosine_similarity
        item_similarity = cosine_similarity(item_matrix)
        
        # Keep only top-k similar items for efficiency
        k_similar = 50
        for i in range(item_similarity.shape[0]):
            # Keep only top-k similarities, set others to 0
            top_k_indices = np.argsort(item_similarity[i])[::-1][:k_similar]
            mask = np.zeros_like(item_similarity[i], dtype=bool)
            mask[top_k_indices] = True
            item_similarity[i][~mask] = 0
        
        self.item_similarity_matrix = item_similarity
        
        # Compute enhanced popularity scores
        self.popularity_scores = train_df['item_id'].value_counts().to_dict()
        
        # Category-wise popularity
        for item_id, count in self.popularity_scores.items():
            if item_id in self.item_features and 'category' in self.item_features[item_id]:
                category = self.item_features[item_id]['category']
                if category not in self.category_popularity:
                    self.category_popularity[category] = 0
                self.category_popularity[category] += count
        
        # Get top popular items with diversity
        popular_items = list(self.popularity_scores.keys())
        self.top_popular_items = self._diversify_recommendations(popular_items[:200])[:100]
        
        print("Advanced collaborative filtering models fitted successfully!")
    
    def _diversify_recommendations(self, item_list, max_per_category=10):
        """Diversify recommendations by limiting items per category"""
        if not self.item_features:
            return item_list
        
        diversified = []
        category_counts = defaultdict(int)
        
        for item_id in item_list:
            if item_id in self.item_features and 'category' in self.item_features[item_id]:
                category = self.item_features[item_id]['category']
                if category_counts[category] < max_per_category:
                    diversified.append(item_id)
                    category_counts[category] += 1
            else:
                diversified.append(item_id)
            
            if len(diversified) >= len(item_list):
                break
        
        return diversified
    
    def get_als_recommendations(self, user_id, top_k=30):
        """Enhanced ALS recommendations"""
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
            
            item_ids = []
            item_scores = []
            for idx, score in zip(item_indices, scores):
                if idx < len(self.item_encoder.classes_):
                    item_id = self.item_encoder.classes_[idx]
                    item_ids.append(item_id)
                    item_scores.append(score)
            
            return list(zip(item_ids, item_scores))
        except:
            return []
    
    def get_bpr_recommendations(self, user_id, top_k=30):
        """BPR recommendations"""
        if user_id not in self.user_encoder.classes_:
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        
        try:
            item_indices, scores = self.bpr_model.recommend(
                user_idx,
                self.interaction_matrix[user_idx],
                N=top_k,
                filter_already_liked_items=True
            )
            
            item_ids = []
            item_scores = []
            for idx, score in zip(item_indices, scores):
                if idx < len(self.item_encoder.classes_):
                    item_id = self.item_encoder.classes_[idx]
                    item_ids.append(item_id)
                    item_scores.append(score)
            
            return list(zip(item_ids, item_scores))
        except:
            return []
    
    def get_item_based_recommendations(self, user_id, top_k=30):
        """Item-based collaborative filtering recommendations"""
        if user_id not in self.user_encoder.classes_:
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        user_items = self.interaction_matrix[user_idx].indices
        
        if len(user_items) == 0:
            return []
        
        # Calculate scores for all items based on item similarity
        item_scores = np.zeros(self.item_similarity_matrix.shape[0])
        
        for item_idx in user_items:
            if item_idx < self.item_similarity_matrix.shape[0]:
                # Weight by user's rating and item similarities
                user_rating = self.interaction_matrix[user_idx, item_idx]
                similarities = self.item_similarity_matrix[item_idx]
                item_scores += user_rating * similarities
        
        # Get top items (excluding already rated)
        item_rankings = []
        for item_idx, score in enumerate(item_scores):
            if item_idx not in user_items and score > 0:
                if item_idx < len(self.item_encoder.classes_):
                    item_id = self.item_encoder.classes_[item_idx]
                    item_rankings.append((item_id, score))
        
        # Sort by score
        item_rankings.sort(key=lambda x: x[1], reverse=True)
        return item_rankings[:top_k]
    
    def get_content_recommendations(self, user_id, top_k=30):
        """Enhanced content-based recommendations"""
        if self.content_similarity is None or user_id not in self.user_encoder.classes_:
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        user_items = self.interaction_matrix[user_idx].indices
        
        if len(user_items) == 0:
            return []
        
        # Map to content item IDs
        user_item_ids = [self.item_encoder.classes_[idx] for idx in user_items 
                        if idx < len(self.item_encoder.classes_)]
        
        content_scores = defaultdict(float)
        total_weight = 0
        
        for item_id in user_item_ids:
            if item_id in self.content_item_ids:
                item_idx = self.content_item_ids.index(item_id)
                similarities = self.content_similarity[item_idx]
                
                # Weight by item popularity (less popular items get higher weight for diversity)
                item_weight = 1.0 / (1 + np.log(1 + self.popularity_scores.get(item_id, 1)))
                total_weight += item_weight
                
                for candidate_idx, sim_score in enumerate(similarities):
                    candidate_id = self.content_item_ids[candidate_idx]
                    if candidate_id != item_id and candidate_id not in user_item_ids:
                        content_scores[candidate_id] += item_weight * sim_score
        
        # Normalize scores
        if total_weight > 0:
            for item_id in content_scores:
                content_scores[item_id] /= total_weight
        
        recommendations = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
    
    def get_category_recommendations(self, user_id, top_k=20):
        """Category-based recommendations for users with limited history"""
        if user_id not in self.user_encoder.classes_:
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        user_items = self.interaction_matrix[user_idx].indices
        
        # Find user's preferred categories
        user_categories = defaultdict(int)
        user_item_ids = [self.item_encoder.classes_[idx] for idx in user_items 
                        if idx < len(self.item_encoder.classes_)]
        
        for item_id in user_item_ids:
            if item_id in self.item_features and 'category' in self.item_features[item_id]:
                category = self.item_features[item_id]['category']
                user_categories[category] += 1
        
        if not user_categories:
            return []
        
        # Get popular items from user's preferred categories
        category_recs = []
        for category, count in sorted(user_categories.items(), key=lambda x: x[1], reverse=True):
            category_items = []
            for item_id, pop_score in self.popularity_scores.items():
                if (item_id in self.item_features and 
                    'category' in self.item_features[item_id] and
                    self.item_features[item_id]['category'] == category and
                    item_id not in user_item_ids):
                    category_items.append((item_id, pop_score))
            
            # Sort by popularity and take top items
            category_items.sort(key=lambda x: x[1], reverse=True)
            category_recs.extend(category_items[:10])  # Top 10 per category
        
        return category_recs[:top_k]
    
    def get_advanced_ensemble_recommendations(self, user_id, top_k=10):
        """Advanced ensemble with adaptive weights"""
        
        # Get recommendations from all models
        als_recs = dict(self.get_als_recommendations(user_id, top_k * 3))
        bpr_recs = dict(self.get_bpr_recommendations(user_id, top_k * 3))
        item_recs = dict(self.get_item_based_recommendations(user_id, top_k * 3))
        content_recs = dict(self.get_content_recommendations(user_id, top_k * 3))
        category_recs = dict(self.get_category_recommendations(user_id, top_k * 2))
        
        # Adaptive weighting based on user's interaction history
        if user_id in self.user_encoder.classes_:
            user_idx = self.user_encoder.transform([user_id])[0]
            n_interactions = len(self.interaction_matrix[user_idx].indices)
        else:
            n_interactions = 0
        
        # Adjust weights based on user's activity level
        if n_interactions <= 1:
            # For users with very few interactions, rely more on popularity and content
            weights = {
                'als': 0.15,
                'bpr': 0.15, 
                'item': 0.10,
                'content': 0.25,
                'category': 0.20,
                'popularity': 0.15
            }
        elif n_interactions <= 5:
            # For users with few interactions, balanced approach
            weights = {
                'als': 0.25,
                'bpr': 0.25,
                'item': 0.15,
                'content': 0.20,
                'category': 0.10,
                'popularity': 0.05
            }
        else:
            # For active users, rely more on collaborative filtering
            weights = {
                'als': 0.35,
                'bpr': 0.35,
                'item': 0.15,
                'content': 0.10,
                'category': 0.03,
                'popularity': 0.02
            }
        
        # Combine scores
        final_scores = defaultdict(float)
        
        # Normalize scores first
        def normalize_scores(score_dict):
            if not score_dict:
                return {}
            scores = list(score_dict.values())
            if max(scores) == min(scores):
                return {k: 1.0 for k in score_dict}
            
            min_score, max_score = min(scores), max(scores)
            return {k: (v - min_score) / (max_score - min_score) 
                   for k, v in score_dict.items()}
        
        als_recs = normalize_scores(als_recs)
        bpr_recs = normalize_scores(bpr_recs)
        item_recs = normalize_scores(item_recs)
        content_recs = normalize_scores(content_recs)
        category_recs = normalize_scores(category_recs)
        
        # Add weighted scores
        for item_id, score in als_recs.items():
            final_scores[item_id] += weights['als'] * score
        
        for item_id, score in bpr_recs.items():
            final_scores[item_id] += weights['bpr'] * score
            
        for item_id, score in item_recs.items():
            final_scores[item_id] += weights['item'] * score
        
        for item_id, score in content_recs.items():
            final_scores[item_id] += weights['content'] * score
        
        for item_id, score in category_recs.items():
            final_scores[item_id] += weights['category'] * score
        
        # Add popularity boost
        for item_id in self.top_popular_items[:50]:
            if item_id in final_scores:
                pop_boost = weights['popularity'] * (self.popularity_scores.get(item_id, 0) / 
                                                   max(self.popularity_scores.values()))
                final_scores[item_id] += pop_boost
        
        # Sort and get top recommendations
        recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        rec_items = [item_id for item_id, score in recommendations[:top_k]]
        
        # Diversify recommendations
        rec_items = self._diversify_recommendations(rec_items)
        
        # Fill with popular items if needed
        if len(rec_items) < top_k:
            for item in self.top_popular_items:
                if item not in rec_items:
                    rec_items.append(item)
                if len(rec_items) == top_k:
                    break
        
        return rec_items[:top_k]
    
    def fit(self, train_df, item_meta_df=None):
        """Fit all models"""
        print("Starting improved recommendation system training...")
        
        # Build temporal features
        self.build_temporal_features(train_df)
        
        # Process metadata
        if item_meta_df is not None:
            self.preprocess_metadata(item_meta_df)
        
        # Fit collaborative models
        self.fit_advanced_models(train_df)
        
        print("Improved recommendation system training completed!")
    
    def recommend(self, user_id, top_k=10):
        """Main recommendation function"""
        return self.get_advanced_ensemble_recommendations(user_id, top_k)

def main():
    """Main function"""
    print("=" * 70)
    print("IMPROVED RECOMMENDATION SYSTEM FOR SPARSE DATA")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Load more metadata for better coverage
    try:
        item_meta_df = pd.read_csv('item_meta.csv', nrows=50000)  # Load more data
        print(f"Item metadata loaded: {item_meta_df.shape}")
    except:
        item_meta_df = None
        print("Item metadata not available")
    
    # Initialize and train the system
    rec_sys = ImprovedRecommendationSystem(n_factors=150, alpha=40)
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
    
    for user_id in tqdm(test_users, desc="Generating improved recommendations"):
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
    submission_file = 'improved_recommendation_submission.csv'
    submission_df.to_csv(submission_file, index=False)
    print(f"\nSubmission saved to: {submission_file}")
    
    print("\nImproved recommendation system completed!")
    print("Key improvements:")
    print("- Enhanced matrix factorization with more factors")
    print("- BPR model for better implicit feedback learning")
    print("- Item-item collaborative filtering")
    print("- Advanced content-based filtering")
    print("- Temporal weighting")
    print("- Adaptive ensemble weights based on user activity")
    print("- Recommendation diversification")

if __name__ == "__main__":
    main() 