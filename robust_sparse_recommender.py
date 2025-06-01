import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, normalize
import implicit
from collections import defaultdict, Counter
import warnings
from tqdm import tqdm
import random
import math

warnings.filterwarnings('ignore')

class RobustSparseRecommendationSystem:
    """Robust recommendation system specifically designed for extremely sparse data"""
    
    def __init__(self, n_factors=150, alpha=60):
        self.n_factors = n_factors
        self.alpha = alpha
        
        # Encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Models
        self.als_model = None
        self.bpr_model = None
        self.svd_model = None
        
        # Data structures
        self.interaction_matrix = None
        self.popularity_scores = {}
        self.item_features = {}
        self.temporal_weights = {}
        
        # Enhanced fallback strategies
        self.category_items = defaultdict(list)
        self.price_range_items = defaultdict(list)
        self.quality_items = defaultdict(list)
        
        # Bias correction
        self.user_bias = {}
        self.item_bias = {}
        self.global_mean = 1.0
        
        # Co-occurrence data
        self.item_cooccurrence = defaultdict(lambda: defaultdict(int))
        
    def preprocess_metadata_efficiently(self, item_meta_df):
        """Process metadata with robust error handling"""
        print("Processing item metadata with robust handling...")
        
        if item_meta_df is None or len(item_meta_df) == 0:
            print("No metadata available")
            return
        
        processed_count = 0
        for _, row in tqdm(item_meta_df.iterrows(), desc="Processing metadata", total=len(item_meta_df)):
            try:
                item_id = row['item_id']
                feature_dict = {}
                
                # Category processing
                if 'main_category' in item_meta_df.columns and pd.notna(row['main_category']):
                    category = str(row['main_category']).lower().strip()
                    feature_dict['category'] = category
                    self.category_items[category].append(item_id)
                
                # Price processing
                if 'price' in item_meta_df.columns and pd.notna(row['price']):
                    try:
                        price = float(row['price'])
                        if price < 20:
                            price_range = 'low'
                        elif price < 75:
                            price_range = 'medium'
                        else:
                            price_range = 'high'
                        feature_dict['price_range'] = price_range
                        self.price_range_items[price_range].append(item_id)
                    except:
                        feature_dict['price_range'] = 'unknown'
                        self.price_range_items['unknown'].append(item_id)
                
                # Quality processing
                if 'average_rating' in item_meta_df.columns and pd.notna(row['average_rating']):
                    try:
                        rating = float(row['average_rating'])
                        if rating >= 4.2:
                            quality = 'excellent'
                        elif rating >= 3.5:
                            quality = 'good'
                        else:
                            quality = 'average'
                        feature_dict['quality'] = quality
                        self.quality_items[quality].append(item_id)
                    except:
                        feature_dict['quality'] = 'unknown'
                        self.quality_items['unknown'].append(item_id)
                
                self.item_features[item_id] = feature_dict
                processed_count += 1
                
            except Exception as e:
                continue  # Skip problematic rows
        
        print(f"Successfully processed features for {processed_count} items")
        print(f"Categories: {len(self.category_items)}")
        print(f"Price ranges: {len(self.price_range_items)}")
        print(f"Quality levels: {len(self.quality_items)}")
    
    def build_cooccurrence_matrix(self, train_df):
        """Build item co-occurrence matrix for users with multiple interactions"""
        print("Building item co-occurrence matrix...")
        
        # Group by user
        user_items = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
        
        cooccurrence_count = 0
        for user_id, items in user_items.items():
            if len(items) > 1:  # Only users with multiple interactions
                # For each pair of items this user interacted with
                for i in range(len(items)):
                    for j in range(i+1, len(items)):
                        item1, item2 = items[i], items[j]
                        self.item_cooccurrence[item1][item2] += 1
                        self.item_cooccurrence[item2][item1] += 1
                        cooccurrence_count += 2
        
        print(f"Built co-occurrence data for {cooccurrence_count} item pairs")
    
    def build_temporal_features(self, train_df):
        """Build temporal features with enhanced weighting"""
        print("Building enhanced temporal features...")
        
        if 'timestamp' not in train_df.columns:
            print("No timestamp data available")
            return
        
        try:
            train_df = train_df.copy()
            train_df['datetime'] = pd.to_datetime(train_df['timestamp'], unit='ms', errors='coerce')
            
            if train_df['datetime'].isna().all():
                print("Invalid timestamp data")
                return
            
            # More aggressive temporal weighting
            max_time = train_df['datetime'].max()
            train_df['days_ago'] = (max_time - train_df['datetime']).dt.days
            
            # Stronger decay for old interactions, but not too aggressive
            train_df['temporal_weight'] = np.exp(-0.002 * train_df['days_ago'])
            
            # Store temporal weights
            for _, row in train_df.iterrows():
                key = (row['user_id'], row['item_id'])
                self.temporal_weights[key] = row['temporal_weight']
            
            print(f"Built temporal features for {len(self.temporal_weights)} interactions")
            print(f"Temporal weight range: {train_df['temporal_weight'].min():.4f} - {train_df['temporal_weight'].max():.4f}")
            
        except Exception as e:
            print(f"Error building temporal features: {e}")
    
    def compute_bias_terms(self, train_df):
        """Compute bias terms for better score calibration"""
        print("Computing bias terms...")
        
        # Global statistics
        if self.temporal_weights:
            all_weights = [self.temporal_weights.get((row['user_id'], row['item_id']), 1.0) 
                          for _, row in train_df.iterrows()]
            self.global_mean = np.mean(all_weights)
        else:
            self.global_mean = 1.0
        
        # User bias - users who interact more or with higher temporal weights
        for user_id in train_df['user_id'].unique():
            user_data = train_df[train_df['user_id'] == user_id]
            if self.temporal_weights:
                user_weights = [self.temporal_weights.get((user_id, row['item_id']), 1.0) 
                               for _, row in user_data.iterrows()]
                user_strength = np.mean(user_weights)
            else:
                user_strength = len(user_data)  # Frequency-based for users without temporal data
            
            self.user_bias[user_id] = np.log(1 + user_strength) - np.log(1 + self.global_mean)
        
        # Item bias - popular items get positive bias
        for item_id in train_df['item_id'].unique():
            item_data = train_df[train_df['item_id'] == item_id]
            item_popularity = len(item_data)
            
            if self.temporal_weights:
                item_weights = [self.temporal_weights.get((row['user_id'], item_id), 1.0) 
                               for _, row in item_data.iterrows()]
                item_strength = np.mean(item_weights) * item_popularity
            else:
                item_strength = item_popularity
            
            self.item_bias[item_id] = np.log(1 + item_strength) - np.log(1 + self.global_mean)
        
        print(f"Computed bias terms for {len(self.user_bias)} users and {len(self.item_bias)} items")
    
    def fit_robust_models(self, train_df):
        """Fit models with robust error handling"""
        print("Fitting robust collaborative filtering models...")
        
        # Encode users and items
        train_df = train_df.copy()
        train_df['user_idx'] = self.user_encoder.fit_transform(train_df['user_id'])
        train_df['item_idx'] = self.item_encoder.fit_transform(train_df['item_id'])
        
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        print(f"Matrix dimensions: {n_users} users x {n_items} items")
        
        # Create interaction matrix with enhanced weighting
        rows = train_df['user_idx'].values
        cols = train_df['item_idx'].values
        
        # Multi-factor weighting
        data = []
        for _, row in train_df.iterrows():
            weight = 1.0
            
            # Temporal weight
            if self.temporal_weights:
                temporal_key = (row['user_id'], row['item_id'])
                weight *= self.temporal_weights.get(temporal_key, 1.0)
            
            # Frequency bonus (but capped)
            user_item_freq = len(train_df[(train_df['user_id'] == row['user_id']) & 
                                         (train_df['item_id'] == row['item_id'])])
            weight *= min(1.5, 1.0 + 0.1 * user_item_freq)
            
            data.append(weight)
        
        data = np.array(data)
        self.interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        print(f"Interaction matrix density: {self.interaction_matrix.nnz / (n_users * n_items):.6f}")
        
        # Model 1: ALS with optimized parameters for sparse data
        print("Training optimized ALS model...")
        try:
            self.als_model = implicit.als.AlternatingLeastSquares(
                factors=self.n_factors,
                regularization=0.0001,  # Very low regularization
                iterations=40,  # More iterations
                alpha=self.alpha,  # Higher alpha for confidence
                calculate_training_loss=True,
                random_state=42
            )
            self.als_model.fit(self.interaction_matrix.T.tocsr())
            print("ALS model trained successfully")
        except Exception as e:
            print(f"ALS training failed: {e}")
            self.als_model = None
        
        # Model 2: BPR for ranking
        print("Training optimized BPR model...")
        try:
            self.bpr_model = implicit.bpr.BayesianPersonalizedRanking(
                factors=self.n_factors,
                regularization=0.0005,
                iterations=150,  # Many more iterations
                learning_rate=0.003,  # Lower learning rate
                random_state=42
            )
            self.bpr_model.fit(self.interaction_matrix.T.tocsr())
            print("BPR model trained successfully")
        except Exception as e:
            print(f"BPR training failed: {e}")
            self.bpr_model = None
        
        # Model 3: SVD with robust parameters
        print("Training robust SVD model...")
        try:
            max_components = min(self.n_factors, min(n_users, n_items) - 1)
            self.svd_model = TruncatedSVD(n_components=max_components, random_state=42)
            self.user_factors_svd = self.svd_model.fit_transform(self.interaction_matrix)
            self.item_factors_svd = self.svd_model.components_.T
            
            # Normalize factors
            self.user_factors_svd = normalize(self.user_factors_svd, axis=1)
            self.item_factors_svd = normalize(self.item_factors_svd, axis=1)
            print("SVD model trained successfully")
        except Exception as e:
            print(f"SVD training failed: {e}")
            self.svd_model = None
        
        # Enhanced popularity scores
        self.popularity_scores = {}
        for item_id in train_df['item_id'].unique():
            item_data = train_df[train_df['item_id'] == item_id]
            base_pop = len(item_data)
            
            # Temporal and quality boosts
            if self.temporal_weights:
                temporal_boost = sum([self.temporal_weights.get((row['user_id'], item_id), 1.0) 
                                    for _, row in item_data.iterrows()])
                popularity = base_pop + 0.5 * temporal_boost
            else:
                popularity = base_pop
            
            # Quality boost from metadata
            if item_id in self.item_features and 'quality' in self.item_features[item_id]:
                quality = self.item_features[item_id]['quality']
                if quality == 'excellent':
                    popularity *= 1.2
                elif quality == 'good':
                    popularity *= 1.1
            
            self.popularity_scores[item_id] = popularity
        
        # Create diversified popular items list
        self.top_popular_items = self._create_diversified_popular_list()
        
        print("Robust collaborative filtering models fitted successfully!")
    
    def _create_diversified_popular_list(self, target_size=200):
        """Create a diversified list of popular items"""
        popular_items = sorted(self.popularity_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not self.item_features:
            return [item[0] for item in popular_items[:target_size]]
        
        diversified = []
        category_counts = defaultdict(int)
        max_per_category = max(10, target_size // 15)  # Allow more per category for sparse data
        
        # First pass: diversified by category
        for item_id, score in popular_items:
            if len(diversified) >= target_size:
                break
                
            if item_id in self.item_features and 'category' in self.item_features[item_id]:
                category = self.item_features[item_id]['category']
                if category_counts[category] < max_per_category:
                    diversified.append(item_id)
                    category_counts[category] += 1
            else:
                diversified.append(item_id)
        
        # Second pass: fill remaining slots
        for item_id, score in popular_items:
            if len(diversified) >= target_size:
                break
            if item_id not in diversified:
                diversified.append(item_id)
        
        return diversified[:target_size]
    
    def get_robust_model_recommendations(self, model, user_id, top_k=50):
        """Get recommendations with robust error handling"""
        if not model or user_id not in self.user_encoder.classes_:
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
                # ROBUST INDEX CHECKING
                if 0 <= idx < len(self.item_encoder.classes_):
                    item_id = self.item_encoder.classes_[idx]
                    
                    # Apply bias correction
                    bias_corrected_score = score + self.item_bias.get(item_id, 0)
                    recommendations.append((item_id, bias_corrected_score))
            
            return recommendations[:top_k]  # Ensure we don't exceed top_k
            
        except Exception as e:
            print(f"Model recommendation error for user {user_id}: {e}")
            return []
    
    def get_svd_recommendations(self, user_id, top_k=50):
        """SVD recommendations with robust error handling"""
        if not self.svd_model or user_id not in self.user_encoder.classes_:
            return []
        
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
            user_items = set(self.interaction_matrix[user_idx].indices)
            
            # Compute scores
            user_vector = self.user_factors_svd[user_idx]
            scores = user_vector @ self.item_factors_svd.T
            
            # Apply bias correction
            user_bias_val = self.user_bias.get(user_id, 0)
            
            recommendations = []
            for item_idx, score in enumerate(scores):
                if item_idx not in user_items and 0 <= item_idx < len(self.item_encoder.classes_):
                    item_id = self.item_encoder.classes_[item_idx]
                    
                    corrected_score = score + user_bias_val + self.item_bias.get(item_id, 0)
                    recommendations.append((item_id, corrected_score))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            print(f"SVD recommendation error for user {user_id}: {e}")
            return []
    
    def get_cooccurrence_recommendations(self, user_id, top_k=30):
        """Get recommendations based on item co-occurrence"""
        if user_id not in self.user_encoder.classes_:
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        user_items = self.interaction_matrix[user_idx].indices
        
        if len(user_items) == 0:
            return []
        
        # Get user's actual item IDs
        user_item_ids = [self.item_encoder.classes_[idx] for idx in user_items 
                        if idx < len(self.item_encoder.classes_)]
        
        # Calculate co-occurrence scores
        cooccurrence_scores = defaultdict(float)
        
        for item_id in user_item_ids:
            if item_id in self.item_cooccurrence:
                for related_item, count in self.item_cooccurrence[item_id].items():
                    if related_item not in user_item_ids:
                        cooccurrence_scores[related_item] += count
        
        # Sort and return
        recommendations = sorted(cooccurrence_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
    
    def get_content_recommendations(self, user_id, top_k=40):
        """Enhanced content-based recommendations"""
        if not self.item_features or user_id not in self.user_encoder.classes_:
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        user_items = self.interaction_matrix[user_idx].indices
        
        user_item_ids = [self.item_encoder.classes_[idx] for idx in user_items 
                        if idx < len(self.item_encoder.classes_)]
        
        if not user_item_ids:
            return []
        
        # Build user profile
        user_categories = defaultdict(float)
        user_price_ranges = defaultdict(float)
        user_qualities = defaultdict(float)
        
        for item_id in user_item_ids:
            if item_id in self.item_features:
                features = self.item_features[item_id]
                
                # Weight by item popularity (inversely)
                item_weight = 1.0 / np.log(2 + self.popularity_scores.get(item_id, 1))
                
                if 'category' in features:
                    user_categories[features['category']] += item_weight
                if 'price_range' in features:
                    user_price_ranges[features['price_range']] += item_weight
                if 'quality' in features:
                    user_qualities[features['quality']] += item_weight
        
        # Normalize preferences
        total_weight = sum(user_categories.values()) + sum(user_price_ranges.values()) + sum(user_qualities.values())
        if total_weight == 0:
            return []
        
        for category in user_categories:
            user_categories[category] /= total_weight
        for price_range in user_price_ranges:
            user_price_ranges[price_range] /= total_weight
        for quality in user_qualities:
            user_qualities[quality] /= total_weight
        
        # Score candidate items
        content_scores = []
        for item_id, features in self.item_features.items():
            if item_id not in user_item_ids:
                score = 0.0
                
                # Category match (highest weight)
                if 'category' in features and features['category'] in user_categories:
                    score += 0.6 * user_categories[features['category']]
                
                # Price range match
                if 'price_range' in features and features['price_range'] in user_price_ranges:
                    score += 0.25 * user_price_ranges[features['price_range']]
                
                # Quality match
                if 'quality' in features and features['quality'] in user_qualities:
                    score += 0.15 * user_qualities[features['quality']]
                
                if score > 0:
                    content_scores.append((item_id, score))
        
        content_scores.sort(key=lambda x: x[1], reverse=True)
        return content_scores[:top_k]
    
    def get_fallback_recommendations(self, user_id, top_k=10):
        """Aggressive fallback recommendations for sparse data"""
        fallback_items = []
        
        # Strategy 1: User's category preferences + popular items in those categories
        if user_id in self.user_encoder.classes_:
            user_idx = self.user_encoder.transform([user_id])[0]
            user_items = self.interaction_matrix[user_idx].indices
            user_item_ids = [self.item_encoder.classes_[idx] for idx in user_items 
                            if idx < len(self.item_encoder.classes_)]
            
            # Find user's preferred categories
            user_categories = set()
            for item_id in user_item_ids:
                if item_id in self.item_features and 'category' in self.item_features[item_id]:
                    user_categories.add(self.item_features[item_id]['category'])
            
            # Add popular items from preferred categories
            for category in user_categories:
                category_items = self.category_items.get(category, [])
                # Sort by popularity and take top items from this category
                cat_pop_items = [(item, self.popularity_scores.get(item, 0)) for item in category_items 
                               if item not in user_item_ids]
                cat_pop_items.sort(key=lambda x: x[1], reverse=True)
                fallback_items.extend([item for item, _ in cat_pop_items[:5]])
        
        # Strategy 2: Add diversified popular items
        for item in self.top_popular_items:
            if item not in fallback_items:
                fallback_items.append(item)
            if len(fallback_items) >= top_k * 2:
                break
        
        # Strategy 3: Random high-quality items
        high_quality_items = self.quality_items.get('excellent', []) + self.quality_items.get('good', [])
        random.shuffle(high_quality_items)
        for item in high_quality_items:
            if item not in fallback_items:
                fallback_items.append(item)
            if len(fallback_items) >= top_k * 3:
                break
        
        return [(item, 1.0) for item in fallback_items[:top_k * 2]]
    
    def get_ensemble_recommendations(self, user_id, top_k=10):
        """Robust ensemble with aggressive sparse data handling"""
        
        # Get recommendations from all available models
        als_recs = dict(self.get_robust_model_recommendations(self.als_model, user_id, top_k * 4))
        bpr_recs = dict(self.get_robust_model_recommendations(self.bpr_model, user_id, top_k * 4))
        svd_recs = dict(self.get_svd_recommendations(user_id, top_k * 4))
        cooccur_recs = dict(self.get_cooccurrence_recommendations(user_id, top_k * 3))
        content_recs = dict(self.get_content_recommendations(user_id, top_k * 3))
        fallback_recs = dict(self.get_fallback_recommendations(user_id, top_k * 2))
        
        # Count successful models
        successful_models = sum([bool(als_recs), bool(bpr_recs), bool(svd_recs), 
                               bool(cooccur_recs), bool(content_recs)])
        
        print(f"User {user_id}: {successful_models} successful models, fallback has {len(fallback_recs)} items")
        
        # Determine user interaction level
        if user_id in self.user_encoder.classes_:
            user_idx = self.user_encoder.transform([user_id])[0]
            n_interactions = len(self.interaction_matrix[user_idx].indices)
        else:
            n_interactions = 0
        
        # AGGRESSIVE WEIGHTING FOR SPARSE DATA
        if n_interactions <= 1:
            # Very sparse users - rely heavily on content and popularity
            weights = {
                'als': 0.15,
                'bpr': 0.15,
                'svd': 0.10,
                'cooccur': 0.10,
                'content': 0.25,
                'fallback': 0.25  # High fallback weight
            }
        elif n_interactions <= 3:
            # Somewhat sparse users
            weights = {
                'als': 0.20,
                'bpr': 0.20,
                'svd': 0.15,
                'cooccur': 0.15,
                'content': 0.20,
                'fallback': 0.10
            }
        else:
            # Users with more data
            weights = {
                'als': 0.30,
                'bpr': 0.30,
                'svd': 0.20,
                'cooccur': 0.15,
                'content': 0.04,
                'fallback': 0.01
            }
        
        # If too few successful models, boost fallback
        if successful_models < 3:
            weights['fallback'] *= 2
            # Redistribute other weights
            other_weight = 1.0 - weights['fallback']
            for key in weights:
                if key != 'fallback':
                    weights[key] = weights[key] * other_weight / sum([weights[k] for k in weights if k != 'fallback'])
        
        # Normalize and combine scores
        def robust_normalize_scores(score_dict):
            if not score_dict:
                return {}
            
            scores = list(score_dict.values())
            if len(scores) <= 1:
                return {k: 1.0 for k in score_dict}
            
            # Use robust statistics (percentile-based)
            p25, p75 = np.percentile(scores, [25, 75])
            iqr = p75 - p25
            
            if iqr == 0:
                return {k: 1.0 for k in score_dict}
            
            # Robust normalization
            normalized = {}
            for k, v in score_dict.items():
                normalized[k] = max(0, (v - p25) / iqr)
            
            return normalized
        
        # Normalize all scores
        als_recs = robust_normalize_scores(als_recs)
        bpr_recs = robust_normalize_scores(bpr_recs)
        svd_recs = robust_normalize_scores(svd_recs)
        cooccur_recs = robust_normalize_scores(cooccur_recs)
        content_recs = robust_normalize_scores(content_recs)
        fallback_recs = robust_normalize_scores(fallback_recs)
        
        # Combine scores
        final_scores = defaultdict(float)
        
        for item_id, score in als_recs.items():
            final_scores[item_id] += weights['als'] * score
        
        for item_id, score in bpr_recs.items():
            final_scores[item_id] += weights['bpr'] * score
        
        for item_id, score in svd_recs.items():
            final_scores[item_id] += weights['svd'] * score
        
        for item_id, score in cooccur_recs.items():
            final_scores[item_id] += weights['cooccur'] * score
        
        for item_id, score in content_recs.items():
            final_scores[item_id] += weights['content'] * score
        
        for item_id, score in fallback_recs.items():
            final_scores[item_id] += weights['fallback'] * score
        
        # Sort and get recommendations
        recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        rec_items = [item_id for item_id, score in recommendations]
        
        # Apply final diversification
        diversified_recs = self._final_diversification(rec_items, top_k)
        
        # Ensure we have enough recommendations
        while len(diversified_recs) < top_k:
            for item in self.top_popular_items:
                if item not in diversified_recs:
                    diversified_recs.append(item)
                    break
            else:
                break  # No more items available
        
        print(f"Final recommendations for user {user_id}: {len(diversified_recs)} items")
        return diversified_recs[:top_k]
    
    def _final_diversification(self, item_list, target_count):
        """Apply final diversification"""
        if not self.item_features:
            return item_list[:target_count]
        
        diversified = []
        category_counts = defaultdict(int)
        max_per_category = max(2, target_count // 4)  # More relaxed for sparse data
        
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
        """Fit all models with robust error handling"""
        print("Starting robust sparse recommendation system training...")
        
        # Build co-occurrence matrix
        self.build_cooccurrence_matrix(train_df)
        
        # Build temporal features
        self.build_temporal_features(train_df)
        
        # Compute bias terms
        self.compute_bias_terms(train_df)
        
        # Process metadata
        if item_meta_df is not None:
            self.preprocess_metadata_efficiently(item_meta_df)
        
        # Fit models
        self.fit_robust_models(train_df)
        
        print("Robust sparse recommendation system training completed!")
    
    def recommend(self, user_id, top_k=10):
        """Main recommendation function"""
        return self.get_ensemble_recommendations(user_id, top_k)

def main():
    """Main function"""
    print("=" * 70)
    print("ROBUST SPARSE DATA RECOMMENDATION SYSTEM")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Load metadata
    try:
        item_meta_df = pd.read_csv('item_meta.csv', nrows=150000)  # Load even more
        print(f"Item metadata loaded: {item_meta_df.shape}")
    except:
        item_meta_df = None
        print("Item metadata not available")
    
    # Initialize and train
    rec_sys = RobustSparseRecommendationSystem(n_factors=150, alpha=60)
    rec_sys.fit(train_df, item_meta_df)
    
    # Generate predictions
    print("\nGenerating robust predictions...")
    
    # Load submission format
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
    
    # Generate predictions with progress tracking
    predictions = {}
    test_users = sample_submission['user_id'].unique()
    
    failed_predictions = 0
    for user_id in tqdm(test_users, desc="Generating robust recommendations"):
        try:
            recommendations = rec_sys.recommend(user_id, top_k=10)
            if len(recommendations) < 10:
                # Aggressive fallback
                for item in rec_sys.top_popular_items:
                    if item not in recommendations:
                        recommendations.append(item)
                    if len(recommendations) == 10:
                        break
            predictions[user_id] = recommendations[:10]
        except Exception as e:
            print(f"Failed to generate recommendations for user {user_id}: {e}")
            predictions[user_id] = rec_sys.top_popular_items[:10]
            failed_predictions += 1
    
    print(f"Failed predictions: {failed_predictions}")
    
    # Create submission
    print("\nCreating submission file...")
    submission_data = []
    
    for idx, row in sample_submission.iterrows():
        user_id = row['user_id']
        recs = predictions.get(user_id, rec_sys.top_popular_items[:10])
        
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
    submission_file = 'robust_sparse_recommendation_submission.csv'
    submission_df.to_csv(submission_file, index=False)
    print(f"\nSubmission saved to: {submission_file}")
    
    print("\nRobust sparse recommendation system completed!")
    print("Key features:")
    print("- Robust index bounds checking")
    print("- Co-occurrence based recommendations")
    print("- Aggressive fallback strategies for sparse users")
    print("- Enhanced temporal and bias modeling")
    print("- Multiple model ensemble with adaptive weights")
    print("- Content-based recommendations with metadata")

if __name__ == "__main__":
    main() 