import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
from tqdm import tqdm
import pickle
import random
from datetime import datetime
import math

warnings.filterwarnings('ignore')

class AdvancedRecommendationSystem:
    """Advanced recommendation system with multiple sophisticated techniques"""
    
    def __init__(self, n_factors=50, alpha=40, reg_lambda=0.1):
        # Core components
        self.user_profiles = {}
        self.item_profiles = {}
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Matrix factorization components
        self.n_factors = n_factors
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.user_factors = None
        self.item_factors = None
        
        # Advanced features
        self.content_features = {}
        self.item_categories = {}
        self.user_preferences = {}
        self.temporal_patterns = {}
        self.popularity_scores = {}
        self.category_popularity = {}
        
        # Diversity and exploration
        self.item_diversity_matrix = None
        self.exploration_rate = 0.15
        
        # Statistics
        self.stats = {}
    
    def build_content_features(self, item_meta_df):
        """Build sophisticated content features from metadata"""
        print("Building advanced content features...")
        
        if item_meta_df is None or len(item_meta_df) == 0:
            return
        
        # Process text features for content-based filtering
        item_descriptions = []
        item_ids = []
        
        for _, row in item_meta_df.iterrows():
            item_id = str(row['item_id'])
            item_ids.append(item_id)
            
            # Combine all textual features
            text_features = []
            features_dict = {}
            
            for col in item_meta_df.columns:
                if col != 'item_id' and pd.notna(row[col]):
                    value = str(row[col]).lower().strip()
                    text_features.append(value)
                    
                    # Categorize features
                    if 'category' in col.lower():
                        features_dict['category'] = value
                        self.item_categories[item_id] = value
                    elif 'brand' in col.lower():
                        features_dict['brand'] = value
                    elif 'price' in col.lower():
                        try:
                            price = float(value)
                            if price < 20:
                                features_dict['price_range'] = 'low'
                            elif price < 100:
                                features_dict['price_range'] = 'medium'
                            else:
                                features_dict['price_range'] = 'high'
                        except:
                            features_dict['price_range'] = 'unknown'
                    elif any(keyword in col.lower() for keyword in ['description', 'title', 'name']):
                        features_dict['description'] = value
            
            item_descriptions.append(' '.join(text_features))
            self.content_features[item_id] = features_dict
        
        # Build TF-IDF vectors for content similarity
        if item_descriptions:
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
            try:
                tfidf_matrix = tfidf.fit_transform(item_descriptions)
                self.content_similarity = cosine_similarity(tfidf_matrix)
                self.content_item_ids = item_ids
            except:
                self.content_similarity = None
                self.content_item_ids = []
    
    def build_temporal_patterns(self, train_df):
        """Analyze temporal patterns in user behavior"""
        print("Building temporal patterns...")
        
        # If timestamp column exists, use it; otherwise simulate
        if 'timestamp' in train_df.columns:
            train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
        else:
            # Simulate timestamps based on interaction order
            train_df = train_df.copy()
            train_df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(train_df), freq='H')
        
        # Analyze temporal patterns
        for user_id in self.user_profiles.keys():
            user_data = train_df[train_df['user_id'] == user_id].sort_values('timestamp')
            
            if len(user_data) > 1:
                # Calculate time between interactions
                time_diffs = user_data['timestamp'].diff().dt.total_seconds() / 3600  # hours
                
                # Store temporal preferences
                self.temporal_patterns[user_id] = {
                    'avg_time_between_interactions': time_diffs.median(),
                    'most_active_hour': user_data['timestamp'].dt.hour.mode().iloc[0] if len(user_data) > 0 else 12,
                    'session_length': len(user_data),
                    'recent_items': list(user_data['item_id'].tail(5))
                }
    
    def fit_matrix_factorization(self, train_df):
        """Fit implicit matrix factorization model"""
        print("Fitting matrix factorization model...")
        
        # Create user-item interaction matrix
        user_ids = self.user_encoder.fit_transform(train_df['user_id'])
        item_ids = self.item_encoder.fit_transform(train_df['item_id'])
        
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        # Create implicit feedback matrix (higher weights for more interactions)
        interaction_matrix = np.zeros((n_users, n_items))
        
        for user_idx, item_idx in zip(user_ids, item_ids):
            interaction_matrix[user_idx, item_idx] += 1
        
        # Apply confidence weighting (1 + alpha * r_ui)
        confidence_matrix = 1 + self.alpha * interaction_matrix
        
        # Use Alternating Least Squares for matrix factorization
        self.user_factors = np.random.normal(0.1, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0.1, 0.1, (n_items, self.n_factors))
        
        # Simplified ALS iterations
        for iteration in range(10):
            # Update user factors
            for u in range(n_users):
                Cu = np.diag(confidence_matrix[u, :])
                pu = interaction_matrix[u, :] > 0
                
                # Solve: (Y^T * Cu * Y + λI) * xu = Y^T * Cu * pu
                YTCu = self.item_factors.T @ Cu
                YTCuY = YTCu @ self.item_factors
                regularization = self.reg_lambda * np.eye(self.n_factors)
                
                try:
                    lhs = YTCuY + regularization
                    rhs = YTCu @ pu.astype(float)
                    self.user_factors[u, :] = np.linalg.solve(lhs, rhs)
                except:
                    pass  # Keep previous values if solve fails
            
            # Update item factors
            for i in range(n_items):
                Ci = np.diag(confidence_matrix[:, i])
                pi = interaction_matrix[:, i] > 0
                
                # Solve: (X^T * Ci * X + λI) * yi = X^T * Ci * pi
                XTCi = self.user_factors.T @ Ci
                XTCiX = XTCi @ self.user_factors
                regularization = self.reg_lambda * np.eye(self.n_factors)
                
                try:
                    lhs = XTCiX + regularization
                    rhs = XTCi @ pi.astype(float)
                    self.item_factors[i, :] = np.linalg.solve(lhs, rhs)
                except:
                    pass  # Keep previous values if solve fails
        
        print(f"Matrix factorization completed with {self.n_factors} factors")
    
    def compute_advanced_popularity(self, train_df):
        """Compute sophisticated popularity scores with diversity"""
        print("Computing advanced popularity scores...")
        
        # Basic popularity
        item_counts = train_df['item_id'].value_counts()
        
        # Apply log scaling to reduce popularity bias
        log_popularity = np.log1p(item_counts)
        
        # Add recency boost (more recent interactions get higher scores)
        if 'timestamp' in train_df.columns:
            recent_interactions = train_df[train_df['timestamp'] > train_df['timestamp'].quantile(0.8)]
            recent_counts = recent_interactions['item_id'].value_counts()
            recency_boost = recent_counts / recent_counts.max() * 0.3
        else:
            # Use last 20% of data as "recent"
            recent_size = int(len(train_df) * 0.2)
            recent_interactions = train_df.tail(recent_size)
            recent_counts = recent_interactions['item_id'].value_counts()
            recency_boost = recent_counts / recent_counts.max() * 0.3 if len(recent_counts) > 0 else pd.Series()
        
        # Combine popularity and recency
        for item in item_counts.index:
            base_score = log_popularity[item]
            recency_score = recency_boost.get(item, 0)
            self.popularity_scores[item] = base_score + recency_score
        
        # Sort by final scores
        sorted_items = sorted(self.popularity_scores.items(), key=lambda x: x[1], reverse=True)
        self.popular_items = [item for item, score in sorted_items]
    
    def build_user_preferences(self, train_df):
        """Build detailed user preference profiles"""
        print("Building user preference profiles...")
        
        for user_id, user_items in self.user_profiles.items():
            preferences = {
                'categories': [],
                'brands': [],
                'price_ranges': [],
                'diversity_score': 0,
                'exploration_tendency': 0
            }
            
            # Analyze content preferences
            for item_id in user_items:
                if item_id in self.content_features:
                    features = self.content_features[item_id]
                    if 'category' in features:
                        preferences['categories'].append(features['category'])
                    if 'brand' in features:
                        preferences['brands'].append(features['brand'])
                    if 'price_range' in features:
                        preferences['price_ranges'].append(features['price_range'])
            
            # Calculate diversity score (how diverse are user's past interactions)
            if len(set(preferences['categories'])) > 0:
                preferences['diversity_score'] = len(set(preferences['categories'])) / len(preferences['categories'])
            
            # Calculate exploration tendency (preference for less popular items)
            item_popularities = [self.popularity_scores.get(item, 0) for item in user_items]
            if item_popularities:
                avg_popularity = np.mean(item_popularities)
                max_popularity = max(self.popularity_scores.values()) if self.popularity_scores else 1
                preferences['exploration_tendency'] = 1 - (avg_popularity / max_popularity)
            
            self.user_preferences[user_id] = preferences
    
    def fit(self, train_df, item_meta_df=None):
        """Fit the advanced recommendation system"""
        print("=== Training Advanced Recommendation System ===")
        
        # 1. Basic preprocessing
        print("1. Preprocessing...")
        train_df = train_df.dropna(subset=['user_id', 'item_id']).copy()
        train_df['user_id'] = train_df['user_id'].astype(str)
        train_df['item_id'] = train_df['item_id'].astype(str)
        train_df = train_df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
        
        print(f"Training data: {len(train_df):,} interactions")
        print(f"Users: {train_df['user_id'].nunique():,}")
        print(f"Items: {train_df['item_id'].nunique():,}")
        
        # 2. Build user profiles
        print("2. Building user profiles...")
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building profiles"):
            user_id = row['user_id']
            item_id = row['item_id']
            
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = set()
            self.user_profiles[user_id].add(item_id)
        
        # 3. Build content features
        if item_meta_df is not None:
            self.build_content_features(item_meta_df)
        
        # 4. Build temporal patterns
        self.build_temporal_patterns(train_df)
        
        # 5. Compute advanced popularity
        self.compute_advanced_popularity(train_df)
        
        # 6. Build user preferences
        self.build_user_preferences(train_df)
        
        # 7. Fit matrix factorization
        self.fit_matrix_factorization(train_df)
        
        # 8. Build item-item similarity matrix
        print("8. Building item similarity matrix...")
        self.build_item_similarity(train_df)
        
        self.stats = {
            'n_users': len(self.user_profiles),
            'n_items': train_df['item_id'].nunique(),
            'n_interactions': len(train_df),
            'avg_items_per_user': np.mean([len(items) for items in self.user_profiles.values()]),
            'content_items': len(self.content_features),
            'n_factors': self.n_factors
        }
        
        print(f"Training completed!")
        print(f"Stats: {self.stats}")
        return self
    
    def build_item_similarity(self, train_df):
        """Build advanced item-item similarity matrix"""
        print("Building advanced item similarity...")
        
        # Co-occurrence based similarity with advanced weighting
        item_cooccurrence = defaultdict(lambda: defaultdict(float))
        user_item_counts = defaultdict(int)
        
        # Count co-occurrences with user frequency weighting
        for user_items in tqdm(self.user_profiles.values(), desc="Computing similarities"):
            user_items_list = list(user_items)
            user_weight = 1.0 / math.log2(len(user_items_list) + 1)  # Downweight users with many items
            
            for item in user_items_list:
                user_item_counts[item] += 1
            
            for i, item1 in enumerate(user_items_list):
                for item2 in user_items_list[i+1:]:
                    item_cooccurrence[item1][item2] += user_weight
                    item_cooccurrence[item2][item1] += user_weight
        
        # Apply TF-IDF style weighting and Jaccard similarity
        total_users = len(self.user_profiles)
        self.item_similarity = {}
        
        for item1, related_items in item_cooccurrence.items():
            self.item_similarity[item1] = {}
            
            for item2, cooc_count in related_items.items():
                # Jaccard-like similarity
                item1_users = user_item_counts[item1]
                item2_users = user_item_counts[item2]
                
                # Jaccard similarity with co-occurrence weighting
                union_size = item1_users + item2_users - cooc_count
                jaccard_sim = cooc_count / union_size if union_size > 0 else 0
                
                # TF-IDF style weighting
                idf_weight = np.log(total_users / (user_item_counts[item2] + 1))
                
                final_similarity = jaccard_sim * idf_weight * cooc_count
                
                if final_similarity > 0:
                    self.item_similarity[item1][item2] = final_similarity
    
    def get_matrix_factorization_recommendations(self, user_id, n_recommendations=20):
        """Get recommendations from matrix factorization"""
        if self.user_factors is None or self.item_factors is None:
            return []
        
        try:
            user_idx = list(self.user_encoder.classes_).index(user_id)
            user_vector = self.user_factors[user_idx]
            
            # Compute scores for all items
            scores = self.item_factors @ user_vector
            
            # Get top items
            item_indices = np.argsort(scores)[::-1]
            
            recommendations = []
            user_items = self.user_profiles.get(user_id, set())
            
            for item_idx in item_indices:
                item_id = self.item_encoder.classes_[item_idx]
                if item_id not in user_items and len(recommendations) < n_recommendations:
                    recommendations.append((item_id, scores[item_idx]))
            
            return recommendations
        except:
            return []
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=15):
        """Get collaborative filtering recommendations with advanced scoring"""
        user_items = self.user_profiles.get(user_id, set())
        if not user_items:
            return []
        
        candidate_scores = defaultdict(float)
        
        # Score based on item similarity
        for item in user_items:
            if item in self.item_similarity:
                for related_item, similarity in self.item_similarity[item].items():
                    if related_item not in user_items:
                        candidate_scores[related_item] += similarity
        
        # Apply user preference weighting
        user_prefs = self.user_preferences.get(user_id, {})
        exploration_tendency = user_prefs.get('exploration_tendency', 0.5)
        
        # Boost scores for items matching user preferences
        for item_id, score in candidate_scores.items():
            if item_id in self.content_features:
                features = self.content_features[item_id]
                
                # Category preference boost
                if 'category' in features and features['category'] in user_prefs.get('categories', []):
                    category_freq = user_prefs['categories'].count(features['category'])
                    category_boost = category_freq / len(user_prefs['categories']) if user_prefs['categories'] else 0
                    score *= (1 + category_boost * 0.5)
                
                # Exploration vs exploitation balance
                item_popularity = self.popularity_scores.get(item_id, 0)
                max_popularity = max(self.popularity_scores.values()) if self.popularity_scores else 1
                popularity_factor = item_popularity / max_popularity
                
                # Exploration-prone users get boost for less popular items
                exploration_boost = (1 - popularity_factor) * exploration_tendency * 0.3
                score *= (1 + exploration_boost)
                
                candidate_scores[item_id] = score
        
        # Sort and return top recommendations
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [(item, score) for item, score in sorted_candidates[:n_recommendations]]
    
    def get_content_based_recommendations(self, user_id, n_recommendations=10):
        """Get content-based recommendations using TF-IDF similarity"""
        if self.content_similarity is None or not self.content_item_ids:
            return []
        
        user_items = self.user_profiles.get(user_id, set())
        if not user_items:
            return []
        
        # Find user's items in content matrix
        user_item_indices = []
        for item_id in user_items:
            try:
                idx = self.content_item_ids.index(item_id)
                user_item_indices.append(idx)
            except ValueError:
                continue
        
        if not user_item_indices:
            return []
        
        # Compute average content profile for user
        user_profile = np.mean(self.content_similarity[user_item_indices], axis=0)
        
        # Score all items
        candidate_scores = []
        for i, item_id in enumerate(self.content_item_ids):
            if item_id not in user_items:
                similarity = user_profile[i]
                candidate_scores.append((item_id, similarity))
        
        # Sort and return top recommendations
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return candidate_scores[:n_recommendations]
    
    def get_diversified_recommendations(self, candidates, user_id, n_recommendations=10):
        """Apply diversification to candidate recommendations"""
        if not candidates:
            return []
        
        user_prefs = self.user_preferences.get(user_id, {})
        diversity_preference = user_prefs.get('diversity_score', 0.5)
        
        # Sort candidates by score
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        
        # Greedy diversification
        selected = []
        selected_categories = set()
        
        for item_id, score in candidates:
            if len(selected) >= n_recommendations:
                break
            
            # Check category diversity
            item_category = None
            if item_id in self.content_features:
                item_category = self.content_features[item_id].get('category')
            
            # Decide whether to add based on diversity preference
            add_item = True
            if item_category and item_category in selected_categories:
                # If category already selected, add with probability based on diversity preference
                if diversity_preference > 0.7:  # High diversity users
                    add_item = random.random() > 0.6  # 40% chance to add duplicate category
                else:  # Lower diversity users
                    add_item = random.random() > 0.3  # 70% chance to add duplicate category
            
            if add_item:
                selected.append((item_id, score))
                if item_category:
                    selected_categories.add(item_category)
        
        return selected
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate final recommendations using ensemble approach"""
        user_id = str(user_id)
        
        # Check if user exists
        if user_id not in self.user_profiles:
            # Cold start: return diversified popular items
            return self.get_cold_start_recommendations(n_recommendations)
        
        all_candidates = []
        
        # 1. Matrix factorization recommendations (40% weight)
        mf_recs = self.get_matrix_factorization_recommendations(user_id, n_recommendations * 2)
        for item, score in mf_recs:
            all_candidates.append((item, score * 0.4, 'mf'))
        
        # 2. Collaborative filtering recommendations (35% weight)
        collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
        for item, score in collab_recs:
            all_candidates.append((item, score * 0.35, 'collab'))
        
        # 3. Content-based recommendations (20% weight)
        content_recs = self.get_content_based_recommendations(user_id, n_recommendations)
        for item, score in content_recs:
            all_candidates.append((item, score * 0.2, 'content'))
        
        # 4. Add some popular items with exploration (5% weight)
        exploration_count = max(1, int(n_recommendations * 0.1))
        user_items = self.user_profiles[user_id]
        
        for item in self.popular_items[:50]:  # Top 50 popular items
            if item not in user_items:
                popularity_score = self.popularity_scores.get(item, 0)
                all_candidates.append((item, popularity_score * 0.05, 'popular'))
                exploration_count -= 1
                if exploration_count <= 0:
                    break
        
        # Combine scores for same items
        item_scores = defaultdict(float)
        for item, score, source in all_candidates:
            item_scores[item] += score
        
        # Convert to list of tuples
        final_candidates = [(item, score) for item, score in item_scores.items()]
        
        # Apply diversification
        diversified_recs = self.get_diversified_recommendations(final_candidates, user_id, n_recommendations)
        
        return [item for item, score in diversified_recs]
    
    def get_cold_start_recommendations(self, n_recommendations=10):
        """Get recommendations for new users"""
        # Mix of popular items and diverse items
        recommendations = []
        used_categories = set()
        
        for item in self.popular_items:
            if len(recommendations) >= n_recommendations:
                break
            
            # Try to diversify by category
            item_category = None
            if item in self.content_features:
                item_category = self.content_features[item].get('category')
            
            # Add item if from new category or if we need more items
            if item_category not in used_categories or len(recommendations) >= n_recommendations * 0.7:
                recommendations.append(item)
                if item_category:
                    used_categories.add(item_category)
        
        return recommendations
    
    def evaluate_recall(self, train_df):
        """Evaluate the system using recall metrics"""
        print("Evaluating advanced recommender...")
        
        test_interactions = {}
        original_profiles = {}
        
        for user_id, user_items in self.user_profiles.items():
            user_items_list = list(user_items)
            if len(user_items_list) >= 3:  # Need at least 3 items
                original_profiles[user_id] = user_items.copy()
                
                # Hold out last 2 items for testing
                test_items = user_items_list[-2:]
                train_items = user_items_list[:-2]
                
                test_interactions[user_id] = test_items
                self.user_profiles[user_id] = set(train_items)
        
        print(f"Testing on {len(test_interactions)} users")
        
        hits_at_5 = 0
        hits_at_10 = 0
        total_users = len(test_interactions)
        
        for user_id, true_items in tqdm(test_interactions.items(), desc="Evaluating"):
            recommendations = self.recommend(user_id, 10)
            
            # Check if any true item is in recommendations
            found_in_5 = any(item in recommendations[:5] for item in true_items)
            found_in_10 = any(item in recommendations[:10] for item in true_items)
            
            if found_in_5:
                hits_at_5 += 1
                hits_at_10 += 1
            elif found_in_10:
                hits_at_10 += 1
        
        # Restore original profiles
        self.user_profiles.update(original_profiles)
        
        recall_at_5 = hits_at_5 / total_users if total_users > 0 else 0
        recall_at_10 = hits_at_10 / total_users if total_users > 0 else 0
        
        print(f"Advanced Recall@5: {recall_at_5:.4f} ({hits_at_5}/{total_users})")
        print(f"Advanced Recall@10: {recall_at_10:.4f} ({hits_at_10}/{total_users})")
        
        return recall_at_10

def main():
    print("=== Advanced Recommendation System ===")
    
    # Load data
    train_df = pd.read_csv('train.csv')
    
    # Load item metadata
    try:
        item_meta_df = pd.read_csv('item_meta.csv')
        print(f"Loaded item metadata: {len(item_meta_df)} items")
    except:
        item_meta_df = None
        print("No item metadata available")
    
    # Train advanced recommender
    recommender = AdvancedRecommendationSystem(n_factors=100, alpha=50, reg_lambda=0.01)
    recommender.fit(train_df, item_meta_df)
    
    # Evaluate
    recall = recommender.evaluate_recall(train_df)
    
    # Generate submission
    print("\nGenerating advanced recommendations for submission...")
    
    try:
        sample_submission = pd.read_csv('sample_submission.csv')
        print(f"Loaded sample submission with {len(sample_submission)} entries")
        
        submission_data = []
        
        for idx, row in tqdm(sample_submission.iterrows(), total=len(sample_submission), desc="Generating submission"):
            user_id = str(row['user_id'])
            original_id = row['ID']
            
            recommendations = recommender.recommend(user_id, 10)
            
            # Ensure we have 10 recommendations
            while len(recommendations) < 10:
                recommendations.extend(recommender.popular_items)
            recommendations = recommendations[:10]
            
            submission_data.append({
                'ID': original_id,
                'user_id': row['user_id'],
                'item_id': ','.join(recommendations)
            })
        
    except FileNotFoundError:
        print("sample_submission.csv not found")
        return
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('advanced_recommender_submission.csv', index=False)
    
    print(f"Submission shape: {submission_df.shape}")
    print("Submission saved to 'advanced_recommender_submission.csv'")
    
    # Save model
    with open('advanced_recommender_model.pkl', 'wb') as f:
        pickle.dump(recommender, f)
    
    print(f"\nFinal Results:")
    print(f"Advanced Recall@10: {recall:.4f}")
    print("Advanced recommendation system with multiple sophisticated techniques!")

if __name__ == "__main__":
    main() 