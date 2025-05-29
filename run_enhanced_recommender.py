import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from tqdm import tqdm
import pickle
import math
from datetime import datetime

warnings.filterwarnings('ignore')

class EnhancedSparseRecommender:
    """Enhanced recommender system with multiple advanced techniques"""
    
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_profiles = {}
        self.user_profiles_weighted = {}  # Time-weighted profiles
        self.popular_items = []
        self.item_similarity = {}
        self.user_clusters = {}
        self.cluster_profiles = {}
        self.content_features = {}
        self.stats = {}
        
    def calculate_time_weights(self, timestamps):
        """Calculate time-based weights (more recent = higher weight)"""
        if len(timestamps) == 0:
            return []
        
        # Convert to datetime if needed
        timestamps = pd.to_datetime(timestamps, errors='coerce')
        max_time = timestamps.max()
        
        # Calculate days since latest interaction
        time_diffs = (max_time - timestamps).days
        
        # Exponential decay: weight = exp(-decay_rate * days)
        decay_rate = 0.1
        weights = np.exp(-decay_rate * time_diffs)
        
        return weights.fillna(1.0).tolist()  # Default weight 1.0 for missing timestamps
    
    def calculate_jaccard_similarity(self, set1, set2):
        """Calculate Jaccard similarity between two sets"""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    def calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)
    
    def build_content_features(self, item_meta_df):
        """Build rich content features from item metadata"""
        print("Building content features...")
        
        self.content_features = {}
        
        if item_meta_df is None or len(item_meta_df) == 0:
            return
        
        for _, row in item_meta_df.iterrows():
            item_id = str(row['item_id'])
            features = {}
            
            # Extract various content features
            for col in item_meta_df.columns:
                if col != 'item_id' and pd.notna(row[col]):
                    value = str(row[col]).lower().strip()
                    
                    if col == 'category':
                        features['category'] = value
                    elif col == 'brand':
                        features['brand'] = value
                    elif 'price' in col.lower():
                        # Normalize price into ranges
                        try:
                            price = float(value)
                            if price < 10:
                                features['price_range'] = 'low'
                            elif price < 50:
                                features['price_range'] = 'medium'
                            else:
                                features['price_range'] = 'high'
                        except:
                            features['price_range'] = 'unknown'
                    elif 'rating' in col.lower():
                        try:
                            rating = float(value)
                            if rating >= 4.0:
                                features['quality'] = 'high'
                            elif rating >= 3.0:
                                features['quality'] = 'medium'
                            else:
                                features['quality'] = 'low'
                        except:
                            features['quality'] = 'unknown'
            
            self.content_features[item_id] = features
    
    def cluster_users(self, n_clusters=50):
        """Cluster users based on their interaction patterns"""
        print(f"Clustering users into {n_clusters} groups...")
        
        # Create user-item interaction matrix
        all_items = set()
        for items in self.user_profiles.values():
            all_items.update(items)
        
        item_list = list(all_items)
        item_to_idx = {item: idx for idx, item in enumerate(item_list)}
        
        user_vectors = []
        user_ids = []
        
        for user_id, items in self.user_profiles.items():
            if len(items) >= 2:  # Only cluster users with multiple interactions
                vector = [0] * len(item_list)
                for item in items:
                    if item in item_to_idx:
                        vector[item_to_idx[item]] = 1
                user_vectors.append(vector)
                user_ids.append(user_id)
        
        if len(user_vectors) > n_clusters:
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(user_vectors)
            
            # Build cluster profiles
            for user_id, cluster_id in zip(user_ids, cluster_labels):
                self.user_clusters[user_id] = cluster_id
            
            # Build cluster item profiles
            cluster_items = defaultdict(lambda: defaultdict(int))
            for user_id, cluster_id in self.user_clusters.items():
                for item in self.user_profiles[user_id]:
                    cluster_items[cluster_id][item] += 1
            
            # Convert to sorted lists
            for cluster_id, items in cluster_items.items():
                sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
                self.cluster_profiles[cluster_id] = [item for item, count in sorted_items[:50]]
        
        print(f"Clustered {len(user_ids)} users into {len(set(self.user_clusters.values()))} clusters")
    
    def fit(self, train_df, item_meta_df=None):
        """Fit the enhanced recommender system"""
        print("=== Training Enhanced Sparse Data Recommender ===")
        
        # 1. Preprocessing
        print("1. Preprocessing...")
        train_df = train_df.dropna(subset=['user_id', 'item_id']).copy()
        train_df['user_id'] = train_df['user_id'].astype(str)
        train_df['item_id'] = train_df['item_id'].astype(str)
        train_df = train_df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
        
        print(f"Training data: {len(train_df):,} interactions")
        print(f"Users: {train_df['user_id'].nunique():,}")
        print(f"Items: {train_df['item_id'].nunique():,}")
        
        # 2. Build user profiles with time weights
        print("2. Building time-weighted user profiles...")
        user_interactions = defaultdict(list)
        
        for _, row in train_df.iterrows():
            user_interactions[row['user_id']].append({
                'item_id': row['item_id'],
                'timestamp': row.get('timestamp', 0)
            })
        
        for user_id, interactions in tqdm(user_interactions.items(), desc="Building profiles"):
            items = [inter['item_id'] for inter in interactions]
            timestamps = [inter['timestamp'] for inter in interactions]
            
            # Regular profile
            self.user_profiles[user_id] = set(items)
            
            # Time-weighted profile
            weights = self.calculate_time_weights(timestamps)
            weighted_items = defaultdict(float)
            for item, weight in zip(items, weights):
                weighted_items[item] += weight
            self.user_profiles_weighted[user_id] = dict(weighted_items)
        
        # 3. Calculate item popularity with time decay
        print("3. Calculating time-weighted item popularity...")
        item_popularity = defaultdict(float)
        for user_id, weighted_items in self.user_profiles_weighted.items():
            for item, weight in weighted_items.items():
                item_popularity[item] += weight
        
        sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
        self.popular_items = [item for item, score in sorted_items[:200]]
        
        # 4. Build enhanced item similarity
        print("4. Building enhanced item similarity...")
        item_cooccurrence = defaultdict(lambda: defaultdict(float))
        
        for user_id, weighted_items in tqdm(self.user_profiles_weighted.items(), desc="Item similarity"):
            items_list = list(weighted_items.keys())
            for i, item1 in enumerate(items_list):
                for item2 in items_list[i+1:]:
                    # Weight by user's interaction strength
                    weight = (weighted_items[item1] + weighted_items[item2]) / 2
                    item_cooccurrence[item1][item2] += weight
                    item_cooccurrence[item2][item1] += weight
        
        # Normalize similarities
        for item1, related_items in item_cooccurrence.items():
            total_weight = sum(related_items.values())
            if total_weight > 0:
                for item2 in related_items:
                    related_items[item2] /= total_weight
        
        self.item_similarity = dict(item_cooccurrence)
        
        # 5. Build content features
        if item_meta_df is not None:
            self.build_content_features(item_meta_df)
        
        # 6. Cluster users
        self.cluster_users()
        
        self.stats = {
            'n_users': len(self.user_profiles),
            'n_items': train_df['item_id'].nunique(),
            'n_interactions': len(train_df),
            'avg_items_per_user': np.mean([len(items) for items in self.user_profiles.values()]),
            'n_clusters': len(set(self.user_clusters.values()))
        }
        
        print(f"Training completed!")
        print(f"Stats: {self.stats}")
        return self
    
    def get_collaborative_recommendations(self, user_items, user_weights, n_recommendations=20):
        """Enhanced collaborative filtering with multiple similarity measures"""
        candidate_scores = defaultdict(float)
        
        for item in user_items:
            user_item_weight = user_weights.get(item, 1.0)
            
            if item in self.item_similarity:
                for related_item, similarity in self.item_similarity[item].items():
                    if related_item not in user_items:
                        # Weight by user's interaction strength and item similarity
                        score = similarity * user_item_weight
                        candidate_scores[related_item] += score
        
        # Sort by score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:n_recommendations]]
    
    def get_cluster_recommendations(self, user_id, user_items, n_recommendations=10):
        """Get recommendations based on user cluster"""
        if user_id not in self.user_clusters:
            return []
        
        cluster_id = self.user_clusters[user_id]
        if cluster_id not in self.cluster_profiles:
            return []
        
        cluster_items = self.cluster_profiles[cluster_id]
        recommendations = []
        
        for item in cluster_items:
            if item not in user_items and item not in recommendations:
                recommendations.append(item)
            if len(recommendations) >= n_recommendations:
                break
        
        return recommendations
    
    def get_content_recommendations(self, user_items, n_recommendations=10):
        """Enhanced content-based recommendations"""
        if not self.content_features:
            return []
        
        # Analyze user's content preferences
        user_preferences = defaultdict(float)
        total_items = 0
        
        for item in user_items:
            if item in self.content_features:
                total_items += 1
                features = self.content_features[item]
                for feature_type, feature_value in features.items():
                    user_preferences[f"{feature_type}:{feature_value}"] += 1
        
        if total_items == 0:
            return []
        
        # Normalize preferences
        for key in user_preferences:
            user_preferences[key] /= total_items
        
        # Score candidate items
        candidate_scores = defaultdict(float)
        
        for item_id, features in self.content_features.items():
            if item_id not in user_items:
                score = 0
                for feature_type, feature_value in features.items():
                    pref_key = f"{feature_type}:{feature_value}"
                    if pref_key in user_preferences:
                        score += user_preferences[pref_key]
                
                if score > 0:
                    candidate_scores[item_id] = score
        
        # Sort by score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:n_recommendations]]
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate enhanced recommendations using multiple strategies"""
        user_id = str(user_id)
        
        # Get user's interaction history
        if user_id in self.user_profiles:
            user_items = self.user_profiles[user_id]
            user_weights = self.user_profiles_weighted.get(user_id, {})
        else:
            # Cold start - return diverse popular items
            return self.popular_items[:n_recommendations]
        
        all_recommendations = []
        
        # 1. Collaborative filtering (40% weight)
        collab_recs = self.get_collaborative_recommendations(user_items, user_weights, n_recommendations)
        all_recommendations.extend(collab_recs[:int(n_recommendations * 0.4)])
        
        # 2. Cluster-based recommendations (25% weight)
        cluster_recs = self.get_cluster_recommendations(user_id, user_items, n_recommendations)
        for item in cluster_recs:
            if item not in all_recommendations:
                all_recommendations.append(item)
            if len(all_recommendations) >= int(n_recommendations * 0.65):
                break
        
        # 3. Content-based recommendations (20% weight)
        content_recs = self.get_content_recommendations(user_items, n_recommendations)
        for item in content_recs:
            if item not in all_recommendations:
                all_recommendations.append(item)
            if len(all_recommendations) >= int(n_recommendations * 0.85):
                break
        
        # 4. Fill with time-weighted popular items (15% weight)
        for item in self.popular_items:
            if len(all_recommendations) >= n_recommendations:
                break
            if item not in user_items and item not in all_recommendations:
                all_recommendations.append(item)
        
        return all_recommendations[:n_recommendations]
    
    def evaluate_recall(self, train_df, test_split_ratio=0.2):
        """Enhanced evaluation with multiple metrics"""
        print("Evaluating enhanced recommender...")
        
        # Create test set by holding out interactions
        test_interactions = {}
        
        for user_id, user_items in self.user_profiles.items():
            user_items_list = list(user_items)
            if len(user_items_list) >= 2:
                # Hold out last item for testing
                test_item = user_items_list[-1]
                train_items = user_items_list[:-1]
                
                test_interactions[user_id] = test_item
                
                # Update profiles for evaluation
                self.user_profiles[user_id] = set(train_items)
                if user_id in self.user_profiles_weighted:
                    # Remove test item from weighted profile
                    if test_item in self.user_profiles_weighted[user_id]:
                        del self.user_profiles_weighted[user_id][test_item]
        
        print(f"Testing on {len(test_interactions)} users with held-out items")
        
        # Generate recommendations and evaluate
        hits_at_5 = 0
        hits_at_10 = 0
        total_users = len(test_interactions)
        
        for user_id, true_item in tqdm(test_interactions.items(), desc="Evaluating"):
            recommendations = self.recommend(user_id, 10)
            
            if true_item in recommendations[:5]:
                hits_at_5 += 1
                hits_at_10 += 1
            elif true_item in recommendations[:10]:
                hits_at_10 += 1
        
        # Restore user profiles
        for user_id, user_items in train_df.groupby('user_id')['item_id'].apply(set).items():
            self.user_profiles[str(user_id)] = user_items
        
        # Rebuild weighted profiles
        user_interactions = defaultdict(list)
        for _, row in train_df.iterrows():
            user_interactions[str(row['user_id'])].append({
                'item_id': row['item_id'],
                'timestamp': row.get('timestamp', 0)
            })
        
        for user_id, interactions in user_interactions.items():
            items = [inter['item_id'] for inter in interactions]
            timestamps = [inter['timestamp'] for inter in interactions]
            weights = self.calculate_time_weights(timestamps)
            weighted_items = defaultdict(float)
            for item, weight in zip(items, weights):
                weighted_items[item] += weight
            self.user_profiles_weighted[user_id] = dict(weighted_items)
        
        recall_at_5 = hits_at_5 / total_users if total_users > 0 else 0
        recall_at_10 = hits_at_10 / total_users if total_users > 0 else 0
        
        print(f"Recall@5: {recall_at_5:.4f} ({hits_at_5}/{total_users})")
        print(f"Recall@10: {recall_at_10:.4f} ({hits_at_10}/{total_users})")
        
        return recall_at_10

def main():
    print("=== Enhanced Sparse Data Recommender ===")
    
    # Load data
    train_df = pd.read_csv('train.csv')
    
    # Load item metadata
    try:
        item_meta_df = pd.read_csv('item_meta.csv')
        print(f"Loaded item metadata: {len(item_meta_df)} items")
    except:
        item_meta_df = None
        print("No item metadata available")
    
    # Train enhanced recommender
    recommender = EnhancedSparseRecommender()
    recommender.fit(train_df, item_meta_df)
    
    # Evaluate
    recall = recommender.evaluate_recall(train_df)
    
    # Generate submission
    print("\nGenerating recommendations for submission...")
    
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
        print("sample_submission.csv not found, using test.csv")
        test_df = pd.read_csv('test.csv')
        test_users = test_df['user_id'].astype(str).unique()[:100]
        
        submission_data = []
        for idx, user_id in enumerate(tqdm(test_users, desc="Generating submission")):
            recommendations = recommender.recommend(user_id, 10)
            
            while len(recommendations) < 10:
                recommendations.extend(recommender.popular_items)
            recommendations = recommendations[:10]
            
            submission_data.append({
                'ID': idx,
                'user_id': user_id,
                'item_id': ','.join(recommendations)
            })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('enhanced_recommender_submission.csv', index=False)
    
    print(f"Submission shape: {submission_df.shape}")
    print("Submission saved to 'enhanced_recommender_submission.csv'")
    
    # Save model
    with open('enhanced_recommender_model.pkl', 'wb') as f:
        pickle.dump(recommender, f)
    
    print(f"\nFinal Results:")
    print(f"Enhanced Recall@10: {recall:.4f}")
    print("Enhanced recommender with multiple advanced techniques!")

if __name__ == "__main__":
    main() 