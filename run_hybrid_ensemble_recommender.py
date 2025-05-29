import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from tqdm import tqdm
import pickle
import random
from scipy.sparse import csr_matrix

warnings.filterwarnings('ignore')

class HybridEnsembleRecommender:
    """Hybrid ensemble recommender combining multiple algorithms"""
    
    def __init__(self):
        self.user_profiles = {}
        self.item_profiles = {}
        self.popular_items = []
        self.user_item_matrix = None
        self.item_similarity = {}
        self.user_similarity = {}
        self.content_features = {}
        self.svd_model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.stats = {}
        
        # Ensemble weights for different algorithms
        self.ensemble_weights = {
            'item_cf': 0.25,        # Item-based collaborative filtering
            'user_cf': 0.20,        # User-based collaborative filtering  
            'matrix_factorization': 0.25,  # SVD matrix factorization
            'content_based': 0.15,  # Content-based filtering
            'popularity': 0.10,     # Popularity-based
            'association_rules': 0.05  # Association rules
        }
    
    def build_user_item_matrix(self, train_df):
        """Build sparse user-item interaction matrix"""
        print("Building user-item matrix...")
        
        # Encode users and items
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        
        self.user_encoder.fit(users)
        self.item_encoder.fit(items)
        
        user_indices = self.user_encoder.transform(train_df['user_id'])
        item_indices = self.item_encoder.transform(train_df['item_id'])
        
        # Create sparse matrix (binary interactions)
        data = np.ones(len(train_df))
        matrix = csr_matrix((data, (user_indices, item_indices)), 
                           shape=(len(users), len(items)))
        
        return matrix
    
    def train_matrix_factorization(self, matrix, n_components=50):
        """Train SVD matrix factorization model"""
        print("Training matrix factorization...")
        
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd_model.fit_transform(matrix)
        self.item_factors = self.svd_model.components_.T
        
        print(f"Matrix factorization completed with {n_components} components")
    
    def build_item_similarity(self, matrix):
        """Build item-item similarity matrix"""
        print("Building item similarity matrix...")
        
        # Transpose to get item-user matrix
        item_matrix = matrix.T
        
        # Calculate cosine similarity between items
        similarity_matrix = cosine_similarity(item_matrix)
        
        # Convert to dictionary for easier access
        item_similarity = {}
        for i in range(similarity_matrix.shape[0]):
            # Get top similar items (excluding self)
            similar_items = np.argsort(similarity_matrix[i])[::-1][1:51]  # Top 50
            item_similarity[i] = {
                j: similarity_matrix[i][j] for j in similar_items 
                if similarity_matrix[i][j] > 0.1  # Threshold
            }
        
        return item_similarity
    
    def build_user_similarity(self, matrix):
        """Build user-user similarity matrix (for top active users only)"""
        print("Building user similarity matrix...")
        
        # Only compute for users with enough interactions to save memory
        user_item_counts = np.array(matrix.sum(axis=1)).flatten()
        active_users = np.where(user_item_counts >= 3)[0]  # Users with 3+ interactions
        
        if len(active_users) > 5000:  # Limit to prevent memory issues
            active_users = active_users[:5000]
        
        active_matrix = matrix[active_users]
        similarity_matrix = cosine_similarity(active_matrix)
        
        user_similarity = {}
        for i, user_idx in enumerate(active_users):
            similar_users = np.argsort(similarity_matrix[i])[::-1][1:31]  # Top 30
            user_similarity[user_idx] = {
                active_users[j]: similarity_matrix[i][j] 
                for j in similar_users 
                if similarity_matrix[i][j] > 0.1
            }
        
        return user_similarity
    
    def build_content_features(self, item_meta_df):
        """Build content-based features"""
        print("Building content features...")
        
        if item_meta_df is None:
            return {}
        
        content_features = {}
        
        # Process text columns for TF-IDF
        text_features = []
        item_ids = []
        
        for _, row in item_meta_df.iterrows():
            item_id = str(row['item_id'])
            text_content = []
            
            for col in item_meta_df.columns:
                if col != 'item_id' and pd.notna(row[col]):
                    text_content.append(str(row[col]).lower())
            
            combined_text = ' '.join(text_content)
            text_features.append(combined_text)
            item_ids.append(item_id)
        
        # Create TF-IDF features
        if text_features:
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english', 
                                   ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform(text_features)
            
            # Store as dictionary
            for i, item_id in enumerate(item_ids):
                content_features[item_id] = tfidf_matrix[i].toarray().flatten()
        
        return content_features
    
    def fit(self, train_df, item_meta_df=None):
        """Fit the hybrid ensemble recommender"""
        print("=== Training Hybrid Ensemble Recommender ===")
        
        # Preprocessing
        print("1. Preprocessing...")
        train_df = train_df.dropna(subset=['user_id', 'item_id']).copy()
        train_df['user_id'] = train_df['user_id'].astype(str)
        train_df['item_id'] = train_df['item_id'].astype(str)
        train_df = train_df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
        
        print(f"Training data: {len(train_df):,} interactions")
        print(f"Users: {train_df['user_id'].nunique():,}")
        print(f"Items: {train_df['item_id'].nunique():,}")
        
        # Build user profiles
        print("2. Building user profiles...")
        for _, row in train_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = set()
            self.user_profiles[user_id].add(item_id)
            
            if item_id not in self.item_profiles:
                self.item_profiles[item_id] = set()
            self.item_profiles[item_id].add(user_id)
        
        # Build user-item matrix
        self.user_item_matrix = self.build_user_item_matrix(train_df)
        
        # Train different models
        self.train_matrix_factorization(self.user_item_matrix)
        self.item_similarity = self.build_item_similarity(self.user_item_matrix)
        self.user_similarity = self.build_user_similarity(self.user_item_matrix)
        
        # Build content features
        if item_meta_df is not None:
            self.content_features = self.build_content_features(item_meta_df)
        
        # Calculate popularity
        item_popularity = train_df['item_id'].value_counts()
        # Apply log transformation to reduce bias
        item_scores = {}
        for item, count in item_popularity.items():
            item_scores[item] = np.log(count + 1)
        
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        self.popular_items = [item for item, score in sorted_items[:200]]
        
        self.stats = {
            'n_users': len(self.user_profiles),
            'n_items': len(self.item_profiles),
            'n_interactions': len(train_df),
            'matrix_density': self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])
        }
        
        print(f"Training completed!")
        print(f"Stats: {self.stats}")
        return self
    
    def get_item_cf_recommendations(self, user_items, n_recommendations=20):
        """Item-based collaborative filtering recommendations"""
        candidate_scores = defaultdict(float)
        
        for item in user_items:
            try:
                item_idx = self.item_encoder.transform([item])[0]
                if item_idx in self.item_similarity:
                    for similar_item_idx, similarity in self.item_similarity[item_idx].items():
                        similar_item = self.item_encoder.inverse_transform([similar_item_idx])[0]
                        if similar_item not in user_items:
                            candidate_scores[similar_item] += similarity
            except:
                continue
        
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:n_recommendations]]
    
    def get_user_cf_recommendations(self, user_id, user_items, n_recommendations=15):
        """User-based collaborative filtering recommendations"""
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
            if user_idx not in self.user_similarity:
                return []
            
            candidate_scores = defaultdict(float)
            
            for similar_user_idx, similarity in self.user_similarity[user_idx].items():
                similar_user = self.user_encoder.inverse_transform([similar_user_idx])[0]
                if similar_user in self.user_profiles:
                    for item in self.user_profiles[similar_user]:
                        if item not in user_items:
                            candidate_scores[item] += similarity
            
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            return [item for item, score in sorted_candidates[:n_recommendations]]
        except:
            return []
    
    def get_mf_recommendations(self, user_id, user_items, n_recommendations=20):
        """Matrix factorization recommendations"""
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
            user_vector = self.user_factors[user_idx]
            
            # Calculate scores for all items
            item_scores = np.dot(self.item_factors, user_vector)
            
            # Get top items that user hasn't interacted with
            recommendations = []
            sorted_indices = np.argsort(item_scores)[::-1]
            
            for item_idx in sorted_indices:
                item = self.item_encoder.inverse_transform([item_idx])[0]
                if item not in user_items and len(recommendations) < n_recommendations:
                    recommendations.append(item)
            
            return recommendations
        except:
            return []
    
    def get_content_recommendations(self, user_items, n_recommendations=15):
        """Content-based recommendations"""
        if not self.content_features:
            return []
        
        # Build user content profile
        user_profile = np.zeros(len(next(iter(self.content_features.values()))))
        profile_count = 0
        
        for item in user_items:
            if item in self.content_features:
                user_profile += self.content_features[item]
                profile_count += 1
        
        if profile_count == 0:
            return []
        
        user_profile /= profile_count  # Average
        
        # Score all items
        candidate_scores = {}
        for item, features in self.content_features.items():
            if item not in user_items:
                similarity = cosine_similarity([user_profile], [features])[0][0]
                if similarity > 0.1:  # Threshold
                    candidate_scores[item] = similarity
        
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:n_recommendations]]
    
    def get_association_rules_recommendations(self, user_items, n_recommendations=10):
        """Simple association rules (frequent itemsets)"""
        candidate_scores = defaultdict(int)
        
        # For each user item, find items that frequently appear together
        for item in user_items:
            if item in self.item_profiles:
                # Get users who also liked this item
                similar_users = list(self.item_profiles[item])[:100]  # Limit for performance
                
                # Count co-occurring items
                for user in similar_users:
                    if user in self.user_profiles:
                        for co_item in self.user_profiles[user]:
                            if co_item not in user_items:
                                candidate_scores[co_item] += 1
        
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:n_recommendations]]
    
    def ensemble_recommendations(self, recommendations_dict, n_recommendations=10):
        """Ensemble different recommendation lists with weighted voting"""
        candidate_scores = defaultdict(float)
        
        # Weight and combine recommendations
        for algo, recommendations in recommendations_dict.items():
            if algo in self.ensemble_weights:
                weight = self.ensemble_weights[algo]
                for i, item in enumerate(recommendations):
                    # Higher position = higher score
                    position_score = (len(recommendations) - i) / len(recommendations)
                    candidate_scores[item] += weight * position_score
        
        # Sort by ensemble score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:n_recommendations]]
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate hybrid ensemble recommendations"""
        user_id = str(user_id)
        
        # Get user's interaction history
        if user_id in self.user_profiles:
            user_items = self.user_profiles[user_id]
        else:
            # Cold start - return popular items with some diversity
            return self.popular_items[:n_recommendations]
        
        # Get recommendations from each algorithm
        recommendations_dict = {}
        
        # 1. Item-based collaborative filtering
        recommendations_dict['item_cf'] = self.get_item_cf_recommendations(user_items, 20)
        
        # 2. User-based collaborative filtering  
        recommendations_dict['user_cf'] = self.get_user_cf_recommendations(user_id, user_items, 15)
        
        # 3. Matrix factorization
        recommendations_dict['matrix_factorization'] = self.get_mf_recommendations(user_id, user_items, 20)
        
        # 4. Content-based
        recommendations_dict['content_based'] = self.get_content_recommendations(user_items, 15)
        
        # 5. Association rules
        recommendations_dict['association_rules'] = self.get_association_rules_recommendations(user_items, 10)
        
        # 6. Popularity baseline
        popularity_recs = []
        for item in self.popular_items:
            if item not in user_items and len(popularity_recs) < 15:
                popularity_recs.append(item)
        recommendations_dict['popularity'] = popularity_recs
        
        # Ensemble all recommendations
        final_recommendations = self.ensemble_recommendations(recommendations_dict, n_recommendations)
        
        # Fill with popular items if needed
        while len(final_recommendations) < n_recommendations:
            for item in self.popular_items:
                if item not in user_items and item not in final_recommendations:
                    final_recommendations.append(item)
                if len(final_recommendations) >= n_recommendations:
                    break
            break
        
        return final_recommendations[:n_recommendations]
    
    def evaluate_recall(self, train_df):
        """Evaluate recall using hold-out method"""
        print("Evaluating hybrid ensemble recommender...")
        
        test_interactions = {}
        original_profiles = {}
        
        for user_id, user_items in self.user_profiles.items():
            user_items_list = list(user_items)
            if len(user_items_list) >= 2:
                original_profiles[user_id] = user_items.copy()
                
                # Hold out last item
                test_item = user_items_list[-1]
                train_items = user_items_list[:-1]
                
                test_interactions[user_id] = test_item
                self.user_profiles[user_id] = set(train_items)
        
        print(f"Testing on {len(test_interactions)} users")
        
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
        
        # Restore profiles
        self.user_profiles.update(original_profiles)
        
        recall_at_5 = hits_at_5 / total_users if total_users > 0 else 0
        recall_at_10 = hits_at_10 / total_users if total_users > 0 else 0
        
        print(f"Recall@5: {recall_at_5:.4f} ({hits_at_5}/{total_users})")
        print(f"Recall@10: {recall_at_10:.4f} ({hits_at_10}/{total_users})")
        
        return recall_at_10

def main():
    print("=== Hybrid Ensemble Recommender System ===")
    
    # Load data
    train_df = pd.read_csv('train.csv')
    
    # Load item metadata
    try:
        item_meta_df = pd.read_csv('item_meta.csv')
        print(f"Loaded item metadata: {len(item_meta_df)} items")
    except:
        item_meta_df = None
        print("No item metadata available")
    
    # Train hybrid ensemble recommender
    recommender = HybridEnsembleRecommender()
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
        print("sample_submission.csv not found")
        return
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('hybrid_ensemble_submission.csv', index=False)
    
    print(f"Submission shape: {submission_df.shape}")
    print("Submission saved to 'hybrid_ensemble_submission.csv'")
    
    # Save model
    with open('hybrid_ensemble_model.pkl', 'wb') as f:
        pickle.dump(recommender, f)
    
    print(f"\nFinal Results:")
    print(f"Hybrid Ensemble Recall@10: {recall:.4f}")
    print("Hybrid ensemble combining 6 different algorithms!")
    
    # Print algorithm contributions
    print("\nEnsemble weights:")
    for algo, weight in recommender.ensemble_weights.items():
        print(f"  {algo}: {weight:.2f}")

if __name__ == "__main__":
    main() 