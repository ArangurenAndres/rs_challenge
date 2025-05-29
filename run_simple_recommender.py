import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import warnings
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')

class SparseDataRecommender:
    """Simple recommender optimized for extremely sparse data"""
    
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_item_matrix = None
        self.item_similarity = None
        self.user_profiles = {}
        self.popular_items = []
        self.stats = {}
        
    def fit(self, train_df):
        """Fit the recommender system"""
        print("=== Training Simple Sparse Data Recommender ===")
        
        # 1. Minimal preprocessing - keep as much data as possible
        print("1. Minimal preprocessing...")
        train_df = train_df.dropna(subset=['user_id', 'item_id']).copy()
        train_df['user_id'] = train_df['user_id'].astype(str)
        train_df['item_id'] = train_df['item_id'].astype(str)
        
        # Remove duplicates but keep one interaction per user-item pair
        train_df = train_df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
        
        print(f"Training data: {len(train_df):,} interactions")
        print(f"Users: {train_df['user_id'].nunique():,}")
        print(f"Items: {train_df['item_id'].nunique():,}")
        
        # 2. Encode users and items
        print("2. Encoding...")
        self.user_encoder.fit(train_df['user_id'])
        self.item_encoder.fit(train_df['item_id'])
        
        train_df['user_encoded'] = self.user_encoder.transform(train_df['user_id'])
        train_df['item_encoded'] = self.item_encoder.transform(train_df['item_id'])
        
        # 3. Build user profiles (items each user interacted with)
        print("3. Building user profiles...")
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building profiles"):
            user_id = row['user_id']
            item_id = row['item_id']
            
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = set()
            self.user_profiles[user_id].add(item_id)
        
        # 4. Calculate item popularity
        print("4. Calculating item popularity...")
        item_counts = train_df['item_id'].value_counts()
        self.popular_items = item_counts.head(100).index.tolist()
        
        # 5. Build item co-occurrence matrix for item-based CF
        print("5. Building item co-occurrence...")
        item_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for user_items in tqdm(self.user_profiles.values(), desc="Co-occurrence"):
            user_items_list = list(user_items)
            for i, item1 in enumerate(user_items_list):
                for item2 in user_items_list[i+1:]:
                    item_cooccurrence[item1][item2] += 1
                    item_cooccurrence[item2][item1] += 1
        
        self.item_cooccurrence = dict(item_cooccurrence)
        
        # 6. Category-based recommendations from metadata
        print("6. Processing item metadata...")
        try:
            item_meta = pd.read_csv('item_meta.csv')
            item_meta['item_id'] = item_meta['item_id'].astype(str)
            
            # Build category mappings
            self.item_categories = {}
            self.category_items = defaultdict(list)
            
            for _, row in item_meta.iterrows():
                item_id = str(row['item_id'])
                # Use multiple features as "categories"
                category_features = []
                
                for col in ['category', 'brand', 'price_range']:
                    if col in row and pd.notna(row[col]):
                        category_features.append(str(row[col]))
                
                if category_features:
                    category = "_".join(category_features)
                    self.item_categories[item_id] = category
                    self.category_items[category].append(item_id)
        except:
            print("No metadata available for category-based recommendations")
            self.item_categories = {}
            self.category_items = defaultdict(list)
        
        self.stats = {
            'n_users': len(self.user_profiles),
            'n_items': train_df['item_id'].nunique(),
            'n_interactions': len(train_df),
            'avg_items_per_user': np.mean([len(items) for items in self.user_profiles.values()])
        }
        
        print(f"Training completed!")
        print(f"Stats: {self.stats}")
        
        return self
    
    def get_item_based_recommendations(self, user_items, n_recommendations=10):
        """Get recommendations based on item co-occurrence"""
        candidate_scores = defaultdict(float)
        
        for item in user_items:
            if item in self.item_cooccurrence:
                for related_item, score in self.item_cooccurrence[item].items():
                    if related_item not in user_items:
                        candidate_scores[related_item] += score
        
        # Sort by score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:n_recommendations]]
    
    def get_category_based_recommendations(self, user_items, n_recommendations=10):
        """Get recommendations based on item categories"""
        if not self.item_categories:
            return []
        
        # Find categories of user's items
        user_categories = []
        for item in user_items:
            if item in self.item_categories:
                user_categories.append(self.item_categories[item])
        
        if not user_categories:
            return []
        
        # Find most common categories
        category_counts = Counter(user_categories)
        top_categories = [cat for cat, count in category_counts.most_common(3)]
        
        # Get items from these categories
        candidate_items = []
        for category in top_categories:
            for item in self.category_items[category]:
                if item not in user_items and item not in candidate_items:
                    candidate_items.append(item)
        
        return candidate_items[:n_recommendations]
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate recommendations for a user"""
        user_id = str(user_id)
        
        # Get user's interaction history
        if user_id in self.user_profiles:
            user_items = self.user_profiles[user_id]
        else:
            # Cold start - return popular items
            return self.popular_items[:n_recommendations]
        
        recommendations = []
        
        # 1. Item-based collaborative filtering
        item_based_recs = self.get_item_based_recommendations(user_items, n_recommendations)
        recommendations.extend(item_based_recs)
        
        # 2. Category-based recommendations
        category_recs = self.get_category_based_recommendations(user_items, n_recommendations)
        for item in category_recs:
            if item not in recommendations:
                recommendations.append(item)
        
        # 3. Fill with popular items
        for item in self.popular_items:
            if len(recommendations) >= n_recommendations:
                break
            if item not in user_items and item not in recommendations:
                recommendations.append(item)
        
        return recommendations[:n_recommendations]
    
    def evaluate_recall(self, train_df, test_split_ratio=0.2):
        """Evaluate recall by holding out last interactions"""
        print("Evaluating recall...")
        
        # Create test set by holding out some interactions
        test_interactions = {}
        train_for_eval = []
        
        for user_id, user_items in self.user_profiles.items():
            user_items_list = list(user_items)
            if len(user_items_list) >= 2:
                # Hold out last item for testing
                test_item = user_items_list[-1]
                train_items = user_items_list[:-1]
                
                test_interactions[user_id] = test_item
                
                # Update user profile for evaluation (without test item)
                self.user_profiles[user_id] = set(train_items)
        
        print(f"Testing on {len(test_interactions)} users with held-out items")
        
        # Generate recommendations and check hits
        hits = 0
        total_users = len(test_interactions)
        
        for user_id, true_item in tqdm(test_interactions.items(), desc="Evaluating"):
            recommendations = self.recommend(user_id, 10)
            if true_item in recommendations:
                hits += 1
        
        # Restore user profiles
        for user_id, user_items in train_df.groupby('user_id')['item_id'].apply(set).items():
            self.user_profiles[str(user_id)] = user_items
        
        recall_at_10 = hits / total_users if total_users > 0 else 0
        print(f"Recall@10: {recall_at_10:.4f}")
        print(f"Hits: {hits}/{total_users}")
        
        return recall_at_10

def main():
    print("=== Simple Sparse Data Recommender ===")
    
    # Load data
    train_df = pd.read_csv('train.csv')
    
    # Train recommender
    recommender = SparseDataRecommender()
    recommender.fit(train_df)
    
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
            original_id = row['ID']  # Use the original ID from sample submission
            
            recommendations = recommender.recommend(user_id, 10)
            
            # Ensure we have 10 recommendations
            while len(recommendations) < 10:
                recommendations.extend(recommender.popular_items)
            recommendations = recommendations[:10]
            
            submission_data.append({
                'ID': original_id,  # Use original ID
                'user_id': row['user_id'],  # Use original user_id format
                'item_id': ','.join(recommendations)
            })
        
    except FileNotFoundError:
        print("sample_submission.csv not found, using test.csv")
        test_df = pd.read_csv('test.csv')
        test_users = test_df['user_id'].astype(str).unique()[:100]
        
        submission_data = []
        for idx, user_id in enumerate(tqdm(test_users, desc="Generating submission")):
            recommendations = recommender.recommend(user_id, 10)
            
            # Ensure we have 10 recommendations
            while len(recommendations) < 10:
                recommendations.extend(recommender.popular_items)
            recommendations = recommendations[:10]
            
            submission_data.append({
                'ID': idx,
                'user_id': user_id,
                'item_id': ','.join(recommendations)
            })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('simple_recommender_submission.csv', index=False)
    
    print(f"Submission shape: {submission_df.shape}")
    print("Submission saved to 'simple_recommender_submission.csv'")
    
    # Save model
    with open('simple_recommender_model.pkl', 'wb') as f:
        pickle.dump(recommender, f)
    
    print(f"\nFinal Results:")
    print(f"Recall@10: {recall:.4f}")
    print("This approach should work much better for sparse data!")

if __name__ == "__main__":
    main() 