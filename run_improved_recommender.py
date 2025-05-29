import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
import warnings
from tqdm import tqdm
import pickle
import random

warnings.filterwarnings('ignore')

class ImprovedSparseRecommender:
    """Improved recommender with key enhancements for better performance"""
    
    def __init__(self):
        self.user_profiles = {}
        self.popular_items = []
        self.item_similarity = {}
        self.content_features = {}
        self.category_popularity = {}
        self.stats = {}
        
    def build_content_features(self, item_meta_df):
        """Build content features from item metadata"""
        print("Building content features...")
        
        self.content_features = {}
        category_items = defaultdict(list)
        
        if item_meta_df is None or len(item_meta_df) == 0:
            return
        
        for _, row in item_meta_df.iterrows():
            item_id = str(row['item_id'])
            features = {}
            
            # Process each column
            for col in item_meta_df.columns:
                if col != 'item_id' and pd.notna(row[col]):
                    value = str(row[col]).lower().strip()
                    
                    if 'category' in col.lower():
                        features['category'] = value
                        category_items[value].append(item_id)
                    elif 'brand' in col.lower():
                        features['brand'] = value
                    elif 'price' in col.lower():
                        try:
                            price = float(value)
                            if price < 20:
                                features['price_range'] = 'low'
                            elif price < 100:
                                features['price_range'] = 'medium'
                            else:
                                features['price_range'] = 'high'
                        except:
                            features['price_range'] = 'unknown'
            
            self.content_features[item_id] = features
        
        # Build category popularity
        for category, items in category_items.items():
            self.category_popularity[category] = len(items)
    
    def fit(self, train_df, item_meta_df=None):
        """Fit the improved recommender system"""
        print("=== Training Improved Sparse Data Recommender ===")
        
        # 1. Preprocessing
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
        
        # 3. Calculate improved item popularity (with diversity boost)
        print("3. Calculating improved item popularity...")
        item_counts = train_df['item_id'].value_counts()
        
        # Apply sqrt transformation to reduce popularity bias
        item_scores = {}
        for item, count in item_counts.items():
            item_scores[item] = np.sqrt(count)  # Reduce popularity bias
        
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        self.popular_items = [item for item, score in sorted_items[:150]]
        
        # 4. Build improved item similarity with TF-IDF style weighting
        print("4. Building improved item similarity...")
        item_cooccurrence = defaultdict(lambda: defaultdict(int))
        user_item_counts = defaultdict(int)
        
        # Count co-occurrences
        for user_items in tqdm(self.user_profiles.values(), desc="Co-occurrence"):
            user_items_list = list(user_items)
            for item in user_items_list:
                user_item_counts[item] += 1
            
            for i, item1 in enumerate(user_items_list):
                for item2 in user_items_list[i+1:]:
                    item_cooccurrence[item1][item2] += 1
                    item_cooccurrence[item2][item1] += 1
        
        # Apply TF-IDF style weighting (reduce weight of very popular items)
        total_users = len(self.user_profiles)
        for item1, related_items in item_cooccurrence.items():
            for item2 in related_items:
                # TF-IDF style: co-occurrence count / log(popularity)
                idf_weight = np.log(total_users / (user_item_counts[item2] + 1))
                related_items[item2] = related_items[item2] * idf_weight
        
        self.item_similarity = dict(item_cooccurrence)
        
        # 5. Build content features
        if item_meta_df is not None:
            self.build_content_features(item_meta_df)
        
        self.stats = {
            'n_users': len(self.user_profiles),
            'n_items': train_df['item_id'].nunique(),
            'n_interactions': len(train_df),
            'avg_items_per_user': np.mean([len(items) for items in self.user_profiles.values()]),
            'content_items': len(self.content_features)
        }
        
        print(f"Training completed!")
        print(f"Stats: {self.stats}")
        return self
    
    def get_improved_collaborative_recommendations(self, user_items, n_recommendations=15):
        """Improved collaborative filtering with better scoring"""
        candidate_scores = defaultdict(float)
        
        for item in user_items:
            if item in self.item_similarity:
                for related_item, score in self.item_similarity[item].items():
                    if related_item not in user_items:
                        candidate_scores[related_item] += score
        
        # Apply item diversity boost
        final_scores = {}
        for item, score in candidate_scores.items():
            # Boost score for items that appear with multiple user items
            item_connections = sum(1 for user_item in user_items 
                                 if user_item in self.item_similarity and 
                                 item in self.item_similarity[user_item])
            diversity_boost = 1 + (item_connections * 0.1)
            final_scores[item] = score * diversity_boost
        
        sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:n_recommendations]]
    
    def get_improved_content_recommendations(self, user_items, n_recommendations=10):
        """Improved content-based recommendations"""
        if not self.content_features:
            return []
        
        # Analyze user preferences
        user_categories = []
        user_brands = []
        user_price_ranges = []
        
        for item in user_items:
            if item in self.content_features:
                features = self.content_features[item]
                if 'category' in features:
                    user_categories.append(features['category'])
                if 'brand' in features:
                    user_brands.append(features['brand'])
                if 'price_range' in features:
                    user_price_ranges.append(features['price_range'])
        
        if not user_categories and not user_brands:
            return []
        
        # Find preferred categories and brands
        preferred_categories = Counter(user_categories).most_common(3)
        preferred_brands = Counter(user_brands).most_common(2)
        preferred_prices = Counter(user_price_ranges).most_common(1)
        
        # Score candidate items
        candidate_scores = defaultdict(float)
        
        for item_id, features in self.content_features.items():
            if item_id not in user_items:
                score = 0
                
                # Category match
                if 'category' in features:
                    for cat, count in preferred_categories:
                        if features['category'] == cat:
                            score += count * 2  # Weight categories highly
                
                # Brand match  
                if 'brand' in features:
                    for brand, count in preferred_brands:
                        if features['brand'] == brand:
                            score += count * 1.5
                
                # Price range match
                if 'price_range' in features:
                    for price_range, count in preferred_prices:
                        if features['price_range'] == price_range:
                            score += count
                
                if score > 0:
                    candidate_scores[item_id] = score
        
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:n_recommendations]]
    
    def get_diversified_popular_items(self, user_items, n_recommendations=10):
        """Get diversified popular items"""
        # Try to get popular items from different categories if possible
        diversified_items = []
        used_categories = set()
        
        for item in self.popular_items:
            if item not in user_items and item not in diversified_items:
                # Check if we should add this item for diversity
                item_category = None
                if item in self.content_features:
                    item_category = self.content_features[item].get('category')
                
                # Add item if it's from a new category or we need more items
                if item_category is None or item_category not in used_categories or len(diversified_items) >= n_recommendations * 0.7:
                    diversified_items.append(item)
                    if item_category:
                        used_categories.add(item_category)
                
                if len(diversified_items) >= n_recommendations:
                    break
        
        return diversified_items
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate improved recommendations using multiple enhanced strategies"""
        user_id = str(user_id)
        
        # Get user's interaction history
        if user_id in self.user_profiles:
            user_items = self.user_profiles[user_id]
        else:
            # Cold start - return diversified popular items
            return self.get_diversified_popular_items(set(), n_recommendations)
        
        all_recommendations = []
        
        # 1. Improved collaborative filtering (50% weight)
        collab_recs = self.get_improved_collaborative_recommendations(user_items, n_recommendations)
        all_recommendations.extend(collab_recs[:int(n_recommendations * 0.7)])
        
        # 2. Improved content-based recommendations (30% weight)
        content_recs = self.get_improved_content_recommendations(user_items, n_recommendations)
        for item in content_recs:
            if item not in all_recommendations:
                all_recommendations.append(item)
            if len(all_recommendations) >= int(n_recommendations * 0.8):
                break
        
        # 3. Fill with diversified popular items (20% weight)
        diversified_popular = self.get_diversified_popular_items(user_items, n_recommendations)
        for item in diversified_popular:
            if len(all_recommendations) >= n_recommendations:
                break
            if item not in all_recommendations:
                all_recommendations.append(item)
        
        return all_recommendations[:n_recommendations]
    
    def evaluate_recall(self, train_df):
        """Evaluate recall with improved methodology"""
        print("Evaluating improved recommender...")
        
        # Create test set by holding out interactions
        test_interactions = {}
        original_profiles = {}
        
        for user_id, user_items in self.user_profiles.items():
            user_items_list = list(user_items)
            if len(user_items_list) >= 2:
                # Store original profile
                original_profiles[user_id] = user_items.copy()
                
                # Hold out last item for testing
                test_item = user_items_list[-1]
                train_items = user_items_list[:-1]
                
                test_interactions[user_id] = test_item
                
                # Update profile for evaluation
                self.user_profiles[user_id] = set(train_items)
        
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
        
        # Restore original user profiles
        self.user_profiles.update(original_profiles)
        
        recall_at_5 = hits_at_5 / total_users if total_users > 0 else 0
        recall_at_10 = hits_at_10 / total_users if total_users > 0 else 0
        
        print(f"Recall@5: {recall_at_5:.4f} ({hits_at_5}/{total_users})")
        print(f"Recall@10: {recall_at_10:.4f} ({hits_at_10}/{total_users})")
        
        return recall_at_10

def main():
    print("=== Improved Sparse Data Recommender ===")
    
    # Load data
    train_df = pd.read_csv('train.csv')
    
    # Load item metadata
    try:
        item_meta_df = pd.read_csv('item_meta.csv')
        print(f"Loaded item metadata: {len(item_meta_df)} items")
    except:
        item_meta_df = None
        print("No item metadata available")
    
    # Train improved recommender
    recommender = ImprovedSparseRecommender()
    recommender.fit(train_df, item_meta_df)
    
    # Evaluate
    #recall = recommender.evaluate_recall(train_df)
    
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
    submission_df.to_csv('improved_recommender_submission.csv', index=False)
    
    print(f"Submission shape: {submission_df.shape}")
    print("Submission saved to 'improved_recommender_submission.csv'")
    
    # Save model
    with open('improved_recommender_model.pkl', 'wb') as f:
        pickle.dump(recommender, f)
    
    print(f"\nFinal Results:")
    print(f"Improved Recall@10: {recall:.4f}")
    print("Improved recommender with enhanced techniques!")

if __name__ == "__main__":
    main() 