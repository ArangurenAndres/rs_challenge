import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
import warnings
from tqdm import tqdm
import pickle
import random

warnings.filterwarnings('ignore')

class LightweightHybridRecommender:
    """Lightweight hybrid recommender optimized for sparse data and memory efficiency"""
    
    def __init__(self):
        self.user_profiles = {}
        self.item_profiles = {}
        self.popular_items = []
        self.item_cooccurrence = {}
        self.user_similarity_simple = {}
        self.content_categories = {}
        self.category_items = defaultdict(list)
        self.stats = {}
        
        # Ensemble weights for different strategies
        self.ensemble_weights = {
            'item_cooccurrence': 0.35,    # Strong for sparse data
            'user_neighborhood': 0.25,    # User-based recommendations
            'popularity_weighted': 0.20,  # Smart popularity
            'content_category': 0.15,     # Category-based
            'diversity_boost': 0.05       # Diversity enhancement
        }
    
    def build_item_cooccurrence(self, max_items_per_user=20):
        """Build item co-occurrence matrix efficiently"""
        print("Building item co-occurrence matrix...")
        
        cooccurrence = defaultdict(lambda: defaultdict(int))
        item_counts = defaultdict(int)
        
        for user_items in tqdm(self.user_profiles.values(), desc="Co-occurrence"):
            user_items_list = list(user_items)
            
            # Limit items per user to manage memory
            if len(user_items_list) > max_items_per_user:
                user_items_list = random.sample(user_items_list, max_items_per_user)
            
            # Count item frequencies
            for item in user_items_list:
                item_counts[item] += 1
            
            # Count co-occurrences
            for i, item1 in enumerate(user_items_list):
                for item2 in user_items_list[i+1:]:
                    cooccurrence[item1][item2] += 1
                    cooccurrence[item2][item1] += 1
        
        # Apply TF-IDF style weighting and keep only strong associations
        print("Applying TF-IDF weighting...")
        total_users = len(self.user_profiles)
        
        filtered_cooccurrence = {}
        for item1, related_items in cooccurrence.items():
            filtered_cooccurrence[item1] = {}
            
            for item2, count in related_items.items():
                if count >= 2:  # Minimum co-occurrence threshold
                    # TF-IDF style: count * log(total_users / item_frequency)
                    idf1 = np.log(total_users / (item_counts[item1] + 1))
                    idf2 = np.log(total_users / (item_counts[item2] + 1))
                    score = count * (idf1 + idf2) / 2
                    
                    if score > 0.5:  # Minimum score threshold
                        filtered_cooccurrence[item1][item2] = score
        
        return filtered_cooccurrence
    
    def build_user_neighborhoods(self, max_neighbors=50):
        """Build user neighborhoods for collaborative filtering"""
        print("Building user neighborhoods...")
        
        user_neighborhoods = {}
        
        # Only process users with multiple interactions
        active_users = {user_id: items for user_id, items in self.user_profiles.items() 
                       if len(items) >= 2}
        
        print(f"Processing {len(active_users)} active users...")
        
        # Sample users to avoid quadratic complexity
        if len(active_users) > 10000:
            sampled_users = dict(random.sample(list(active_users.items()), 10000))
        else:
            sampled_users = active_users
        
        for user_id, user_items in tqdm(sampled_users.items(), desc="User neighborhoods"):
            similar_users = []
            
            # Find users with similar items (Jaccard similarity)
            for other_user_id, other_items in sampled_users.items():
                if user_id != other_user_id:
                    intersection = len(user_items.intersection(other_items))
                    union = len(user_items.union(other_items))
                    
                    if intersection > 0 and union > 0:
                        jaccard = intersection / union
                        if jaccard > 0.1:  # Minimum similarity threshold
                            similar_users.append((other_user_id, jaccard))
            
            # Keep top similar users
            similar_users.sort(key=lambda x: x[1], reverse=True)
            user_neighborhoods[user_id] = similar_users[:max_neighbors]
        
        return user_neighborhoods
    
    def build_content_features(self, item_meta_df):
        """Build lightweight content features"""
        print("Building content features...")
        
        if item_meta_df is None:
            return
        
        for _, row in item_meta_df.iterrows():
            item_id = str(row['item_id'])
            
            # Extract category information
            category = None
            for col in item_meta_df.columns:
                if 'category' in col.lower() and pd.notna(row[col]):
                    category = str(row[col]).lower().strip()
                    break
            
            if category:
                self.content_categories[item_id] = category
                self.category_items[category].append(item_id)
    
    def fit(self, train_df, item_meta_df=None):
        """Fit the lightweight hybrid recommender"""
        print("=== Training Lightweight Hybrid Recommender ===")
        
        # Preprocessing
        print("1. Preprocessing...")
        train_df = train_df.dropna(subset=['user_id', 'item_id']).copy()
        train_df['user_id'] = train_df['user_id'].astype(str)
        train_df['item_id'] = train_df['item_id'].astype(str)
        train_df = train_df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
        
        print(f"Training data: {len(train_df):,} interactions")
        print(f"Users: {train_df['user_id'].nunique():,}")
        print(f"Items: {train_df['item_id'].nunique():,}")
        
        # Build user and item profiles
        print("2. Building user and item profiles...")
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Profiles"):
            user_id = row['user_id']
            item_id = row['item_id']
            
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = set()
            self.user_profiles[user_id].add(item_id)
            
            if item_id not in self.item_profiles:
                self.item_profiles[item_id] = set()
            self.item_profiles[item_id].add(user_id)
        
        # Build components
        self.item_cooccurrence = self.build_item_cooccurrence()
        self.user_similarity_simple = self.build_user_neighborhoods()
        
        if item_meta_df is not None:
            self.build_content_features(item_meta_df)
        
        # Calculate smart popularity (diversity-adjusted)
        print("3. Calculating smart popularity...")
        item_popularity = train_df['item_id'].value_counts()
        
        # Apply sqrt to reduce popularity bias
        item_scores = {}
        for item, count in item_popularity.items():
            # Diversity bonus for items in less popular categories
            category_bonus = 1.0
            if item in self.content_categories:
                category = self.content_categories[item]
                category_size = len(self.category_items[category])
                if category_size < 1000:  # Boost smaller categories
                    category_bonus = 1.2
            
            item_scores[item] = np.sqrt(count) * category_bonus
        
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        self.popular_items = [item for item, score in sorted_items[:300]]
        
        self.stats = {
            'n_users': len(self.user_profiles),
            'n_items': len(self.item_profiles),
            'n_interactions': len(train_df),
            'avg_items_per_user': np.mean([len(items) for items in self.user_profiles.values()]),
            'cooccurrence_pairs': sum(len(items) for items in self.item_cooccurrence.values()),
            'user_neighborhoods': len(self.user_similarity_simple)
        }
        
        print(f"Training completed!")
        print(f"Stats: {self.stats}")
        return self
    
    def get_item_cooccurrence_recs(self, user_items, n_recommendations=25):
        """Get recommendations from item co-occurrence"""
        candidate_scores = defaultdict(float)
        
        for item in user_items:
            if item in self.item_cooccurrence:
                for related_item, score in self.item_cooccurrence[item].items():
                    if related_item not in user_items:
                        candidate_scores[related_item] += score
        
        # Boost items that co-occur with multiple user items
        final_scores = {}
        for item, score in candidate_scores.items():
            connections = sum(1 for user_item in user_items 
                            if user_item in self.item_cooccurrence and 
                            item in self.item_cooccurrence[user_item])
            boost = 1 + (connections * 0.1)
            final_scores[item] = score * boost
        
        sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:n_recommendations]]
    
    def get_user_neighborhood_recs(self, user_id, user_items, n_recommendations=20):
        """Get recommendations from user neighborhoods"""
        if user_id not in self.user_similarity_simple:
            return []
        
        candidate_scores = defaultdict(float)
        
        for similar_user_id, similarity in self.user_similarity_simple[user_id]:
            if similar_user_id in self.user_profiles:
                for item in self.user_profiles[similar_user_id]:
                    if item not in user_items:
                        candidate_scores[item] += similarity
        
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:n_recommendations]]
    
    def get_category_based_recs(self, user_items, n_recommendations=15):
        """Get recommendations based on user's preferred categories"""
        if not self.content_categories:
            return []
        
        # Find user's preferred categories
        user_categories = []
        for item in user_items:
            if item in self.content_categories:
                user_categories.append(self.content_categories[item])
        
        if not user_categories:
            return []
        
        # Get most common categories
        category_counts = Counter(user_categories)
        top_categories = [cat for cat, count in category_counts.most_common(3)]
        
        # Get items from these categories
        recommendations = []
        for category in top_categories:
            for item in self.category_items[category]:
                if item not in user_items and item not in recommendations:
                    recommendations.append(item)
                if len(recommendations) >= n_recommendations:
                    break
            if len(recommendations) >= n_recommendations:
                break
        
        return recommendations
    
    def get_weighted_popular_recs(self, user_items, n_recommendations=15):
        """Get popularity-based recommendations with user preference weighting"""
        recommendations = []
        
        # Try to match user's category preferences in popular items
        user_categories = set()
        for item in user_items:
            if item in self.content_categories:
                user_categories.add(self.content_categories[item])
        
        # First, add popular items from user's preferred categories
        for item in self.popular_items:
            if item not in user_items and item not in recommendations:
                if not user_categories or (item in self.content_categories and 
                                         self.content_categories[item] in user_categories):
                    recommendations.append(item)
            if len(recommendations) >= n_recommendations * 0.7:
                break
        
        # Fill with other popular items
        for item in self.popular_items:
            if item not in user_items and item not in recommendations:
                recommendations.append(item)
            if len(recommendations) >= n_recommendations:
                break
        
        return recommendations
    
    def get_diversity_boost_recs(self, user_items, existing_recs, n_recommendations=10):
        """Add diversity by recommending from underrepresented categories"""
        if not self.content_categories:
            return []
        
        # Find categories already in recommendations
        rec_categories = set()
        for item in existing_recs:
            if item in self.content_categories:
                rec_categories.add(self.content_categories[item])
        
        # Find underrepresented categories
        all_categories = set(self.category_items.keys())
        underrepresented = all_categories - rec_categories
        
        diversity_recs = []
        for category in list(underrepresented)[:3]:  # Top 3 missing categories
            category_items = self.category_items[category]
            for item in category_items[:5]:  # Top 5 items per category
                if item not in user_items and item not in existing_recs:
                    diversity_recs.append(item)
                if len(diversity_recs) >= n_recommendations:
                    break
        
        return diversity_recs
    
    def ensemble_recommendations(self, recommendations_dict, n_recommendations=10):
        """Combine recommendations with weighted ensemble"""
        candidate_scores = defaultdict(float)
        
        for strategy, recommendations in recommendations_dict.items():
            if strategy in self.ensemble_weights and recommendations:
                weight = self.ensemble_weights[strategy]
                for i, item in enumerate(recommendations):
                    # Position-based scoring: higher position = higher score
                    position_score = (len(recommendations) - i) / len(recommendations)
                    candidate_scores[item] += weight * position_score
        
        # Sort by ensemble score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:n_recommendations]]
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate hybrid recommendations"""
        user_id = str(user_id)
        
        if user_id not in self.user_profiles:
            # Cold start: return smart popular items
            return self.get_weighted_popular_recs(set(), n_recommendations)
        
        user_items = self.user_profiles[user_id]
        
        # Get recommendations from different strategies
        recommendations_dict = {}
        
        # 1. Item co-occurrence (strongest for sparse data)
        recommendations_dict['item_cooccurrence'] = self.get_item_cooccurrence_recs(user_items, 25)
        
        # 2. User neighborhood
        recommendations_dict['user_neighborhood'] = self.get_user_neighborhood_recs(user_id, user_items, 20)
        
        # 3. Weighted popularity
        recommendations_dict['popularity_weighted'] = self.get_weighted_popular_recs(user_items, 15)
        
        # 4. Category-based
        recommendations_dict['content_category'] = self.get_category_based_recs(user_items, 15)
        
        # 5. Get initial ensemble
        initial_recs = self.ensemble_recommendations(recommendations_dict, n_recommendations * 2)
        
        # 6. Add diversity boost
        diversity_recs = self.get_diversity_boost_recs(user_items, initial_recs, 10)
        recommendations_dict['diversity_boost'] = diversity_recs
        
        # Final ensemble
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
        print("Evaluating lightweight hybrid recommender...")
        
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
    print("=== Lightweight Hybrid Recommender System ===")
    
    # Load data
    train_df = pd.read_csv('train.csv')
    
    # Load item metadata
    try:
        item_meta_df = pd.read_csv('item_meta.csv')
        print(f"Loaded item metadata: {len(item_meta_df)} items")
    except:
        item_meta_df = None
        print("No item metadata available")
    
    # Train lightweight hybrid recommender
    recommender = LightweightHybridRecommender()
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
    submission_df.to_csv('lightweight_hybrid_submission.csv', index=False)
    
    print(f"Submission shape: {submission_df.shape}")
    print("Submission saved to 'lightweight_hybrid_submission.csv'")
    
    # Save model
    with open('lightweight_hybrid_model.pkl', 'wb') as f:
        pickle.dump(recommender, f)
    
    print(f"\nFinal Results:")
    print(f"Lightweight Hybrid Recall@10: {recall:.4f}")
    print("Memory-efficient hybrid combining 5 strategies!")
    
    # Print strategy contributions
    print("\nEnsemble weights:")
    for strategy, weight in recommender.ensemble_weights.items():
        print(f"  {strategy}: {weight:.2f}")

if __name__ == "__main__":
    main() 