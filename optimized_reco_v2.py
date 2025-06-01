import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
import implicit
import warnings
warnings.filterwarnings('ignore')

print("=== PERSONALIZED RECOMMENDER V3 ===")
print("Ensuring each user gets unique recommendations\n")

# Load data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
item_meta_df = pd.read_csv('item_meta.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Get test users
test_users = set(sample_submission['user_id'].unique())
print(f"Total test users to predict: {len(test_users)}")

# ========== ANALYZE TEST USER PREFERENCES ==========
print("\n1. Analyzing test user historical preferences...")

# Get test user interactions
test_user_history = defaultdict(list)
test_user_items_set = defaultdict(set)
test_user_last_time = {}

for user in test_users:
    user_data = train_df[train_df['user_id'] == user].sort_values('timestamp')
    if len(user_data) > 0:
        test_user_history[user] = user_data['item_id'].tolist()
        test_user_items_set[user] = set(user_data['item_id'].values)
        test_user_last_time[user] = user_data['timestamp'].max()

print(f"Test users with history: {len(test_user_history)}")
print(f"Test users without history: {len(test_users) - len(test_user_history)}")

# ========== BUILD QUALITY ITEM POOL ==========
print("\n2. Building quality item pool...")

# Get item statistics
item_stats = train_df.groupby('item_id').agg({
    'user_id': 'count',
    'timestamp': ['min', 'max']
}).reset_index()
item_stats.columns = ['item_id', 'popularity', 'first_seen', 'last_seen']

# Merge with ratings
item_stats = item_stats.merge(
    item_meta_df[['item_id', 'average_rating', 'rating_number', 'main_category']], 
    on='item_id', 
    how='left'
)

# Quality criteria based on test user preferences
quality_items = item_stats[
    (item_stats['average_rating'] >= 3.5) &  # Good rating
    (item_stats['popularity'] >= 5) &        # Not too niche
    (item_stats['popularity'] <= 200) &      # Not too mainstream
    (item_stats['rating_number'] >= 2)       # Has ratings
].copy()

print(f"Quality items: {len(quality_items)}")

# Calculate item scores
quality_items['quality_score'] = (
    quality_items['average_rating'] * 0.4 +
    np.log1p(quality_items['popularity']) * 0.3 +
    np.log1p(quality_items['rating_number']) * 0.3
)

# ========== ITEM-ITEM SIMILARITY ==========
print("\n3. Building item-item similarity...")

# Create item co-occurrence matrix
item_pairs = defaultdict(int)
for user, items in test_user_history.items():
    if len(items) >= 2:
        for i in range(len(items)):
            for j in range(max(0, i-5), min(len(items), i+5)):
                if i != j:
                    item_pairs[(items[i], items[j])] += 1

# Convert to similarity scores
item_similarity = defaultdict(dict)
for (item1, item2), count in item_pairs.items():
    item_similarity[item1][item2] = count

print(f"Built similarities for {len(item_similarity)} items")

# ========== USER-BASED COLLABORATIVE FILTERING ==========
print("\n4. Building user-based CF model...")

# Find similar users based on common items
def get_user_similarity(user1_items, user2_items):
    if not user1_items or not user2_items:
        return 0
    intersection = len(user1_items & user2_items)
    union = len(user1_items | user2_items)
    return intersection / union if union > 0 else 0

# For each test user, find similar users
user_neighbors = defaultdict(list)
all_user_items = train_df.groupby('user_id')['item_id'].apply(set).to_dict()

for test_user in test_users:
    if test_user in test_user_items_set:
        test_items = test_user_items_set[test_user]
        similarities = []
        
        # Sample other users for efficiency
        sampled_users = np.random.choice(
            [u for u in all_user_items.keys() if u != test_user], 
            min(1000, len(all_user_items)), 
            replace=False
        )
        
        for other_user in sampled_users:
            sim = get_user_similarity(test_items, all_user_items[other_user])
            if sim > 0.05:  # At least 5% similarity
                similarities.append((other_user, sim))
        
        # Keep top 20 similar users
        similarities.sort(key=lambda x: x[1], reverse=True)
        user_neighbors[test_user] = similarities[:20]

# ========== MATRIX FACTORIZATION ==========
print("\n5. Building matrix factorization model...")

# Create mappings
all_users = list(set(train_df['user_id'].unique()) | test_users)
all_items = list(quality_items['item_id'].unique())

user_to_idx = {user: idx for idx, user in enumerate(all_users)}
item_to_idx = {item: idx for idx, item in enumerate(all_items)}
idx_to_item = {idx: item for item, idx in item_to_idx.items()}

# Build sparse matrix
rows, cols, data = [], [], []
for _, row in train_df.iterrows():
    if row['user_id'] in user_to_idx and row['item_id'] in item_to_idx:
        rows.append(user_to_idx[row['user_id']])
        cols.append(item_to_idx[row['item_id']])
        # Weight by recency
        recency_weight = np.exp(-0.01 * (train_df['timestamp'].max() - row['timestamp']) / 86400000)
        data.append(recency_weight)

interaction_matrix = csr_matrix((data, (rows, cols)), shape=(len(all_users), len(all_items)))

# Train model
model = implicit.als.AlternatingLeastSquares(
    factors=32,
    regularization=0.2,
    iterations=15,
    random_state=42
)
model.fit(interaction_matrix.T.tocsr(), show_progress=False)

# ========== RECOMMENDATION FUNCTION ==========
def get_personalized_recommendations(user_id, k=10):
    """Generate personalized recommendations for each user"""
    
    recommendations = []
    scores = defaultdict(float)
    
    # Get user history
    user_items = test_user_items_set.get(user_id, set())
    user_history_list = test_user_history.get(user_id, [])
    
    if user_id in user_to_idx and len(user_items) > 0:
        # User with history
        user_idx = user_to_idx[user_id]
        
        # 1. Matrix factorization (30%)
        if user_idx < interaction_matrix.shape[0]:
            try:
                # Get user factors
                user_vector = interaction_matrix[user_idx]
                scores_mf = model.user_factors[user_idx] @ model.item_factors.T
                
                # Add scores for items in our quality pool
                for item_idx, score in enumerate(scores_mf):
                    if item_idx < len(idx_to_item):
                        item_id = idx_to_item[item_idx]
                        if item_id not in user_items:
                            scores[item_id] += 0.3 * score
            except:
                pass
        
        # 2. Item-based CF (30%)
        # Look at last 5 items
        recent_items = user_history_list[-5:] if len(user_history_list) > 0 else []
        for recent_item in recent_items:
            if recent_item in item_similarity:
                similar_items = item_similarity[recent_item]
                for sim_item, sim_score in sorted(similar_items.items(), key=lambda x: x[1], reverse=True)[:10]:
                    if sim_item not in user_items and sim_item in item_to_idx:
                        scores[sim_item] += 0.3 * sim_score / len(recent_items)
        
        # 3. User-based CF (20%)
        if user_id in user_neighbors:
            for neighbor, sim in user_neighbors[user_id]:
                neighbor_items = all_user_items.get(neighbor, set()) - user_items
                for item in neighbor_items:
                    if item in item_to_idx:
                        scores[item] += 0.2 * sim / len(user_neighbors[user_id])
        
        # 4. Quality boost (20%)
        for _, item_row in quality_items.iterrows():
            item_id = item_row['item_id']
            if item_id not in user_items:
                # Personalized quality score based on user's average rating
                user_avg_rating = 4.0  # Default
                if len(user_items) > 0:
                    user_item_ratings = item_meta_df[item_meta_df['item_id'].isin(user_items)]['average_rating']
                    if len(user_item_ratings) > 0:
                        user_avg_rating = user_item_ratings.mean()
                
                if abs(item_row['average_rating'] - user_avg_rating) < 0.5:  # Similar rating preference
                    scores[item_id] += 0.2 * item_row['quality_score'] / 10
    
    else:
        # Cold start user - use quality items with diversity
        quality_items_sorted = quality_items.sort_values('quality_score', ascending=False)
        
        # Add some randomness for diversity
        top_items = quality_items_sorted.head(50)
        selected_indices = np.random.choice(len(top_items), min(20, len(top_items)), replace=False)
        
        for idx in selected_indices:
            item_id = top_items.iloc[idx]['item_id']
            scores[item_id] = top_items.iloc[idx]['quality_score'] + np.random.normal(0, 0.1)
    
    # Sort and filter
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Ensure we only recommend quality items not in history
    for item_id, score in sorted_scores:
        if item_id not in user_items and item_id in quality_items['item_id'].values:
            recommendations.append(item_id)
        if len(recommendations) >= k:
            break
    
    # Fill if needed
    if len(recommendations) < k:
        # Get random quality items
        remaining_items = quality_items[~quality_items['item_id'].isin(recommendations + list(user_items))]
        if len(remaining_items) > 0:
            fill_items = remaining_items.sample(min(k - len(recommendations), len(remaining_items)))['item_id'].tolist()
            recommendations.extend(fill_items)
    
    return recommendations[:k]

# ========== GENERATE PREDICTIONS ==========
print("\n6. Generating personalized predictions...")

predictions = {}
unique_recommendations = set()

for i, user_id in enumerate(sample_submission['user_id'].values):
    if i % 5 == 0:
        print(f"  Processing user {i}/{len(sample_submission)}")
    
    recs = get_personalized_recommendations(user_id, k=10)
    predictions[user_id] = recs
    unique_recommendations.update(recs)

print(f"\nTotal unique items recommended: {len(unique_recommendations)}")

# ========== CREATE SUBMISSION ==========
print("\n7. Creating submission...")

submission_data = []
for idx, row in sample_submission.iterrows():
    user_id = row['user_id']
    recs = predictions[user_id]
    
    submission_data.append({
        'ID': row['ID'],
        'user_id': user_id,
        'item_id': ','.join(map(str, recs))
    })

submission_df = pd.DataFrame(submission_data)

# Verify diversity
all_recs = []
for _, row in submission_df.iterrows():
    all_recs.extend(row['item_id'].split(','))

rec_counter = Counter(all_recs)
print(f"\nDiversity check:")
print(f"  Unique items: {len(rec_counter)}")
print(f"  Most common item appears: {rec_counter.most_common(1)[0][1]} times")
print(f"  Top 5 most recommended:")
for item, count in rec_counter.most_common(5):
    print(f"    Item {item}: {count} times ({count/len(submission_df)*100:.1f}%)")

# Save
submission_df.to_csv('submission_personalized_v3.csv', index=False)
print("\nSubmission saved to 'submission_personalized_v3.csv'")