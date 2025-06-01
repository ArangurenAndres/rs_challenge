import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
import implicit
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=== OPTIMIZED RECOMMENDER SYSTEM ===")
print("Designed for extreme sparsity and temporal patterns\n")

# Load data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
item_meta_df = pd.read_csv('item_meta.csv')
sample_submission = pd.read_csv('submission_v3.csv')

# Load pre-computed statistics
item_stats = pd.read_csv('item_statistics.csv')

print(f"Data loaded. Processing {len(train_df)} interactions...")

# ========== TEMPORAL PREPROCESSING ==========
print("\n1. Adding temporal features...")

# Convert timestamps and add temporal decay
max_timestamp = train_df['timestamp'].max()
min_timestamp = train_df['timestamp'].min()
time_range = max_timestamp - min_timestamp

# Strong temporal decay - recent interactions are MUCH more important
train_df['recency_score'] = np.exp(3.0 * (train_df['timestamp'] - min_timestamp) / time_range)
train_df['days_ago'] = (max_timestamp - train_df['timestamp']) / 86400000

# ========== USER SEGMENTATION ==========
print("\n2. Segmenting users...")

user_interaction_counts = train_df['user_id'].value_counts()
power_users = set(user_interaction_counts[user_interaction_counts >= 10].index)
regular_users = set(user_interaction_counts[(user_interaction_counts >= 2) & (user_interaction_counts < 10)].index)
single_interaction_users = set(user_interaction_counts[user_interaction_counts == 1].index)

print(f"  Power users (10+ interactions): {len(power_users)}")
print(f"  Regular users (2-9 interactions): {len(regular_users)}")
print(f"  Single interaction users: {len(single_interaction_users)}")

# ========== POPULARITY MODELS ==========
print("\n3. Building popularity models...")

# Global popularity with strong recency bias
item_popularity_weighted = train_df.groupby('item_id')['recency_score'].sum().to_dict()

# Recent popularity (last 30 days)
recent_mask = train_df['days_ago'] <= 30
recent_popularity = train_df[recent_mask]['item_id'].value_counts().to_dict()

# Very recent popularity (last 7 days)
very_recent_mask = train_df['days_ago'] <= 7
very_recent_popularity = train_df[very_recent_mask]['item_id'].value_counts().to_dict()

# Category-specific popularity
category_popularity = {}
for category in ['All Beauty', 'Premium Beauty']:
    cat_items = item_meta_df[item_meta_df['main_category'] == category]['item_id'].values
    cat_interactions = train_df[train_df['item_id'].isin(cat_items)]
    category_popularity[category] = cat_interactions.groupby('item_id')['recency_score'].sum().to_dict()

# ========== CO-OCCURRENCE PATTERNS ==========
print("\n4. Building co-occurrence model...")

# Build co-occurrence for users with multiple interactions
user_sequences = train_df.sort_values(['user_id', 'timestamp']).groupby('user_id')['item_id'].apply(list).to_dict()

item_to_items = defaultdict(Counter)
for user, items in user_sequences.items():
    if len(items) >= 2:  # Only users with multiple interactions
        # Weight recent items more
        for i in range(len(items)):
            for j in range(max(0, i-5), min(len(items), i+5)):
                if i != j:
                    weight = 1.0 / (1 + abs(i-j))  # Distance decay
                    item_to_items[items[i]][items[j]] += weight

# ========== COLLABORATIVE FILTERING ==========
print("\n5. Building collaborative filtering model...")

# Create mappings
all_users = np.unique(np.concatenate([train_df['user_id'].unique(), test_df['user_id'].unique()]))
all_items = train_df['item_id'].unique()

user_to_idx = {user: idx for idx, user in enumerate(all_users)}
idx_to_user = {idx: user for user, idx in user_to_idx.items()}
item_to_idx = {item: idx for idx, item in enumerate(all_items)}
idx_to_item = {idx: item for item, idx in item_to_idx.items()}

# Create weighted interaction matrix
train_df['user_idx'] = train_df['user_id'].map(user_to_idx)
train_df['item_idx'] = train_df['item_id'].map(item_to_idx)

row = train_df['user_idx'].values
col = train_df['item_idx'].values
data = train_df['recency_score'].values  # Use recency as weight

interaction_matrix = csr_matrix((data, (row, col)), shape=(len(all_users), len(all_items)))

# Train BPR model (best for implicit feedback with extreme sparsity)
model_bpr = implicit.bpr.BayesianPersonalizedRanking(
    factors=64,  # Lower dimension due to sparsity
    learning_rate=0.05,
    regularization=0.1,
    iterations=30,
    verify_negative_samples=True,
    random_state=42
)

print("  Training BPR model...")
model_bpr.fit(interaction_matrix.T.tocsr(), show_progress=False)

# ========== RECOMMENDATION FUNCTION ==========
def get_recommendations(user_id, k=10):
    """Hybrid recommendation strategy based on user type"""
    
    # Check user type
    if user_id not in user_to_idx:
        # Cold start - use very recent popularity
        items = sorted(very_recent_popularity.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in items[:k]]
    
    user_idx = user_to_idx[user_id]
    user_history = set(train_df[train_df['user_id'] == user_id]['item_id'].values)
    interaction_count = len(user_history)
    
    recommendations = []
    scores = defaultdict(float)
    
    # Strategy based on user type
    if interaction_count == 1:
        # Single interaction users - mostly popularity with some personalization
        
        # 1. Co-occurrence from their single item (30%)
        user_item = list(user_history)[0]
        if user_item in item_to_items:
            for item, count in item_to_items[user_item].most_common(20):
                if item not in user_history:
                    scores[item] += 0.3 * count / sum(item_to_items[user_item].values())
        
        # 2. Recent popularity (50%)
        for item_id, pop_score in recent_popularity.items():
            if item_id not in user_history:
                scores[item_id] += 0.5 * pop_score / max(recent_popularity.values())
        
        # 3. Very recent popularity (20%)
        for item_id, pop_score in very_recent_popularity.items():
            if item_id not in user_history:
                scores[item_id] += 0.2 * pop_score / max(very_recent_popularity.values())
    
    elif interaction_count < 10:
        # Regular users - balanced approach
        
        # 1. BPR recommendations (40%)
        try:
            bpr_items, bpr_scores = model_bpr.recommend(
                user_idx,
                interaction_matrix[user_idx],
                N=k*3,
                filter_already_liked_items=True
            )
            for i in range(len(bpr_items)):
                if bpr_items[i] < len(idx_to_item):
                    item_id = idx_to_item[bpr_items[i]]
                    scores[item_id] += 0.4 * float(bpr_scores[i])
        except:
            pass
        
        # 2. Co-occurrence (30%)
        for hist_item in list(user_history)[-5:]:  # Recent 5 items
            if hist_item in item_to_items:
                for item, count in item_to_items[hist_item].most_common(10):
                    if item not in user_history:
                        scores[item] += 0.3 * count / (5 * max(item_to_items[hist_item].values()))
        
        # 3. Recent popularity (30%)
        for item_id, pop_score in recent_popularity.items():
            if item_id not in user_history:
                scores[item_id] += 0.3 * pop_score / max(recent_popularity.values())
    
    else:
        # Power users - personalization focused
        
        # 1. BPR recommendations (60%)
        try:
            bpr_items, bpr_scores = model_bpr.recommend(
                user_idx,
                interaction_matrix[user_idx],
                N=k*2,
                filter_already_liked_items=True
            )
            for i in range(len(bpr_items)):
                if bpr_items[i] < len(idx_to_item):
                    item_id = idx_to_item[bpr_items[i]]
                    scores[item_id] += 0.6 * float(bpr_scores[i])
        except:
            pass
        
        # 2. Co-occurrence (30%)
        for hist_item in list(user_history)[-10:]:
            if hist_item in item_to_items:
                for item, count in item_to_items[hist_item].most_common(5):
                    if item not in user_history:
                        scores[item] += 0.3 * count / (10 * max(item_to_items[hist_item].values()))
        
        # 3. Trending items (10%)
        for item_id, pop_score in very_recent_popularity.items():
            if item_id not in user_history:
                scores[item_id] += 0.1 * pop_score / max(very_recent_popularity.values())
    
    # Sort and get top k
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = [item_id for item_id, _ in sorted_items[:k]]
    
    # Fill with recent popular items if needed
    if len(recommendations) < k:
        backup_items = sorted(recent_popularity.items(), key=lambda x: x[1], reverse=True)
        for item_id, _ in backup_items:
            if item_id not in user_history and item_id not in recommendations:
                recommendations.append(item_id)
            if len(recommendations) == k:
                break
    
    return recommendations[:k]

# ========== GENERATE PREDICTIONS ==========
print("\n6. Generating predictions...")

predictions = {}
test_users = sample_submission['user_id'].unique()

for i, user_id in enumerate(test_users):
    if i % 100 == 0:
        print(f"  Processing user {i}/{len(test_users)}")
    
    recommendations = get_recommendations(user_id, k=10)
    predictions[user_id] = recommendations

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
submission_df = submission_df[sample_submission.columns]

# Verify submission
assert len(submission_df) == len(sample_submission)
for idx, row in submission_df.iterrows():
    items = row['item_id'].split(',')
    assert len(items) == 10, f"User {row['user_id']} has {len(items)} recommendations"

# Save submission
submission_df.to_csv('submission_optimized.csv', index=False)

print("\n=== SUBMISSION COMPLETE ===")
print(f"Submission saved to 'submission_optimized.csv'")
print(f"\nStrategy summary:")
print(f"- {len(single_interaction_users)} users: Popularity + co-occurrence")
print(f"- {len(regular_users)} users: Balanced hybrid approach")
print(f"- {len(power_users)} users: Personalized recommendations")
print(f"\nKey optimizations:")
print(f"- Strong temporal weighting (3x exponential)")
print(f"- User-specific strategies")
print(f"- Co-occurrence patterns")
print(f"- Recent popularity emphasis")