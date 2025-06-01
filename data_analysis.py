import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

print("=== RECOMMENDER SYSTEM DATA ANALYSIS ===\n")

# Load all data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
item_meta_df = pd.read_csv('item_meta.csv')
sample_submission = pd.read_csv('submission_v3.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Item metadata shape: {item_meta_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# ========== 1. BASIC STATISTICS ==========
print("\n=== BASIC STATISTICS ===")
print(f"Unique users in train: {train_df['user_id'].nunique()}")
print(f"Unique items in train: {train_df['item_id'].nunique()}")
print(f"Unique users in test: {test_df['user_id'].nunique()}")
print(f"Total interactions: {len(train_df)}")
print(f"Sparsity: {1 - len(train_df) / (train_df['user_id'].nunique() * train_df['item_id'].nunique()):.6f}")

# Check test users
test_users = set(test_df['user_id'].unique())
train_users = set(train_df['user_id'].unique())
cold_start_users = test_users - train_users
print(f"\nCold start users (in test but not in train): {len(cold_start_users)} ({len(cold_start_users)/len(test_users)*100:.1f}%)")

# ========== 2. USER BEHAVIOR ANALYSIS ==========
print("\n=== USER BEHAVIOR ANALYSIS ===")
user_stats = train_df.groupby('user_id').agg({
    'item_id': ['count', 'nunique'],
    'timestamp': ['min', 'max']
}).reset_index()
user_stats.columns = ['user_id', 'total_interactions', 'unique_items', 'first_timestamp', 'last_timestamp']

print(f"\nInteractions per user:")
print(f"  Mean: {user_stats['total_interactions'].mean():.2f}")
print(f"  Median: {user_stats['total_interactions'].median():.2f}")
print(f"  Min: {user_stats['total_interactions'].min()}")
print(f"  Max: {user_stats['total_interactions'].max()}")

# User distribution
interaction_counts = user_stats['total_interactions'].value_counts().sort_index()
print(f"\nUser distribution by interaction count:")
print(f"  1 interaction: {interaction_counts.get(1, 0)} users")
print(f"  2-5 interactions: {interaction_counts[2:6].sum()} users")
print(f"  6-10 interactions: {interaction_counts[6:11].sum()} users")
print(f"  11-20 interactions: {interaction_counts[11:21].sum()} users")
print(f"  21-50 interactions: {interaction_counts[21:51].sum()} users")
print(f"  50+ interactions: {interaction_counts[51:].sum()} users")

# ========== 3. ITEM POPULARITY ANALYSIS ==========
print("\n=== ITEM POPULARITY ANALYSIS ===")
item_stats = train_df.groupby('item_id').agg({
    'user_id': ['count', 'nunique'],
    'timestamp': ['min', 'max']
}).reset_index()
item_stats.columns = ['item_id', 'total_interactions', 'unique_users', 'first_timestamp', 'last_timestamp']

print(f"\nInteractions per item:")
print(f"  Mean: {item_stats['total_interactions'].mean():.2f}")
print(f"  Median: {item_stats['total_interactions'].median():.2f}")
print(f"  Min: {item_stats['total_interactions'].min()}")
print(f"  Max: {item_stats['total_interactions'].max()}")

# Top items
top_items = item_stats.nlargest(20, 'total_interactions')
print(f"\nTop 10 most popular items:")
for idx, row in top_items.head(10).iterrows():
    item_title = item_meta_df[item_meta_df['item_id'] == row['item_id']]['title'].values
    title = item_title[0][:50] + '...' if len(item_title) > 0 and len(item_title[0]) > 50 else item_title[0] if len(item_title) > 0 else 'Unknown'
    print(f"  Item {row['item_id']}: {row['total_interactions']} interactions - {title}")

# ========== 4. TEMPORAL ANALYSIS ==========
print("\n=== TEMPORAL ANALYSIS ===")
# Convert timestamps
train_df['timestamp_dt'] = pd.to_datetime(train_df['timestamp'], unit='ms')
print(f"Date range: {train_df['timestamp_dt'].min()} to {train_df['timestamp_dt'].max()}")
print(f"Time span: {(train_df['timestamp_dt'].max() - train_df['timestamp_dt'].min()).days} days")

# Temporal patterns
train_df['hour'] = train_df['timestamp_dt'].dt.hour
train_df['day_of_week'] = train_df['timestamp_dt'].dt.dayofweek
train_df['day'] = train_df['timestamp_dt'].dt.date

daily_interactions = train_df.groupby('day').size()
print(f"\nDaily interactions:")
print(f"  Mean: {daily_interactions.mean():.2f}")
print(f"  Min: {daily_interactions.min()}")
print(f"  Max: {daily_interactions.max()}")

# Check if there's growth over time
first_week = daily_interactions.iloc[:7].mean()
last_week = daily_interactions.iloc[-7:].mean()
print(f"\nTemporal trend:")
print(f"  First week avg: {first_week:.2f}")
print(f"  Last week avg: {last_week:.2f}")
print(f"  Growth: {(last_week/first_week - 1)*100:.1f}%")

# ========== 5. CATEGORY ANALYSIS ==========
print("\n=== CATEGORY ANALYSIS ===")
# Merge with metadata
train_with_meta = train_df.merge(item_meta_df[['item_id', 'main_category', 'price', 'average_rating']], 
                                 on='item_id', how='left')

category_stats = train_with_meta.groupby('main_category').agg({
    'user_id': 'count',
    'item_id': 'nunique',
    'price': 'mean',
    'average_rating': 'mean'
}).sort_values('user_id', ascending=False)

print(f"\nTop 15 categories by interactions:")
for cat, row in category_stats.head(15).iterrows():
    print(f"  {cat}: {row['user_id']} interactions, {row['item_id']} unique items, avg price ${row['price']:.2f}")

# ========== 6. PRICE ANALYSIS ==========
print("\n=== PRICE ANALYSIS ===")
price_stats = item_meta_df['price'].describe()
print(f"Price distribution:")
print(f"  Mean: ${price_stats['mean']:.2f}")
print(f"  Median: ${price_stats['50%']:.2f}")
print(f"  Min: ${price_stats['min']:.2f}")
print(f"  Max: ${price_stats['max']:.2f}")
print(f"  Items with price > $1000: {(item_meta_df['price'] > 1000).sum()}")
print(f"  Items with missing price: {item_meta_df['price'].isna().sum()}")

# ========== 7. REPEAT INTERACTION ANALYSIS ==========
print("\n=== REPEAT INTERACTION ANALYSIS ===")
repeat_interactions = train_df.groupby(['user_id', 'item_id']).size().reset_index(name='count')
repeat_users = repeat_interactions[repeat_interactions['count'] > 1]
print(f"Users who interacted with same item multiple times: {repeat_users['user_id'].nunique()}")
print(f"Total repeat interactions: {(repeat_interactions['count'] - 1).sum()}")
print(f"Max times a user interacted with same item: {repeat_interactions['count'].max()}")

# ========== 8. SEQUENCE ANALYSIS ==========
print("\n=== SEQUENCE ANALYSIS ===")
# Sample user sequences
sample_users = train_df['user_id'].value_counts().head(5).index
print("Sample user interaction sequences (top 5 most active users):")
for user in sample_users:
    user_data = train_df[train_df['user_id'] == user].sort_values('timestamp')
    user_items = user_data.merge(item_meta_df[['item_id', 'main_category']], on='item_id', how='left')
    categories = user_items['main_category'].tolist()
    print(f"\nUser {user} ({len(user_data)} interactions):")
    print(f"  Category sequence:  ->  ({categories[:10]})...")
    
    # Time between interactions
    user_data['time_diff'] = user_data['timestamp'].diff() / (1000 * 60 * 60 * 24)  # Convert to days
    print(f"  Avg days between interactions: {user_data['time_diff'].mean():.2f}")

# ========== 9. CO-OCCURRENCE PATTERNS ==========
print("\n=== CO-OCCURRENCE PATTERNS ===")
# Find items frequently bought together
user_baskets = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

# Count co-occurrences for top items
top_100_items = item_stats.nlargest(100, 'total_interactions')['item_id'].tolist()
cooccurrence_matrix = defaultdict(Counter)

for user, items in user_baskets.items():
    items_in_top = [item for item in items if item in top_100_items]
    for i in range(len(items_in_top)):
        for j in range(i+1, min(i+10, len(items_in_top))):  # Window of 10
            cooccurrence_matrix[items_in_top[i]][items_in_top[j]] += 1
            cooccurrence_matrix[items_in_top[j]][items_in_top[i]] += 1

# Show some examples
print("Sample item co-occurrences (for top items):")
sample_item = top_100_items[0]
if sample_item in cooccurrence_matrix:
    top_cooc = sorted(cooccurrence_matrix[sample_item].items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nItems frequently interacted after item {sample_item}:")
    for item, count in top_cooc:
        print(f"  Item {item}: {count} times")

# ========== 10. TEST SET ANALYSIS ==========
print("\n=== TEST SET ANALYSIS ===")
# Analyze test users
test_user_history = []
for user in test_df['user_id'].unique()[:100]:  # Sample 100 users
    if user in train_users:
        history_count = len(train_df[train_df['user_id'] == user])
        test_user_history.append(history_count)

if test_user_history:
    print(f"Test users with history in train (sample of 100):")
    print(f"  Mean interactions: {np.mean(test_user_history):.2f}")
    print(f"  Median interactions: {np.median(test_user_history):.2f}")

# ========== 11. RATING ANALYSIS ==========
print("\n=== RATING ANALYSIS ===")
rating_dist = item_meta_df['average_rating'].value_counts().sort_index()
print(f"Rating distribution:")
for rating, count in rating_dist.items():
    print(f"  Rating {rating}: {count} items ({count/len(item_meta_df)*100:.1f}%)")

# High rated items popularity
high_rated_items = item_meta_df[item_meta_df['average_rating'] >= 4.5]['item_id']
high_rated_interactions = train_df[train_df['item_id'].isin(high_rated_items)]
print(f"\nHigh-rated items (4.5+) account for {len(high_rated_interactions)/len(train_df)*100:.1f}% of interactions")

# ========== 12. FEATURE IMPORTANCE INDICATORS ==========
print("\n=== FEATURE IMPORTANCE INDICATORS ===")
# Check if recent items are more popular
train_df['days_since_first'] = (train_df['timestamp'] - train_df['timestamp'].min()) / (86400000)
recent_items = train_df[train_df['days_since_first'] > train_df['days_since_first'].max() * 0.8]['item_id'].value_counts()
old_items = train_df[train_df['days_since_first'] < train_df['days_since_first'].max() * 0.2]['item_id'].value_counts()

print(f"Recent items (last 20% of time) avg popularity: {recent_items.mean():.2f}")
print(f"Old items (first 20% of time) avg popularity: {old_items.mean():.2f}")

# ========== SAVE KEY STATISTICS ==========
print("\n=== SAVING KEY STATISTICS ===")
key_stats = {
    'n_users_train': train_df['user_id'].nunique(),
    'n_items_train': train_df['item_id'].nunique(),
    'n_interactions': len(train_df),
    'n_test_users': len(test_users),
    'n_cold_start_users': len(cold_start_users),
    'avg_interactions_per_user': user_stats['total_interactions'].mean(),
    'avg_interactions_per_item': item_stats['total_interactions'].mean(),
    'top_category': category_stats.index[0],
    'time_span_days': (train_df['timestamp_dt'].max() - train_df['timestamp_dt'].min()).days,
    'sparsity': 1 - len(train_df) / (train_df['user_id'].nunique() * train_df['item_id'].nunique())
}

# Save for model building
pd.DataFrame([key_stats]).to_csv('key_statistics.csv', index=False)
user_stats.to_csv('user_statistics.csv', index=False)
item_stats.to_csv('item_statistics.csv', index=False)

print("\n=== ANALYSIS COMPLETE ===")
print("\nKey insights saved to CSV files.")
print("\nBased on this analysis, I'll create an optimized model that:")
print("1. Handles the cold start problem")
print("2. Leverages temporal patterns")
print("3. Uses category preferences")
print("4. Exploits co-occurrence patterns")
print("5. Balances popular and personalized recommendations")