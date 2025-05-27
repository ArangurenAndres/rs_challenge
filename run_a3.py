import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import implicit
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
train_df = pd.read_csv('rs3/train.csv')
test_df = pd.read_csv('rs3/test.csv')
item_meta_df = pd.read_csv('rs3/item_meta.csv')
sample_submission = pd.read_csv('rs3/sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Item metadata shape: {item_meta_df.shape}")
print(f"Number of unique users in train: {train_df['user_id'].nunique()}")
print(f"Number of unique items in train: {train_df['item_id'].nunique()}")

# Analyze the data
print("\nTrain data sample:")
print(train_df.head())
print("\nTest data sample:")
print(test_df.head())

# Create user-item interaction matrix
print("\nCreating user-item interaction matrix...")

# Get all unique users and items
all_users = np.concatenate([train_df['user_id'].unique(), test_df['user_id'].unique()])
all_users = np.unique(all_users)
all_items = train_df['item_id'].unique()

# Create mappings
user_to_idx = {user: idx for idx, user in enumerate(all_users)}
idx_to_user = {idx: user for user, idx in user_to_idx.items()}
item_to_idx = {item: idx for idx, item in enumerate(all_items)}
idx_to_item = {idx: item for item, idx in item_to_idx.items()}

# Create sparse matrix
n_users = len(all_users)
n_items = len(all_items)

# Map train data to indices
train_df['user_idx'] = train_df['user_id'].map(user_to_idx)
train_df['item_idx'] = train_df['item_id'].map(item_to_idx)

# Create interaction matrix (using implicit feedback - all interactions are 1)
row = train_df['user_idx'].values
col = train_df['item_idx'].values
data = np.ones(len(train_df))

interaction_matrix = csr_matrix((data, (row, col)), shape=(n_users, n_items))

print(f"Interaction matrix shape: {interaction_matrix.shape}")
print(f"Sparsity: {1 - interaction_matrix.nnz / (n_users * n_items):.4f}")

# Model 1: Popularity-based baseline
print("\nCreating popularity-based recommendations...")
item_popularity = train_df['item_id'].value_counts().to_dict()
top_popular_items = train_df['item_id'].value_counts().head(50).index.tolist()

# Model 2: Collaborative Filtering using ALS (Alternating Least Squares)
print("\nTraining ALS model...")
# Implicit library expects item-user matrix (transpose of our user-item matrix)
# Also, it works with CSR format
item_user_matrix = interaction_matrix.T.tocsr()

model_als = implicit.als.AlternatingLeastSquares(
    factors=100,
    regularization=0.01,
    iterations=20,
    calculate_training_loss=True,
    random_state=42
)

# Train the model
model_als.fit(item_user_matrix)

# Model 3: SVD-based approach
print("\nTraining SVD model...")
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(interaction_matrix)
item_factors = svd.components_.T

# Normalize factors for better similarity computation
user_factors_norm = normalize(user_factors, axis=1)
item_factors_norm = normalize(item_factors, axis=1)

# Function to get recommendations for a user
def get_recommendations_ensemble(user_id, top_k=10):
    if user_id not in user_to_idx:
        # Cold start - return popular items
        return top_popular_items[:top_k]
    
    user_idx = user_to_idx[user_id]
    
    # Get already interacted items
    interacted_items = set(train_df[train_df['user_id'] == user_id]['item_id'].values)
    
    # Initialize final scores
    final_scores = {}
    
    # ALS predictions
    try:
        # Get recommendations
        als_items, als_scores = model_als.recommend(
            user_idx, 
            interaction_matrix[user_idx], 
            N=top_k * 2,  # Get more candidates
            filter_already_liked_items=True
        )
        
        # Add ALS scores
        for i in range(len(als_items)):
            item_idx = als_items[i]
            score = als_scores[i]
            # Ensure item_idx is valid
            if 0 <= item_idx < len(idx_to_item):
                item_id = idx_to_item[item_idx]
                if item_id not in interacted_items:
                    final_scores[item_id] = 0.7 * float(score)
    except Exception as e:
        # If ALS fails, we'll rely on SVD and popularity
        pass
    
    # SVD predictions
    scores_svd = user_factors_norm[user_idx] @ item_factors_norm.T
    
    # Add SVD scores
    top_svd_indices = np.argsort(scores_svd)[::-1][:top_k * 3]  # Get more candidates
    for item_idx in top_svd_indices:
        if item_idx < len(idx_to_item):  # Ensure valid index
            item_id = idx_to_item[item_idx]
            if item_id not in interacted_items:
                if item_id in final_scores:
                    final_scores[item_id] += 0.3 * scores_svd[item_idx]
                else:
                    final_scores[item_id] = 0.3 * scores_svd[item_idx]
    
    # Add popularity boost for diversification
    for item_id in top_popular_items[:20]:
        if item_id not in interacted_items:
            if item_id in final_scores:
                final_scores[item_id] += 0.1 * (item_popularity.get(item_id, 0) / max(item_popularity.values()))
            else:
                final_scores[item_id] = 0.05 * (item_popularity.get(item_id, 0) / max(item_popularity.values()))
    
    # Sort and get top k
    recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    rec_items = [item_id for item_id, _ in recommendations]
    
    # If we don't have enough recommendations, fill with popular items
    if len(rec_items) < top_k:
        for item in top_popular_items:
            if item not in rec_items and item not in interacted_items:
                rec_items.append(item)
            if len(rec_items) == top_k:
                break
    
    return rec_items[:top_k]

# Generate predictions for test users
print("\nGenerating predictions...")
predictions = {}

# Get test users from sample submission for exact order
test_users = sample_submission['user_id'].unique()

print(f"Number of test users: {len(test_users)}")

for i, user_id in enumerate(test_users):
    if i % 100 == 0:
        print(f"Processing user {i}/{len(test_users)}")
    
    recommendations = get_recommendations_ensemble(user_id, top_k=10)
    
    # Ensure we have exactly 10 recommendations
    if len(recommendations) < 10:
        # Fill with popular items not already recommended
        for item in top_popular_items:
            if item not in recommendations:
                recommendations.append(item)
            if len(recommendations) == 10:
                break
    
    predictions[user_id] = recommendations[:10]

# Create submission
print("\nCreating submission file...")
submission_data = []

# Create submission in the exact format
for idx, row in sample_submission.iterrows():
    user_id = row['user_id']
    
    # Get recommendations for this user
    if user_id in predictions:
        recs = predictions[user_id]
    else:
        # This shouldn't happen, but just in case
        recs = get_recommendations_ensemble(user_id, top_k=10)
        if len(recs) < 10:
            for item in top_popular_items:
                if item not in recs:
                    recs.append(item)
                if len(recs) == 10:
                    break
    
    # Create the row matching sample submission format
    submission_data.append({
        'ID': row['ID'],  # Use the ID from sample submission
        'user_id': user_id,
        'item_id': ','.join(map(str, recs[:10]))
    })

submission_df = pd.DataFrame(submission_data)

# Ensure columns are in the same order as sample submission
submission_df = submission_df[sample_submission.columns]

# Verify submission format
print(f"\nSubmission shape: {submission_df.shape}")
print("Submission sample:")
print(submission_df.head())

# Check that each user has exactly 10 recommendations
for idx, row in submission_df.iterrows():
    items = row['item_id'].split(',')
    assert len(items) == 10, f"User {row['user_id']} has {len(items)} recommendations"

# Save submission
submission_df.to_csv('rs3/saved_submission2.csv', index=False)
print("\nSubmission saved to 'submission.csv'")

# Additional Analysis and Improvements
print("\n=== Additional Analysis ===")

# Analyze item metadata for potential content-based filtering
print("\nItem metadata analysis:")
print(item_meta_df.head())
print(f"\nUnique categories: {item_meta_df['main_category'].nunique()}")
print(f"Average rating: {item_meta_df['average_rating'].mean():.2f}")

# User activity analysis
user_activity = train_df.groupby('user_id').size()
print(f"\nAverage items per user: {user_activity.mean():.2f}")
print(f"Max items per user: {user_activity.max()}")
print(f"Min items per user: {user_activity.min()}")

print("\n=== Model Performance Insights ===")
print("The ensemble model combines:")
print("1. ALS (70% weight) - Good for collaborative filtering")
print("2. SVD (30% weight) - Captures latent factors")
print("3. Popularity baseline - Handles cold start users")
print("\nTo improve further, consider:")
print("- Incorporating item metadata (categories, ratings)")
print("- Using temporal features (recent interactions)")
print("- Fine-tuning hyperparameters")
print("- Adding more sophisticated models (LightGCN, BERT4Rec)")