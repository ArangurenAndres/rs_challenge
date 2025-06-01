import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset():
    """Comprehensive analysis of the recommendation dataset"""
    
    print("=" * 60)
    print("RECOMMENDATION SYSTEM DATA ANALYSIS")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading datasets...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"✓ Training data loaded: {train_df.shape}")
    print(f"✓ Test data loaded: {test_df.shape}")
    
    # Basic statistics
    print("\n2. Basic Dataset Statistics:")
    print(f"   - Train columns: {list(train_df.columns)}")
    print(f"   - Test columns: {list(test_df.columns)}")
    print(f"   - Total interactions in train: {len(train_df):,}")
    print(f"   - Unique users in train: {train_df['user_id'].nunique():,}")
    print(f"   - Unique items in train: {train_df['item_id'].nunique():,}")
    print(f"   - Unique users in test: {test_df['user_id'].nunique():,}")
    
    # Sparsity analysis
    n_users = train_df['user_id'].nunique()
    n_items = train_df['item_id'].nunique()
    n_interactions = len(train_df)
    sparsity = 1 - (n_interactions / (n_users * n_items))
    print(f"   - Matrix sparsity: {sparsity:.6f} ({sparsity*100:.4f}% empty)")
    
    # User behavior analysis
    print("\n3. User Behavior Analysis:")
    user_interactions = train_df['user_id'].value_counts()
    print(f"   - Average interactions per user: {user_interactions.mean():.2f}")
    print(f"   - Median interactions per user: {user_interactions.median():.2f}")
    print(f"   - Min interactions per user: {user_interactions.min()}")
    print(f"   - Max interactions per user: {user_interactions.max()}")
    
    # Distribution of user interactions
    print(f"   - Users with 1 interaction: {(user_interactions == 1).sum():,} ({(user_interactions == 1).sum()/len(user_interactions)*100:.1f}%)")
    print(f"   - Users with 2-5 interactions: {((user_interactions >= 2) & (user_interactions <= 5)).sum():,}")
    print(f"   - Users with 6-10 interactions: {((user_interactions >= 6) & (user_interactions <= 10)).sum():,}")
    print(f"   - Users with >10 interactions: {(user_interactions > 10).sum():,}")
    
    # Item popularity analysis
    print("\n4. Item Popularity Analysis:")
    item_interactions = train_df['item_id'].value_counts()
    print(f"   - Average interactions per item: {item_interactions.mean():.2f}")
    print(f"   - Median interactions per item: {item_interactions.median():.2f}")
    print(f"   - Min interactions per item: {item_interactions.min()}")
    print(f"   - Max interactions per item: {item_interactions.max()}")
    
    # Cold start analysis
    print("\n5. Cold Start Analysis:")
    train_users = set(train_df['user_id'].unique())
    test_users = set(test_df['user_id'].unique())
    cold_start_users = test_users - train_users
    print(f"   - Cold start users (in test but not train): {len(cold_start_users):,}")
    print(f"   - Warm start users (in both): {len(test_users & train_users):,}")
    print(f"   - Cold start ratio: {len(cold_start_users)/len(test_users)*100:.2f}%")
    
    # Temporal analysis (if timestamp available)
    print("\n6. Temporal Analysis:")
    if 'timestamp' in train_df.columns:
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], unit='ms', errors='coerce')
        if not train_df['timestamp'].isna().all():
            date_range = train_df['timestamp'].max() - train_df['timestamp'].min()
            print(f"   - Date range: {date_range}")
            print(f"   - Start date: {train_df['timestamp'].min()}")
            print(f"   - End date: {train_df['timestamp'].max()}")
            
            # Temporal patterns
            train_df['hour'] = train_df['timestamp'].dt.hour
            train_df['day_of_week'] = train_df['timestamp'].dt.dayofweek
            
            print(f"   - Peak hour: {train_df['hour'].mode().iloc[0]}:00")
            print(f"   - Peak day: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][train_df['day_of_week'].mode().iloc[0]]}")
        else:
            print("   - Timestamp data appears to be invalid")
    else:
        print("   - No timestamp data available")
    
    # Try to load item metadata
    print("\n7. Item Metadata Analysis:")
    try:
        # Load first few rows to understand structure
        item_meta_sample = pd.read_csv('item_meta.csv', nrows=1000)
        print(f"   - Item metadata available: {item_meta_sample.shape}")
        print(f"   - Metadata columns: {list(item_meta_sample.columns)}")
        
        # Check for text features
        text_columns = []
        numeric_columns = []
        for col in item_meta_sample.columns:
            if col != 'item_id':
                dtype = item_meta_sample[col].dtype
                if dtype == 'object':
                    text_columns.append(col)
                else:
                    numeric_columns.append(col)
        
        print(f"   - Text features: {text_columns}")
        print(f"   - Numeric features: {numeric_columns}")
        
        # Check coverage
        meta_items = set(item_meta_sample['item_id'].unique())
        train_items = set(train_df['item_id'].unique())
        coverage = len(meta_items & train_items) / len(train_items)
        print(f"   - Metadata coverage: {coverage*100:.2f}% of training items")
        
    except Exception as e:
        print(f"   - Error loading item metadata: {str(e)}")
    
    # Model recommendations
    print("\n" + "=" * 60)
    print("MODELING RECOMMENDATIONS")
    print("=" * 60)
    
    print("\nBased on the data analysis, here are the recommended approaches:")
    
    print("\n🔥 HIGH PRIORITY TECHNIQUES:")
    if sparsity > 0.999:
        print("   ✓ Matrix Factorization (SVD/NMF) - Very sparse data")
        print("   ✓ Implicit Collaborative Filtering (ALS) - Binary interactions")
    
    if len(cold_start_users) > 0:
        print("   ✓ Popularity-based baseline - Cold start users")
        
    if len(text_columns) > 0:
        print("   ✓ Content-based filtering - Rich metadata available")
        print("   ✓ Hybrid approach combining collaborative + content")
    
    print("\n⚡ MEDIUM PRIORITY TECHNIQUES:")
    if user_interactions.std() > user_interactions.mean():
        print("   ✓ User-based collaborative filtering - Varied user behavior")
        
    if item_interactions.std() > item_interactions.mean():
        print("   ✓ Item-based collaborative filtering - Varied item popularity")
        
    if 'timestamp' in train_df.columns:
        print("   ✓ Sequential/Temporal models - Temporal data available")
        print("   ✓ BERT4Rec or other sequence models")
    
    print("\n🚀 ADVANCED TECHNIQUES (if computational resources allow):")
    print("   ✓ Neural Collaborative Filtering (NCF)")
    print("   ✓ Graph-based methods (LightGCN) - Good for sparse data")
    print("   ✓ Variational Autoencoders for recommendations")
    print("   ✓ Multi-task learning (if user/item features available)")
    
    print("\n📊 ENSEMBLE STRATEGY:")
    print("   ✓ Weighted combination of multiple models")
    print("   ✓ Stacking with meta-learner")
    print("   ✓ Model selection based on user/item characteristics")
    
    return {
        'n_users': n_users,
        'n_items': n_items,
        'sparsity': sparsity,
        'cold_start_ratio': len(cold_start_users)/len(test_users),
        'has_metadata': len(text_columns) > 0,
        'has_temporal': 'timestamp' in train_df.columns,
        'user_interactions_stats': {
            'mean': user_interactions.mean(),
            'std': user_interactions.std(),
            'median': user_interactions.median()
        },
        'item_interactions_stats': {
            'mean': item_interactions.mean(),
            'std': item_interactions.std(),
            'median': item_interactions.median()
        }
    }

if __name__ == "__main__":
    stats = analyze_dataset()
    print(f"\nAnalysis complete! Statistics saved.") 