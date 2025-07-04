# goemotions_evaluation/utils/data_loader.py

"""
Data loading utilities for the GoEmotions dataset.
Simplified to ONLY handle GoEmotions format data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

def load_and_preprocess_goemotions_data(file_path: str):
    """
    Load and preprocess GoEmotions data.
    """
    print(f"Loading data from: {file_path}")
    
    # Read CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")
    
    # Process GoEmotions format
    print("Processing GoEmotions dataset format.")
    return process_goemotions_format(df)

def process_goemotions_format(df):
    """Process GoEmotions dataset format."""
    processed_data = []
    
    # The 28 GoEmotions categories
    categories = {
        'admiration': 'admiration',
        'amusement': 'amusement',
        'anger': 'anger',
        'annoyance': 'annoyance',
        'approval': 'approval',
        'caring': 'caring',
        'confusion': 'confusion',
        'curiosity': 'curiosity',
        'desire': 'desire',
        'disappointment': 'disappointment',
        'disapproval': 'disapproval',
        'disgust': 'disgust',
        'embarrassment': 'embarrassment',
        'excitement': 'excitement',
        'fear': 'fear',
        'gratitude': 'gratitude',
        'grief': 'grief',
        'joy': 'joy',
        'love': 'love',
        'nervousness': 'nervousness',
        'optimism': 'optimism',
        'pride': 'pride',
        'realization': 'realization',
        'relief': 'relief',
        'remorse': 'remorse',
        'sadness': 'sadness',
        'surprise': 'surprise',
        'neutral': 'neutral'
    }
    
    # Process each comment
    for idx, row in df.iterrows():
        if pd.isna(row.get('text', None)):
            continue
            
        # Basic comment info
        comment_data = {
            'comment_id': row.get('id', idx),
            'text': row['text'],
            'author': row.get('author', ''),
            'subreddit': row.get('subreddit', ''),
            'created_utc': row.get('created_utc', ''),
            'datasplit': 'train'  # Default split
        }
        
        # Extract emotion annotations (binary 0/1 values)
        for category in categories.keys():
            # Get the binary annotation (0 or 1)
            emotion_value = row.get(category, np.nan)
            comment_data[f'expert_{category}'] = bool(emotion_value) if not pd.isna(emotion_value) and emotion_value == 1 else False
        
        processed_data.append(comment_data)
    
    result_df = pd.DataFrame(processed_data)
    print(f"\nProcessed {len(result_df)} comments with emotion annotations")
    
    # Print annotation statistics
    print("\nEmotion annotation statistics:")
    for category in categories.keys():
        expert_col = f'expert_{category}'
        if expert_col in result_df.columns:
            count = result_df[expert_col].sum()
            total = len(result_df)
            print(f"  {category:15s}: {count}/{total} ({count/total*100:.1f}%)")
    
    return result_df

def get_comment_emotions(row: pd.Series, annotator: str = 'expert') -> List[str]:
    """
    Extract emotions assigned to a comment.
    
    Args:
        row: A row from the dataframe
        annotator: Which annotator to use ('expert' for GoEmotions ground truth)
    
    Returns:
        List of emotion names that were marked as True
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    assigned_emotions = []
    for emotion in emotions:
        col_name = f'{annotator}_{emotion}'
        if col_name in row and row[col_name] is True:
            assigned_emotions.append(emotion)
    
    return assigned_emotions

def filter_annotated_comments(df: pd.DataFrame, min_emotions: int = 1, 
                            annotator: str = 'expert') -> pd.DataFrame:
    """
    Filter to keep only comments that have at least min_emotions emotions assigned.
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # Count how many emotions each comment has
    emotion_counts = []
    for idx, row in df.iterrows():
        count = 0
        for emotion in emotions:
            col_name = f'{annotator}_{emotion}'
            if col_name in row and row[col_name] is True:
                count += 1
        emotion_counts.append(count)
    
    df['emotion_count'] = emotion_counts
    filtered_df = df[df['emotion_count'] >= min_emotions].copy()
    
    print(f"Filtered to {len(filtered_df)} comments with at least {min_emotions} emotions")
    return filtered_df

def get_emotion_statistics(df: pd.DataFrame) -> Dict:
    """
    Get statistics about the GoEmotions dataset.
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    stats = {
        'total_comments': len(df),
        'emotion_distribution': {},
        'multi_label_stats': {}
    }
    
    # Emotion distribution
    for emotion in emotions:
        expert_col = f'expert_{emotion}'
        if expert_col in df.columns:
            count = df[expert_col].sum()
            stats['emotion_distribution'][emotion] = {
                'count': count,
                'percentage': count / len(df) * 100
            }
    
    # Multi-label statistics
    emotion_counts_per_comment = []
    for _, row in df.iterrows():
        emotions_assigned = get_comment_emotions(row, 'expert')
        emotion_counts_per_comment.append(len(emotions_assigned))
    
    from collections import Counter
    emotion_count_dist = Counter(emotion_counts_per_comment)
    
    stats['multi_label_stats'] = {
        'avg_emotions_per_comment': np.mean(emotion_counts_per_comment),
        'emotion_count_distribution': dict(emotion_count_dist),
        'single_emotion_percentage': emotion_count_dist.get(1, 0) / len(df) * 100,
        'multi_emotion_percentage': sum(count for num, count in emotion_count_dist.items() if num > 1) / len(df) * 100
    }
    
    return stats

def check_data_format(df: pd.DataFrame):
    """
    Check the GoEmotions data format.
    """
    print("=== GOEMOTIONS DATA FORMAT CHECK ===")
    
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    print(f"Total rows: {len(df)}")
    print(f"Comments with text: {df['text'].notna().sum()}")
    print()
    
    for emotion in emotions:
        if emotion in df.columns:
            col_data = df[emotion].dropna()
            unique_values = sorted(col_data.unique())
            
            print(f"{emotion:15s}: unique values = {unique_values}")
            
            if all(val in [0, 1, 0.0, 1.0] for val in unique_values):
                ones = (col_data == 1).sum()
                zeros = (col_data == 0).sum()
                print(f"{'':17s}  ✅ Binary: {ones} ones, {zeros} zeros ({ones/(ones+zeros)*100:.1f}% positive)")
            else:
                print(f"{'':17s}  ⚠️  Contains non-binary values!")
        else:
            print(f"{emotion:15s}: MISSING COLUMN")
    
    print(f"\n=== RESULT ===")
    print("✅ GoEmotions dataset uses binary emotion labels (0/1)")
    print("📝 Multi-label: Comments can have multiple emotions simultaneously")

if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        check_data_format(df)
    else:
        print("Usage: python data_loader.py path/to/goemotions_data.csv")