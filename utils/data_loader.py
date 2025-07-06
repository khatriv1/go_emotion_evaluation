"""
Data loader for GoEmotions dataset with proper multi-label handling
Fixed to include missing import functions that existing evaluation files expect
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import config

class GoEmotionsDataLoader:
    """Data loader for GoEmotions multi-label emotion classification dataset"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.emotions_list = config.GOEMOTIONS_EMOTIONS
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and process GoEmotions data
        
        Args:
            sample_size: Number of samples to return (default: 100 for testing, None for all)
            
        Returns:
            Tuple of (processed_dataframe, emotions_list)
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load raw data
        df = pd.read_csv(self.data_path)
        self.logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Process the data
        df = self._process_goemotions_format(df)
        
        # Apply sampling if requested
        if sample_size is not None:
            if sample_size < len(df):
                df = df.sample(n=sample_size, random_state=config.RANDOM_SEED)
                print(f"Sampled {len(df)} comments for evaluation")
        elif len(df) > 1000:  # If no sample size specified but data is large, use reasonable default
            df = df.sample(n=100, random_state=config.RANDOM_SEED)
            print(f"Using default sample of {len(df)} comments for testing")
        
        # Validate the processed data
        self._validate_data(df)
        
        return df, self.emotions_list
    
    def _process_goemotions_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process GoEmotions dataset format"""
        
        # Check if we have the expected columns
        text_column = None
        if 'text' in df.columns:
            text_column = 'text'
        elif 'comment_text' in df.columns:
            text_column = 'comment_text'
        elif 'comment' in df.columns:
            text_column = 'comment'
        else:
            # Find any column that might contain text
            for col in df.columns:
                if 'text' in col.lower() or 'comment' in col.lower():
                    text_column = col
                    break
        
        if text_column is None:
            raise ValueError(f"No text column found. Available columns: {list(df.columns)}")
        
        # Rename text column to standard name
        if text_column != 'text':
            df = df.rename(columns={text_column: 'text'})
        
        # Ensure we have all emotion columns
        for emotion in self.emotions_list:
            if emotion not in df.columns:
                self.logger.warning(f"Emotion column '{emotion}' not found, setting to 0")
                df[emotion] = 0
        
        # Convert emotion columns to binary (0/1)
        for emotion in self.emotions_list:
            df[emotion] = pd.to_numeric(df[emotion], errors='coerce').fillna(0)
            df[emotion] = (df[emotion] > 0.5).astype(int)
        
        # Create emotion lists for each comment
        df['emotions'] = df[self.emotions_list].apply(
            lambda row: [emotion for emotion, value in row.items() if value == 1], 
            axis=1
        )
        
        # Filter out comments with no emotions (if desired)
        initial_count = len(df)
        df = df[df['emotions'].apply(len) > 0]
        
        if len(df) < initial_count:
            self.logger.info(f"Filtered to {len(df)} comments with at least 1 emotion")
            print(f"Filtered to {len(df)} comments with at least 1 emotions")
        
        # Print emotion statistics
        self._print_emotion_statistics(df)
        
        return df
    
    def _print_emotion_statistics(self, df: pd.DataFrame):
        """Print statistics about emotion distribution"""
        print("Emotion annotation statistics:")
        total_comments = len(df)
        
        for emotion in self.emotions_list:
            count = df[emotion].sum()
            percentage = (count / total_comments) * 100
            print(f"  {emotion:<15}: {count}/{total_comments} ({percentage:.1f}%)")
    
    def _validate_data(self, df: pd.DataFrame):
        """Validate the processed data"""
        if df.empty:
            raise ValueError("Dataset is empty after processing")
        
        if 'text' not in df.columns:
            raise ValueError("No text column found in dataset")
        
        if 'emotions' not in df.columns:
            raise ValueError("No emotions column created")
        
        # Check for missing text
        missing_text = df['text'].isna().sum()
        if missing_text > 0:
            self.logger.warning(f"{missing_text} comments have missing text")
        
        # Check emotion distribution
        emotion_counts = df[self.emotions_list].sum()
        zero_emotions = emotion_counts[emotion_counts == 0]
        if len(zero_emotions) > 0:
            self.logger.warning(f"Emotions with zero occurrences: {list(zero_emotions.index)}")
        
        self.logger.info(f"Data validation passed. {len(df)} comments ready for evaluation.")
    
    def get_few_shot_examples(self, n_examples: int = 5) -> List[Dict]:
        """Get few-shot examples for prompting"""
        df, _ = self.load_data()
        
        # Sample examples ensuring diverse emotion representation
        examples = []
        used_indices = set()
        
        # Try to get examples with different emotion patterns
        for _ in range(n_examples * 2):  # Try more to ensure diversity
            if len(examples) >= n_examples:
                break
                
            # Sample a random comment
            idx = np.random.choice(len(df))
            if idx in used_indices:
                continue
                
            used_indices.add(idx)
            row = df.iloc[idx]
            
            example = {
                'text': row['text'],
                'emotions': row['emotions']
            }
            examples.append(example)
        
        return examples[:n_examples]


# MISSING FUNCTIONS THAT EXISTING EVALUATION FILES EXPECT
# These functions maintain compatibility with existing evaluation files

def load_and_preprocess_goemotions_data(data_path: str, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and preprocess GoEmotions data (compatibility function)
    This function was expected by existing evaluation files
    """
    loader = GoEmotionsDataLoader(data_path)
    return loader.load_data(sample_size)

def get_comment_emotions(row, emotions_list: List[str]) -> List[str]:
    """
    Get emotions for a comment row (compatibility function)
    """
    if 'emotions' in row and isinstance(row['emotions'], list):
        return row['emotions']
    
    # Fallback: extract from individual emotion columns
    emotions = []
    for emotion in emotions_list:
        if emotion in row and row[emotion] == 1:
            emotions.append(emotion)
    return emotions

def filter_annotated_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter comments that have at least one emotion annotation (compatibility function)
    """
    # If 'emotions' column exists, filter by it
    if 'emotions' in df.columns:
        return df[df['emotions'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
    
    # Fallback: check individual emotion columns
    emotion_columns = [col for col in df.columns if col in config.GOEMOTIONS_EMOTIONS]
    if emotion_columns:
        has_emotion = df[emotion_columns].sum(axis=1) > 0
        return df[has_emotion]
    
    return df

def process_goemotions_dataset(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Process GoEmotions dataset (compatibility function)
    """
    return load_and_preprocess_goemotions_data(data_path)

def get_emotion_distribution(df: pd.DataFrame, emotions_list: List[str]) -> Dict:
    """Get emotion distribution statistics (compatibility function)"""
    distribution = {}
    total_comments = len(df)
    
    for emotion in emotions_list:
        if emotion in df.columns:
            count = df[emotion].sum()
        else:
            count = 0
        
        distribution[emotion] = {
            'count': int(count),
            'percentage': float(count / total_comments * 100) if total_comments > 0 else 0,
            'total_comments': total_comments
        }
    
    return distribution

def create_emotion_lists(df: pd.DataFrame, emotions_list: List[str]) -> pd.DataFrame:
    """
    Create emotion lists from binary columns (compatibility function)
    """
    df = df.copy()
    
    if 'emotions' not in df.columns:
        df['emotions'] = df[emotions_list].apply(
            lambda row: [emotion for emotion, value in row.items() if value == 1], 
            axis=1
        )
    
    return df

def validate_goemotions_data(df: pd.DataFrame) -> bool:
    """
    Validate GoEmotions data format (compatibility function)
    """
    required_columns = ['text']
    
    for col in required_columns:
        if col not in df.columns:
            return False
    
    # Check if we have emotion data
    emotion_columns = [col for col in df.columns if col in config.GOEMOTIONS_EMOTIONS]
    has_emotions_col = 'emotions' in df.columns
    
    return len(emotion_columns) > 0 or has_emotions_col

# For backward compatibility, also support the class being imported directly
def GoEmotionsDataLoader_legacy(data_path: str):
    """Legacy support for different import patterns"""
    return GoEmotionsDataLoader(data_path)


# Add these functions to your existing utils/data_loader.py file

def load_and_preprocess_goemotions_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess GoEmotions data (compatibility function for evaluation files)
    
    Args:
        data_path: Path to the GoEmotions CSV file
        
    Returns:
        Processed DataFrame with comments and emotion annotations
    """
    # Load the CSV file
    df = pd.read_csv(data_path)
    
    # Check if we have the expected format
    # GoEmotions format typically has: text, then emotion columns
    expected_emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # Find text column
    text_column = None
    possible_text_cols = ['text', 'comment_text', 'comment']
    for col in possible_text_cols:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        raise ValueError(f"No text column found. Available columns: {list(df.columns)}")
    
    # Rename to standard format
    if text_column != 'text':
        df = df.rename(columns={text_column: 'text'})
    
    # Ensure we have comment_id
    if 'comment_id' not in df.columns:
        if 'id' in df.columns:
            df = df.rename(columns={'id': 'comment_id'})
        else:
            df['comment_id'] = df.index.astype(str)
    
    # Check for emotion columns
    missing_emotions = []
    for emotion in expected_emotions:
        if emotion not in df.columns:
            missing_emotions.append(emotion)
    
    if missing_emotions:
        print(f"Warning: Missing emotion columns: {missing_emotions}")
        # Add missing columns as zeros
        for emotion in missing_emotions:
            df[emotion] = 0
    
    # Convert emotion columns to binary
    for emotion in expected_emotions:
        if emotion in df.columns:
            df[emotion] = pd.to_numeric(df[emotion], errors='coerce').fillna(0)
            df[emotion] = (df[emotion] > 0.5).astype(int)
    
    print(f"Loaded {len(df)} comments with emotion annotations")
    return df


def get_comment_emotions(row, annotator='expert') -> List[str]:
    """
    Extract emotions from a comment row (compatibility function)
    
    Args:
        row: DataFrame row containing emotion annotations
        annotator: Type of annotator ('expert' or other)
        
    Returns:
        List of emotions present in the comment
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    present_emotions = []
    for emotion in emotions:
        if emotion in row and row[emotion] == 1:
            present_emotions.append(emotion)
    
    return present_emotions


def filter_annotated_comments(df: pd.DataFrame, min_emotions: int = 1, annotator: str = 'expert') -> pd.DataFrame:
    """
    Filter to comments that have at least min_emotions annotations (compatibility function)
    
    Args:
        df: DataFrame with emotion annotations
        min_emotions: Minimum number of emotions required
        annotator: Type of annotator (ignored for compatibility)
        
    Returns:
        Filtered DataFrame
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # Count emotions per comment
    emotion_counts = df[emotions].sum(axis=1)
    
    # Filter to comments with at least min_emotions
    filtered_df = df[emotion_counts >= min_emotions]
    
    print(f"Filtered from {len(df)} to {len(filtered_df)} comments with at least {min_emotions} emotion(s)")
    return filtered_df