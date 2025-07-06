"""
GoEmotions Evaluation Configuration Example
Rename this file to config.py and update with your actual values
"""

import os
from pathlib import Path

# API Configuration - NEVER commit real API keys to GitHub
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")  # Set as environment variable
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MAX_TOKENS = 1000
OPENAI_TEMPERATURE = 0.0  # Deterministic for evaluation

# Data Configuration
DATA_PATH = "data/goemotions_1.csv"
RESULTS_BASE_DIR = "results"

# GoEmotions Dataset Configuration - Complete list of 28 emotions
GOEMOTIONS_EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Emotion categories for analysis
EMOTION_GROUPS = {
    'positive': ['admiration', 'amusement', 'approval', 'caring', 'excitement', 
                 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'],
    'negative': ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 
                 'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'],
    'cognitive': ['confusion', 'curiosity', 'realization', 'surprise'],
    'ambiguous': ['desire', 'neutral']
}

# Emotion definitions for prompting
EMOTION_DEFINITIONS = {
    'admiration': 'Finding something impressive or worthy of respect',
    'amusement': 'Finding something funny or entertaining',
    'anger': 'Strong feeling of displeasure or antagonism',
    'annoyance': 'Mild anger, irritation',
    'approval': 'Having or expressing a favorable opinion',
    'caring': 'Displaying kindness and concern for others',
    'confusion': 'Lack of understanding, uncertainty',
    'curiosity': 'Strong desire to know or learn something',
    'desire': 'Strong feeling of wanting something',
    'disappointment': 'Sadness caused by non-fulfillment of hopes',
    'disapproval': 'Having or expressing an unfavorable opinion',
    'disgust': 'Revulsion or strong disapproval',
    'embarrassment': 'Self-consciousness, shame, or awkwardness',
    'excitement': 'Feeling of great enthusiasm and eagerness',
    'fear': 'Being afraid or worried',
    'gratitude': 'Feeling of thankfulness and appreciation',
    'grief': 'Intense sorrow, especially from loss',
    'joy': 'Feeling of pleasure and happiness',
    'love': 'Strong positive emotion of regard and affection',
    'nervousness': 'Apprehension, worry, anxiety',
    'optimism': 'Hopefulness and confidence about the future',
    'pride': 'Pleasure or satisfaction due to achievements',
    'realization': 'Becoming aware of something',
    'relief': 'Reassurance following release from anxiety',
    'remorse': 'Regret or guilty feeling',
    'sadness': 'Emotional pain, sorrow',
    'surprise': 'Being astonished by something unexpected',
    'neutral': 'No particular emotion expressed'
}

# Evaluation Configuration
MIN_COMMENTS_FOR_EVALUATION = 10
DEFAULT_SAMPLE_SIZE = 100
RANDOM_SEED = 42

# Processing Configuration - Added missing values for your prompting files
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
REQUEST_DELAY = 0.5  # seconds between API calls

# Few-shot Configuration
FEW_SHOT_EXAMPLES = 5
ACTIVE_PROMPTING_UNCERTAINTY_THRESHOLD = 0.3

# Self-consistency Configuration
SELF_CONSISTENCY_SAMPLES = 3

# Metrics Configuration
BINARY_THRESHOLD = 0.5  # For converting probabilities to binary predictions
CONFIDENCE_THRESHOLD = 0.7  # For high-confidence predictions

# Output Configuration
SAVE_DETAILED_RESULTS = True
SAVE_PREDICTIONS = True
SAVE_METRICS = True
CREATE_VISUALIZATIONS = True

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# File paths
def get_data_path():
    """Get the full path to the data file"""
    return Path(DATA_PATH)

def get_results_dir():
    """Get the results directory path"""
    return Path(RESULTS_BASE_DIR)

def validate_config():
    """Validate configuration settings"""
    errors = []
    # Check data file exists
    if not get_data_path().exists():
        errors.append(f"Data file not found: {DATA_PATH}")
    
    # Check emotions list
    if len(GOEMOTIONS_EMOTIONS) != 28:
        errors.append(f"Expected 28 emotions, got {len(GOEMOTIONS_EMOTIONS)}")
    
    # Check emotion groups
    all_grouped_emotions = set()
    for group_emotions in EMOTION_GROUPS.values():
        all_grouped_emotions.update(group_emotions)
    
    if all_grouped_emotions != set(GOEMOTIONS_EMOTIONS):
        missing = set(GOEMOTIONS_EMOTIONS) - all_grouped_emotions
        extra = all_grouped_emotions - set(GOEMOTIONS_EMOTIONS)
        if missing:
            errors.append(f"Emotions not in groups: {missing}")
        if extra:
            errors.append(f"Extra emotions in groups: {extra}")
    
    return errors

# Model configurations for different techniques
MODEL_CONFIGS = {
    'zero_shot': {
        'temperature': 0.0,
        'max_tokens': 500,
        'top_p': 1.0
    },
    'few_shot': {
        'temperature': 0.0,
        'max_tokens': 600,
        'top_p': 1.0
    },
    'cot': {
        'temperature': 0.0,
        'max_tokens': 800,
        'top_p': 1.0
    },
    'self_consistency': {
        'temperature': 0.7,
        'max_tokens': 600,
        'top_p': 0.9
    }
}

# Validate configuration on import
config_errors = validate_config()
if config_errors:
    import warnings
    for error in config_errors:
        warnings.warn(f"Configuration warning: {error}")