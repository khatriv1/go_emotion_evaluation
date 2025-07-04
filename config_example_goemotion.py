# GoEmotions Evaluation Configuration Example
# Rename to config.py and update with your values

# OpenAI API Configuration
OPENAI_API_KEY = "your-api-key-here"  # Replace with your OpenAI API key

# Model Configuration
MODEL_ID = "gpt-3.5-turbo-0125"  # Base model ID

# Data Configuration - Using GoEmotions dataset
DATA_PATH = "data/goemotions_1.csv"  # Path to GoEmotions dataset file

# Output Configuration
OUTPUT_DIR = "results"  # Directory for saving results

# Rate Limiting
SLEEP_TIME = 1  # Seconds to wait between API calls

# GoEmotions Categories (28 emotions from the dataset)
GOEMOTIONS_CATEGORIES = [
    "admiration",
    "amusement", 
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral"
]