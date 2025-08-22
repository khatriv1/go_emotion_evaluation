# goemotions_evaluation/prompting/auto_cot.py
"""
Auto-CoT prompting for GoEmotions emotion classification.
FIXED: Just Zero-Shot + "Let's think step by step"
"""

import time
import re
import ast
import logging
from typing import List
import config
import openai

logger = logging.getLogger(__name__)

def parse_emotion_response(response_text: str, valid_emotions: List[str]) -> List[str]:
    """Parse emotion response from LLM output"""
    response_text = response_text.strip()
    
    try:
        list_match = re.search(r'\[([^\]]+)\]', response_text)
        if list_match:
            list_str = '[' + list_match.group(1) + ']'
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                emotions = [str(item).strip().strip("'\"") for item in parsed]
                return [e for e in emotions if e in valid_emotions]
    except:
        pass
    
    found_emotions = []
    response_lower = response_text.lower()
    
    for emotion in valid_emotions:
        if emotion.lower() in response_lower:
            found_emotions.append(emotion)
    
    return found_emotions if found_emotions else ['neutral']

def get_auto_cot_prediction_all_emotions(text: str, subreddit: str, author: str, client) -> List[str]:
    """
    Auto-CoT: Just Zero-Shot + "Let's think step by step"
    EXACTLY like Zero-Shot but with one added line
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    prompt = f"""Classify this Reddit comment into the most appropriate emotions.

Available emotions: {', '.join(emotions)}

Comment: "{text}"
Subreddit: {subreddit}
Author: {author}

Let's think step by step.

Instructions:
- Select only PRIMARY emotions clearly expressed
- Most comments have 1-2 emotions maximum
- Be selective - don't over-predict
- If no clear emotion, select 'neutral'

Response as Python list: ['emotion1', 'emotion2']

Response:"""

    try:
        response = client.chat.completions.create(
            model=config.MODEL_ID,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Think step by step and be selective."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=400
        )
        
        result = response.choices[0].message.content.strip()
        predicted_emotions = parse_emotion_response(result, emotions)
        
        if not predicted_emotions:
            predicted_emotions = ['neutral']
            
        return predicted_emotions
        
    except Exception as e:
        logger.error(f"Error in auto-CoT prediction: {e}")
        return ['neutral']