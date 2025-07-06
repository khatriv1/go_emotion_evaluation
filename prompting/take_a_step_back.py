# goemotions_evaluation/prompting/take_a_step_back.py

"""
Take a Step Back prompting for GoEmotions emotion classification.
FIXED: Multi-label approach with high-level principles
"""

import time
import re
import ast
from typing import List, Optional

def parse_emotion_response(response_text: str, valid_emotions: List[str]) -> List[str]:
    """Parse emotion response from LLM output"""
    response_text = response_text.strip()
    
    # Try to parse as Python list first
    try:
        # Look for list pattern like ['emotion1', 'emotion2']
        list_match = re.search(r'\[([^\]]+)\]', response_text)
        if list_match:
            list_str = '[' + list_match.group(1) + ']'
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                emotions = [str(item).strip().strip("'\"") for item in parsed]
                # Filter to valid emotions only
                return [e for e in emotions if e in valid_emotions]
    except:
        pass
    
    # Try to find emotions mentioned in the text
    found_emotions = []
    response_lower = response_text.lower()
    
    for emotion in valid_emotions:
        if emotion.lower() in response_lower:
            found_emotions.append(emotion)
    
    return found_emotions

def derive_classification_principles(client) -> str:
    """
    Derive high-level principles for emotion classification.
    """
    prompt = f"""Take a step back and think about the fundamental principles for identifying emotions in Reddit comments.

What are the key characteristics, patterns, and principles that would help identify emotions in text? 

List 5-7 high-level principles for emotion classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotional patterns in text. Derive clear, high-level principles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error deriving principles: {str(e)}")
        return "Consider the explicit emotional words, context, and tone of the comment."

def get_take_step_back_prediction_all_emotions(text: str,
                                             subreddit: str,
                                             author: str,
                                             client) -> List[str]:
    """
    Get Take a Step Back predictions using multi-label approach.
    FIXED: Single comprehensive prompt with high-level principles
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # Step 1: Derive high-level principles (cached for efficiency)
    print("Deriving classification principles...")
    principles = derive_classification_principles(client)
    
    # Step 2: Apply principles to classify
    prompt = f"""Classify this Reddit comment using high-level emotion classification principles.

High-level principles for emotion classification:
{principles}

Available emotions: {', '.join(emotions)}

Comment: "{text}"
Subreddit: {subreddit}
Author: {author}

Apply the principles above to classify this comment:
1. What high-level patterns do you see?
2. What emotions are clearly expressed based on the principles?
3. Be selective - most comments have 1-2 emotions maximum

Instructions:
- Select only PRIMARY emotions clearly expressed
- Don't over-predict - follow the principles
- If no clear emotion, select 'neutral'

Response as Python list: ['emotion1', 'emotion2']

Response:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Apply the given principles to make accurate classifications. Be selective."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        predicted_emotions = parse_emotion_response(result, emotions)
        
        # Fallback to neutral if no emotions found
        if not predicted_emotions:
            predicted_emotions = ['neutral']
            
        return predicted_emotions
        
    except Exception as e:
        print(f"Error getting take-step-back prediction: {str(e)}")
        return ['neutral']