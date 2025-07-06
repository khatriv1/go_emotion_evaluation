# goemotions_evaluation/prompting/rephrase_and_respond.py

"""
Rephrase and Respond prompting for GoEmotions emotion classification.
FIXED: Multi-label approach with rephrasing step
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

def rephrase_comment(comment: str, client) -> Optional[str]:
    """
    Rephrase the comment to clarify its meaning.
    """
    prompt = f"""Rephrase the following Reddit comment to make its emotional intent and meaning clearer, while preserving all important information:

Original comment: "{comment}"

Rephrased comment:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at understanding and clarifying the emotional meaning of comments. Rephrase to make emotional intent clear while preserving all information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error rephrasing comment: {str(e)}")
        return None

def get_rephrase_respond_prediction_all_emotions(text: str,
                                               subreddit: str,
                                               author: str,
                                               client) -> List[str]:
    """
    Get Rephrase and Respond predictions using multi-label approach.
    FIXED: Single comprehensive prompt after rephrasing
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # Step 1: Rephrase the comment
    rephrased = rephrase_comment(text, client)
    if rephrased is None:
        print(f"Failed to rephrase comment, using original")
        rephrased = text
    
    # Step 2: Classify using both original and rephrased versions
    prompt = f"""Classify this Reddit comment into the most appropriate emotions.

Available emotions: {', '.join(emotions)}

Original comment: "{text}"
Rephrased for clarity: "{rephrased}"
Subreddit: {subreddit}
Author: {author}

Instructions:
- Consider both the original and rephrased versions
- Select only PRIMARY emotions clearly expressed
- Most comments have 1-2 emotions maximum
- Be selective - don't over-predict
- If no clear emotion, select 'neutral'

Response as Python list: ['emotion1', 'emotion2']

Response:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Consider both the original and rephrased versions to make accurate classifications. Be selective."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        
        result = response.choices[0].message.content.strip()
        predicted_emotions = parse_emotion_response(result, emotions)
        
        # Fallback to neutral if no emotions found
        if not predicted_emotions:
            predicted_emotions = ['neutral']
            
        return predicted_emotions
        
    except Exception as e:
        print(f"Error getting rephrase-respond prediction: {str(e)}")
        return ['neutral']