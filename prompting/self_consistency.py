# goemotions_evaluation/prompting/self_consistency.py

"""
Self-Consistency prompting for GoEmotions emotion classification.
FIXED: Multi-label approach with multiple reasoning paths
"""

import time
import re
import ast
from typing import List, Optional, Dict
from collections import Counter

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

def get_single_reasoning_path(text: str,
                            subreddit: str,
                            author: str,
                            client,
                            temperature: float = 0.7) -> List[str]:
    """
    Get a single reasoning path for classification.
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

Think through this step-by-step:
1. What is the main emotional tone of this comment?
2. What specific words indicate emotions?
3. What emotions are clearly expressed?

Instructions:
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
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Provide your reasoning and be selective with emotion selection."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        predicted_emotions = parse_emotion_response(result, emotions)
        
        # Fallback to neutral if no emotions found
        if not predicted_emotions:
            predicted_emotions = ['neutral']
            
        return predicted_emotions
            
    except Exception as e:
        print(f"Error in reasoning path: {str(e)}")
        return ['neutral']

def get_self_consistency_prediction_all_emotions(text: str,
                                               subreddit: str,
                                               author: str,
                                               client,
                                               n_samples: int = 5) -> List[str]:
    """
    Get Self-Consistency predictions using multiple reasoning paths.
    FIXED: Multi-label approach with majority voting
    """
    
    # Collect predictions from multiple reasoning paths
    all_predictions = []
    
    for i in range(n_samples):
        # Vary temperature for diversity
        temp = 0.5 + (i * 0.1)  # 0.5, 0.6, 0.7, 0.8, 0.9
        
        prediction = get_single_reasoning_path(
            text, subreddit, author, client, temp
        )
        
        if prediction:
            all_predictions.append(prediction)
        
        # Small delay between samples
        time.sleep(0.2)
    
    if not all_predictions:
        print(f"No valid predictions obtained")
        return ['neutral']
    
    # Aggregate predictions using majority voting for each emotion
    emotion_votes = Counter()
    for pred_list in all_predictions:
        for emotion in pred_list:
            emotion_votes[emotion] += 1
    
    # Select emotions that appear in majority of predictions (> 50%)
    threshold = len(all_predictions) * 0.4  # Lower threshold for multi-label
    final_emotions = [emotion for emotion, count in emotion_votes.items() 
                     if count >= threshold]
    
    # If no emotions meet threshold, take the most common ones
    if not final_emotions:
        # Take top 2 most common emotions
        most_common = emotion_votes.most_common(2)
        final_emotions = [emotion for emotion, count in most_common if count > 0]
    
    # Fallback to neutral
    if not final_emotions:
        final_emotions = ['neutral']
    
    print(f"Self-consistency votes: {dict(emotion_votes)}")
    print(f"Final emotions: {final_emotions}")
    
    return final_emotions