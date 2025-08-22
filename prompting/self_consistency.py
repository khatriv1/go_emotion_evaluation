# goemotions_evaluation/prompting/self_consistency.py
"""
Self-Consistency prompting for GoEmotions.
FIXED: Includes Few-shot examples + Auto-CoT like Bloom's
"""

import time
import re
import ast
import logging
from typing import List, Dict
from collections import Counter
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

def get_single_reasoning_path(text: str, subreddit: str, author: str, 
                            client, temperature: float = 0.7) -> List[str]:
    """
    Get a single reasoning path with Few-shot + Auto-CoT
    NOW WITH: Few-shot examples + Auto-CoT
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # FEW-SHOT EXAMPLES (same as few_shot.py)
    examples_text = """EXAMPLES:
Comment: "I'm so happy about this news!"
Response: ['joy']

Comment: "This is really frustrating and annoying"  
Response: ['anger', 'annoyance']

Comment: "The meeting is scheduled for 3pm"
Response: ['neutral']

Comment: "I can't believe how amazing this performance was!"
Response: ['admiration', 'excitement']

Comment: "I'm worried about what might happen"
Response: ['fear', 'nervousness']

Comment: "Thank you so much for helping me"
Response: ['gratitude']
"""

    prompt = f"""{examples_text}

Now classify this Reddit comment:
Comment: "{text}"
Subreddit: {subreddit}
Author: {author}

Let's think step by step.

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
            model=config.MODEL_ID,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Learn from the examples, think step by step, and be selective."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=400
        )
        
        result = response.choices[0].message.content.strip()
        predicted_emotions = parse_emotion_response(result, emotions)
        
        if not predicted_emotions:
            predicted_emotions = ['neutral']
            
        return predicted_emotions
        
    except Exception as e:
        logger.error(f"Error in reasoning path: {e}")
        return ['neutral']

def get_self_consistency_prediction_all_emotions(text: str, subreddit: str, author: str,
                                               client, n_samples: int = 5) -> List[str]:
    """
    Get Self-Consistency predictions using multiple reasoning paths.
    NOW WITH: Few-shot examples + Auto-CoT in each attempt
    """
    
    all_predictions = []
    
    for i in range(n_samples):
        # Fixed temperature for all samples
        temp = 0.7
        
        prediction = get_single_reasoning_path(text, subreddit, author, client, temp)
        
        if prediction:
            all_predictions.append(tuple(sorted(prediction)))
        
        time.sleep(0.2)
    
    if not all_predictions:
        logger.warning("No valid predictions obtained from any reasoning path")
        return ['neutral']
    
    # Aggregate predictions using majority voting
    counter = Counter(all_predictions)
    most_common_emotions = counter.most_common(1)[0][0]
    
    logger.info(f"Self-consistency: Used {len(all_predictions)} predictions")
    
    return list(most_common_emotions)