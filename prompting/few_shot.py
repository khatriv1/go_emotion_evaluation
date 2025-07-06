# goemotions_evaluation/prompting/few_shot.py

"""
Few-shot prompting for GoEmotions emotion classification.
FIXED: Multi-label approach with focused examples
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

def get_few_shot_examples() -> str:
    """Get few-shot examples for emotion classification"""
    examples = """Here are some examples:

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

Comment: "This is so confusing, I don't understand"
Response: ['confusion']"""
    
    return examples

def get_few_shot_prediction_all_emotions(text: str,
                                       subreddit: str,
                                       author: str,
                                       client) -> List[str]:
    """
    Get few-shot predictions using multi-label approach.
    FIXED: Single comprehensive prompt with examples
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    examples = get_few_shot_examples()
    
    # Create comprehensive prompt with examples
    prompt = f"""{examples}

Now classify this Reddit comment:

Available emotions: {', '.join(emotions)}

Comment: "{text}"
Subreddit: {subreddit}
Author: {author}

Instructions:
- Select only PRIMARY emotions clearly expressed
- Most comments have 1-2 emotions maximum
- Be selective - don't over-predict
- Follow the pattern from examples above
- If no clear emotion, select 'neutral'

Response:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Follow the examples provided and be selective. Only choose emotions that are clearly expressed."},
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
        print(f"Error getting few-shot prediction: {str(e)}")
        return ['neutral']