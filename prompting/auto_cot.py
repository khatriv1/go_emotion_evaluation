# goemotions_evaluation/prompting/auto_cot.py

"""
Auto-CoT (Automatic Chain of Thought) prompting for GoEmotions emotion classification.
FIXED: Multi-label approach with automatic reasoning chains
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

def generate_auto_cot_examples() -> str:
    """Generate automatic reasoning chain examples"""
    return """Here are examples with step-by-step reasoning:

Comment: "That was absolutely incredible work!"
Let's think step by step: The comment uses "absolutely incredible" which shows high praise and positive regard for someone's work. This indicates admiration for the quality of work.
Answer: ['admiration']

Comment: "I hate this so much, it's completely unfair"
Let's think step by step: The comment expresses "hate" and "unfair" which are strong negative emotions. "Hate" indicates anger, and "unfair" suggests disapproval of the situation.
Answer: ['anger', 'disapproval']

Comment: "The report is due tomorrow at 5pm"
Let's think step by step: This is a factual statement providing deadline information. There are no emotional words or expressions of feeling. This is purely informational.
Answer: ['neutral']"""

def get_auto_cot_prediction_all_emotions(text: str,
                                       subreddit: str,
                                       author: str,
                                       client) -> List[str]:
    """
    Get Auto-CoT predictions using multi-label approach.
    FIXED: Single comprehensive prompt with auto-generated reasoning
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    examples = generate_auto_cot_examples()
    
    # Create Auto-CoT prompt
    prompt = f"""{examples}

Now classify this comment using the same step-by-step reasoning:

Available emotions: {', '.join(emotions)}

Comment: "{text}"
Subreddit: {subreddit}
Author: {author}

Let's think step by step: [Analyze the emotional content, key words, and tone]

Instructions:
- Select only PRIMARY emotions clearly expressed
- Most comments have 1-2 emotions maximum
- Be selective - don't over-predict
- If no clear emotion, select 'neutral'

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Think step by step like the examples and be selective. End with 'Answer: [emotions_list]'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip()
        predicted_emotions = parse_emotion_response(result, emotions)
        
        # Fallback to neutral if no emotions found
        if not predicted_emotions:
            predicted_emotions = ['neutral']
            
        return predicted_emotions
        
    except Exception as e:
        print(f"Error getting Auto-CoT prediction: {str(e)}")
        return ['neutral']