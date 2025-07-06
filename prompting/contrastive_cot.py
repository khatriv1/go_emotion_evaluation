import time
import re
import ast
from typing import List, Optional
from utils.emotion_rubric import GoEmotionsRubric

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

def get_contrastive_cot_prediction_all_emotions(
    text: str,
    subreddit: str,
    author: str,
    client
) -> List[str]:
    """
    Perform contrastive Chain-of-Thought analysis using multi-label approach.
    FIXED: Single comprehensive prompt with contrastive reasoning
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # Get contrastive examples from rubric
    rubric = GoEmotionsRubric()
    
    prompt = f"""Classify this Reddit comment using contrastive reasoning.

Available emotions: {', '.join(emotions)}

Comment: "{text}"
Subreddit: {subreddit}
Author: {author}

Think contrastively:
1. Positive reasoning: What emotions does this comment clearly express?
2. Negative reasoning: What emotions might it seem like but actually isn't?
3. Contrast: Why are some emotions present while others are not?
4. Final decision: What are the most accurate emotions?

Instructions:
- Select only PRIMARY emotions clearly expressed
- Most comments have 1-2 emotions maximum
- Use contrastive thinking to avoid over-prediction
- If no clear emotion, select 'neutral'

Response as Python list: ['emotion1', 'emotion2']

Final Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at emotion classification using contrastive reasoning. Think about what emotions are present vs. absent and be selective."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=250
        )
        
        result = response.choices[0].message.content.strip()
        predicted_emotions = parse_emotion_response(result, emotions)
        
        # Fallback to neutral if no emotions found
        if not predicted_emotions:
            predicted_emotions = ['neutral']
            
        return predicted_emotions

    except Exception as e:
        print(f"Error in Contrastive CoT: {e}")
        return ['neutral']