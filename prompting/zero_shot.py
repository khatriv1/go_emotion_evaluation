# goemotions_evaluation/prompting/zero_shot.py

"""
Zero-shot prompting for GoEmotions emotion classification.
Classifies Reddit comments into GoEmotions' 28 emotion categories without examples.
"""

import time
from typing import List, Optional
from utils.emotion_rubric import GoEmotionsRubric

def get_zero_shot_prediction(text: str, 
                           subreddit: str, 
                           author: str,
                           client,
                           emotion: str) -> Optional[bool]:
    """
    Get zero-shot prediction for a single emotion using GoEmotions rubric.
    
    Args:
        text: The Reddit comment text
        subreddit: Name of the subreddit  
        author: Comment author
        client: OpenAI client
        emotion: Emotion to classify for
    
    Returns:
        Boolean indicating if comment expresses this emotion, None if failed
    """
    rubric = GoEmotionsRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    if emotion not in prompt_descriptions:
        raise ValueError(f"Unknown emotion: {emotion}")
    
    # Create zero-shot prompt
    prompt = f"""Consider a Reddit comment below:

Subreddit: {subreddit}
Author: {author}
Comment: {text}

If the statement below is true, please respond "true"; otherwise, please respond "false":

{prompt_descriptions[emotion]}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Respond only with 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        if result == "true":
            return True
        elif result == "false":
            return False
        else:
            print(f"Unexpected response for {emotion}: {result}")
            return None
            
    except Exception as e:
        print(f"Error getting prediction for {emotion}: {str(e)}")
        return None


def get_zero_shot_prediction_all_emotions(text: str,
                                        subreddit: str, 
                                        author: str,
                                        client) -> List[str]:
    """
    Get zero-shot predictions for all GoEmotions categories.
    
    Args:
        text: The Reddit comment text
        subreddit: Name of the subreddit
        author: Comment author  
        client: OpenAI client
    
    Returns:
        List of emotions assigned to the comment
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    assigned_emotions = []
    
    for emotion in emotions:
        prediction = get_zero_shot_prediction(
            text, subreddit, author, client, emotion
        )
        
        if prediction is True:
            assigned_emotions.append(emotion)
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_emotions