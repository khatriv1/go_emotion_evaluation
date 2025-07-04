# goemotions_evaluation/prompting/rephrase_and_respond.py

"""
Rephrase and Respond prompting for GoEmotions emotion classification.
First rephrases the comment to better understand it, then classifies.
"""

import time
from typing import List, Optional, Tuple
from utils.emotion_rubric import GoEmotionsRubric

def rephrase_comment(comment: str, client) -> Optional[str]:
    """
    Rephrase the comment to clarify its meaning.
    
    Args:
        comment: The original Reddit comment
        client: OpenAI client
    
    Returns:
        Rephrased comment or None if failed
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

def get_rephrase_respond_prediction(text: str,
                                  subreddit: str,
                                  author: str, 
                                  client,
                                  emotion: str) -> Optional[bool]:
    """
    Get Rephrase and Respond prediction for a single emotion.
    
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
    
    # Step 1: Rephrase the comment
    rephrased = rephrase_comment(text, client)
    if rephrased is None:
        print(f"Failed to rephrase comment for {emotion}, using original")
        rephrased = text
    
    # Step 2: Create classification prompt with both original and rephrased
    prompt = f"""You are classifying emotions in Reddit comments.

Emotion: {emotion}
Definition: {prompt_descriptions[emotion]}

Comment Context:
- Subreddit: {subreddit}
- Author: {author}

Original comment: "{text}"
Rephrased for clarity: "{rephrased}"

Based on both the original comment and its clarified meaning, does this comment express the '{emotion}' emotion?

Answer with 'true' if it expresses this emotion, or 'false' if it does not."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Consider both the original and rephrased versions to make accurate classifications. Respond only with 'true' or 'false'."},
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
        print(f"Error getting rephrase-respond prediction for {emotion}: {str(e)}")
        return None


def get_rephrase_respond_prediction_all_emotions(text: str,
                                               subreddit: str,
                                               author: str,
                                               client) -> List[str]:
    """
    Get Rephrase and Respond predictions for all GoEmotions categories.
    
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
    
    # First, get the rephrased version once (to avoid rephrasing multiple times)
    rephrased = rephrase_comment(text, client)
    if rephrased is None:
        rephrased = text
    
    for emotion in emotions:
        # For efficiency, we'll use the already rephrased version
        rubric = GoEmotionsRubric()
        prompt_descriptions = rubric.get_prompt_descriptions()
        
        prompt = f"""You are classifying emotions in Reddit comments.

Emotion: {emotion}
Definition: {prompt_descriptions[emotion]}

Comment Context:
- Subreddit: {subreddit}
- Author: {author}

Original comment: "{text}"
Rephrased for clarity: "{rephrased}"

Based on both the original comment and its clarified meaning, does this comment express the '{emotion}' emotion?

Answer with 'true' if it expresses this emotion, or 'false' if it does not."""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing emotions in text. Consider both the original and rephrased versions to make accurate classifications. Respond only with 'true' or 'false'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            if result == "true":
                assigned_emotions.append(emotion)
                
        except Exception as e:
            print(f"Error getting rephrase-respond prediction for {emotion}: {str(e)}")
            continue
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_emotions