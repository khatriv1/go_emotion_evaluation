# goemotions_evaluation/prompting/take_a_step_back.py

"""
Take a Step Back prompting for GoEmotions emotion classification.
First derives high-level principles, then applies them to classification.
"""

import time
from typing import List, Optional, Dict
from utils.emotion_rubric import GoEmotionsRubric

def derive_classification_principles(emotion: str, client) -> Optional[str]:
    """
    Derive high-level principles for classifying an emotion.
    
    Args:
        emotion: The emotion to derive principles for
        client: OpenAI client
    
    Returns:
        String containing derived principles or None if failed
    """
    rubric = GoEmotionsRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    prompt = f"""Take a step back and think about the fundamental principles for identifying the '{emotion}' emotion in text.

Emotion definition: {prompt_descriptions[emotion]}

What are the key characteristics, patterns, and principles that would help identify if a Reddit comment expresses this emotion? 

List 3-5 high-level principles:"""

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
        print(f"Error deriving principles for {emotion}: {str(e)}")
        return None

def get_take_step_back_prediction(text: str,
                                subreddit: str,
                                author: str, 
                                client,
                                emotion: str) -> Optional[bool]:
    """
    Get Take a Step Back prediction for a single emotion.
    
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
    
    # Step 1: Derive high-level principles
    principles = derive_classification_principles(emotion, client)
    if principles is None:
        print(f"Failed to derive principles for {emotion}")
        principles = "Consider the emotion definition carefully."
    
    # Step 2: Apply principles to classify
    prompt = f"""You are classifying emotions in Reddit comments.

Emotion: {emotion}
Definition: {prompt_descriptions[emotion]}

High-level principles for this emotion:
{principles}

Now apply these principles to classify this specific comment:

Subreddit: {subreddit}
Author: {author}
Comment: {text}

Based on the principles above, does this comment express the '{emotion}' emotion?
Answer 'true' or 'false'."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Apply the given principles to make accurate classifications. Respond only with 'true' or 'false'."},
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
        print(f"Error getting take-step-back prediction for {emotion}: {str(e)}")
        return None


def get_take_step_back_prediction_all_emotions(text: str,
                                             subreddit: str,
                                             author: str,
                                             client) -> List[str]:
    """
    Get Take a Step Back predictions for all GoEmotions categories.
    
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
    
    # Pre-derive principles for all emotions to be efficient
    principles_cache = {}
    print("Deriving classification principles for all emotions...")
    for emotion in emotions:
        principles = derive_classification_principles(emotion, client)
        if principles:
            principles_cache[emotion] = principles
        else:
            # Fallback principles if derivation fails
            principles_cache[emotion] = f"Consider if the comment fits the definition of {emotion}."
        time.sleep(0.3)  # Rate limiting for principle derivation
    
    # Now classify using cached principles
    for emotion in emotions:
        rubric = GoEmotionsRubric()
        prompt_descriptions = rubric.get_prompt_descriptions()
        
        prompt = f"""You are classifying emotions in Reddit comments.

Emotion: {emotion}
Definition: {prompt_descriptions[emotion]}

High-level principles for this emotion:
{principles_cache[emotion]}

Now apply these principles to classify this specific comment:

Subreddit: {subreddit}
Author: {author}
Comment: {text}

Based on the principles above, does this comment express the '{emotion}' emotion?
Answer 'true' or 'false'."""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing emotions in text. Apply the given principles to make accurate classifications. Respond only with 'true' or 'false'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            if result == "true":
                assigned_emotions.append(emotion)
                
        except Exception as e:
            print(f"Error getting take-step-back prediction for {emotion}: {str(e)}")
            continue
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_emotions