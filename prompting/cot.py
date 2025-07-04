# goemotions_evaluation/prompting/cot.py

"""
Chain of Thought prompting for GoEmotions emotion classification.
Uses step-by-step reasoning before making classification decisions.
"""

import time
from typing import List, Optional
from utils.emotion_rubric import GoEmotionsRubric

def get_cot_prediction(text: str,
                      subreddit: str,
                      author: str, 
                      client,
                      emotion: str) -> Optional[bool]:
    """
    Get Chain of Thought prediction for a single emotion.
    
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
    
    # Create Chain of Thought prompt with reasoning steps
    prompt = f"""Consider a Reddit comment below:

Subreddit: {subreddit}
Author: {author}
Comment: {text}

Please analyze this comment step by step to determine if it expresses this emotion:

Emotion: {emotion}
Definition: {prompt_descriptions[emotion]}

Step 1: What is the main content/sentiment of this comment?
Step 2: Does this comment express the emotion '{emotion}' based on the definition?
Step 3: What specific words or phrases support your decision?

Based on your step-by-step analysis, respond with "true" if the comment expresses the {emotion} emotion, or "false" if it does not."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Think step by step and end your response with either 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract the final true/false from the reasoning
        if "true" in result.split()[-5:]:  # Check last few words
            return True
        elif "false" in result.split()[-5:]:
            return False
        else:
            print(f"Could not extract true/false from CoT response for {emotion}: {result}")
            return None
            
    except Exception as e:
        print(f"Error getting CoT prediction for {emotion}: {str(e)}")
        return None


def get_cot_prediction_all_emotions(text: str,
                                  subreddit: str,
                                  author: str,
                                  client) -> List[str]:
    """
    Get Chain of Thought predictions for all GoEmotions categories.
    
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
        prediction = get_cot_prediction(
            text, subreddit, author, client, emotion
        )
        
        if prediction is True:
            assigned_emotions.append(emotion)
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_emotions