# goemotions_evaluation/prompting/self_consistency.py

"""
Self-Consistency prompting for GoEmotions emotion classification.
Samples multiple reasoning paths and takes majority vote.
"""

import time
from typing import List, Optional, Dict
from collections import Counter
from utils.emotion_rubric import GoEmotionsRubric

def get_single_reasoning_path(text: str,
                            subreddit: str,
                            author: str,
                            client,
                            emotion: str,
                            temperature: float = 0.7) -> Optional[bool]:
    """
    Get a single reasoning path for classification.
    
    Args:
        text: The Reddit comment text
        subreddit: Name of the subreddit
        author: Comment author
        client: OpenAI client
        emotion: Emotion to classify for
        temperature: Sampling temperature for diversity
    
    Returns:
        Boolean prediction or None if failed
    """
    rubric = GoEmotionsRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    prompt = f"""Consider a Reddit comment:

Subreddit: {subreddit}
Author: {author}
Comment: {text}

Emotion: {emotion}
Definition: {prompt_descriptions[emotion]}

Think through this step-by-step and explain your reasoning:
1. What is the main emotional tone of this comment?
2. How does it relate to the {emotion} emotion definition?
3. What specific aspects make it fit or not fit?

Based on your analysis, does this comment express the '{emotion}' emotion? Answer 'true' or 'false'."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Provide your reasoning and end with 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract answer
        if "true" in result.split()[-5:]:
            return True
        elif "false" in result.split()[-5:]:
            return False
        else:
            return None
            
    except Exception as e:
        print(f"Error in reasoning path: {str(e)}")
        return None

def get_self_consistency_prediction(text: str,
                                  subreddit: str,
                                  author: str, 
                                  client,
                                  emotion: str,
                                  n_samples: int = 5) -> Optional[bool]:
    """
    Get Self-Consistency prediction using multiple reasoning paths.
    
    Args:
        text: The Reddit comment text
        subreddit: Name of the subreddit
        author: Comment author
        client: OpenAI client
        emotion: Emotion to classify for
        n_samples: Number of reasoning paths to sample
    
    Returns:
        Boolean indicating if comment expresses this emotion, None if failed
    """
    rubric = GoEmotionsRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    if emotion not in prompt_descriptions:
        raise ValueError(f"Unknown emotion: {emotion}")
    
    # Collect predictions from multiple reasoning paths
    predictions = []
    
    for i in range(n_samples):
        # Vary temperature for diversity
        temp = 0.5 + (i * 0.1)  # 0.5, 0.6, 0.7, 0.8, 0.9
        
        prediction = get_single_reasoning_path(
            text, subreddit, author, client, emotion, temp
        )
        
        if prediction is not None:
            predictions.append(prediction)
        
        # Small delay between samples
        time.sleep(0.2)
    
    if not predictions:
        print(f"No valid predictions obtained for {emotion}")
        return None
    
    # Take majority vote
    vote_counts = Counter(predictions)
    majority_vote = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[majority_vote] / len(predictions)
    
    print(f"Self-consistency for {emotion}: {vote_counts}, confidence: {confidence:.2f}")
    
    return majority_vote


def get_self_consistency_prediction_all_emotions(text: str,
                                               subreddit: str,
                                               author: str,
                                               client,
                                               n_samples: int = 5) -> List[str]:
    """
    Get Self-Consistency predictions for all GoEmotions categories.
    
    Args:
        text: The Reddit comment text
        subreddit: Name of the subreddit
        author: Comment author
        client: OpenAI client
        n_samples: Number of reasoning paths per emotion
    
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
        prediction = get_self_consistency_prediction(
            text, subreddit, author, client, emotion, n_samples
        )
        
        if prediction is True:
            assigned_emotions.append(emotion)
        
        # Rate limiting between emotions
        time.sleep(0.5)
    
    return assigned_emotions