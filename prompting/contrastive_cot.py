# goemotions_evaluation/prompting/contrastive_cot.py

"""
Contrastive Chain of Thought prompting for GoEmotions emotion classification.
Uses both positive and negative reasoning to improve classification.
"""

import time
from typing import List, Optional, Dict
from utils.emotion_rubric import GoEmotionsRubric

def get_contrastive_examples(emotion: str) -> str:
    """Generate contrastive reasoning examples for each emotion."""
    contrastive_examples = {
        'admiration': {
            "positive": {
                "comment": "That was absolutely incredible work, I'm so impressed!",
                "positive_reasoning": "This expresses strong positive regard ('incredible work') and explicit impression ('so impressed'), showing respect and finding something worthy of admiration.",
                "negative_reasoning": "This is NOT just general happiness (joy), NOT amusement (no humor), NOT gratitude (no thanks) - it's specifically about being impressed by achievement.",
                "answer": "true"
            },
            "negative": {
                "comment": "Thank you so much for helping me!",
                "positive_reasoning": "This shows appreciation and positive sentiment.",
                "negative_reasoning": "However, this is expressing gratitude and thankfulness, NOT admiration for impressive work or respect for achievement.",
                "answer": "false"
            }
        },
        'joy': {
            "positive": {
                "comment": "I'm so happy and delighted about this news!",
                "positive_reasoning": "This explicitly states happiness ('so happy') and delight, which are direct indicators of joy and positive feelings.",
                "negative_reasoning": "This is NOT just approval (no agreement), NOT gratitude (no thanks), NOT excitement about future events - it's present happiness.",
                "answer": "true"
            },
            "negative": {
                "comment": "I can't wait for this to happen tomorrow!",
                "positive_reasoning": "This shows positive anticipation and eagerness.",
                "negative_reasoning": "However, this is excitement about future events, NOT current joy or happiness about something that already brings pleasure.",
                "answer": "false"
            }
        },
        'anger': {
            "positive": {
                "comment": "This makes me absolutely furious and livid!",
                "positive_reasoning": "This explicitly expresses intense anger ('furious', 'livid') with strong emphasis ('absolutely'), showing displeasure and antagonism.",
                "negative_reasoning": "This is NOT mild annoyance, NOT disappointment, NOT disapproval - it's intense anger and fury.",
                "answer": "true"
            },
            "negative": {
                "comment": "This is getting on my nerves a bit",
                "positive_reasoning": "This shows irritation and negative feeling.",
                "negative_reasoning": "However, this is mild annoyance and irritation, NOT the strong displeasure and antagonism that characterizes anger.",
                "answer": "false"
            }
        }
    }
    
    # Get examples for the emotion
    if emotion not in contrastive_examples:
        # Provide default structure
        return "No contrastive examples available for this emotion."
    
    examples = contrastive_examples[emotion]
    
    # Format contrastive examples
    formatted = []
    for example_type, example in examples.items():
        formatted.append(f"Example ({example_type}):\nComment: {example['comment']}\nPositive reasoning: {example['positive_reasoning']}\nNegative reasoning: {example['negative_reasoning']}\nAnswer: {example['answer']}")
    
    return "\n\n".join(formatted)

def get_contrastive_cot_prediction(text: str,
                                 subreddit: str,
                                 author: str, 
                                 client,
                                 emotion: str) -> Optional[bool]:
    """
    Get Contrastive CoT prediction for a single emotion.
    
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
    
    # Get contrastive examples
    contrastive_examples = get_contrastive_examples(emotion)
    
    # Create Contrastive CoT prompt
    prompt = f"""You are classifying emotions in Reddit comments.

Emotion: {emotion}
Definition: {prompt_descriptions[emotion]}

Here are contrastive examples showing both why comments do and don't belong to this emotion:

{contrastive_examples}

Now classify this comment using both positive and negative reasoning:

Subreddit: {subreddit}
Author: {author}
Comment: {text}

First, provide positive reasoning (why it MIGHT express {emotion}).
Then, provide negative reasoning (why it might NOT express {emotion}).
Finally, based on both reasonings, answer with 'true' or 'false'."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Use contrastive reasoning to make accurate classifications. Always end with 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=250
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract final answer
        if "true" in result.split()[-5:]:
            return True
        elif "false" in result.split()[-5:]:
            return False
        else:
            print(f"Could not extract answer from Contrastive CoT response for {emotion}: {result}")
            return None
            
    except Exception as e:
        print(f"Error getting Contrastive CoT prediction for {emotion}: {str(e)}")
        return None


def get_contrastive_cot_prediction_all_emotions(text: str,
                                              subreddit: str,
                                              author: str,
                                              client) -> List[str]:
    """
    Get Contrastive CoT predictions for all GoEmotions categories.
    
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
        prediction = get_contrastive_cot_prediction(
            text, subreddit, author, client, emotion
        )
        
        if prediction is True:
            assigned_emotions.append(emotion)
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_emotions