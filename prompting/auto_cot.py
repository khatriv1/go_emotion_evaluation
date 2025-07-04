# goemotions_evaluation/prompting/auto_cot.py

"""
Auto-CoT (Automatic Chain of Thought) prompting for GoEmotions emotion classification.
Automatically generates reasoning chains for classification.
"""

import time
from typing import List, Optional, Dict
from utils.emotion_rubric import GoEmotionsRubric

def generate_auto_cot_examples(emotion: str) -> str:
    """Generate automatic reasoning chains for examples."""
    reasoning_examples = {
        'admiration': [
            {
                "comment": "That was absolutely incredible work!",
                "reasoning": "Let's think step by step. The comment uses the word 'incredible' which is strongly positive and expresses high regard for someone's work. This shows respect and finding something impressive. This matches the 'admiration' emotion.",
                "answer": "true"
            },
            {
                "comment": "I hate this so much",
                "reasoning": "Let's think step by step. This comment expresses strong negative feeling ('hate') which is the opposite of admiration. There's no respect or positive regard shown. This doesn't match the 'admiration' emotion.",
                "answer": "false"
            }
        ],
        'amusement': [
            {
                "comment": "Haha this is hilarious!",
                "reasoning": "Let's think step by step. The comment starts with 'Haha' which indicates laughter, and explicitly states 'hilarious' which means very funny. This clearly shows the commenter finds something entertaining and amusing.",
                "answer": "true"
            },
            {
                "comment": "This is very serious and important",
                "reasoning": "Let's think step by step. This comment emphasizes seriousness and importance, which is the opposite of amusement. There's no indication of humor or entertainment. This doesn't match the 'amusement' emotion.",
                "answer": "false"
            }
        ],
        'anger': [
            {
                "comment": "This makes me absolutely furious!",
                "reasoning": "Let's think step by step. The word 'furious' is a strong indicator of anger, and 'absolutely' intensifies this feeling. The comment expresses strong displeasure and antagonism towards something.",
                "answer": "true"
            },
            {
                "comment": "I'm so happy about this outcome",
                "reasoning": "Let's think step by step. This comment expresses happiness and satisfaction, which is the opposite of anger. There's no indication of displeasure or antagonism. This doesn't match the 'anger' emotion.",
                "answer": "false"
            }
        ],
        'joy': [
            {
                "comment": "I'm so happy and delighted!",
                "reasoning": "Let's think step by step. The comment explicitly states 'happy' and 'delighted' which are direct indicators of joy and positive feelings. This clearly expresses pleasure and happiness.",
                "answer": "true"
            },
            {
                "comment": "This makes me incredibly sad",
                "reasoning": "Let's think step by step. This comment expresses sadness, which is the opposite of joy. There's no indication of happiness or pleasure. This doesn't match the 'joy' emotion.",
                "answer": "false"
            }
        ],
        'neutral': [
            {
                "comment": "The meeting is scheduled for 3pm today",
                "reasoning": "Let's think step by step. This is a factual statement providing information about meeting time. There are no emotional words or expressions of feeling. This is purely informational and neutral.",
                "answer": "true"
            },
            {
                "comment": "I'm absolutely thrilled about this!",
                "reasoning": "Let's think step by step. This comment expresses strong positive emotion ('thrilled') with emphasis ('absolutely'). This shows excitement and joy, not neutrality. This doesn't match the 'neutral' emotion.",
                "answer": "false"
            }
        ]
    }
    
    # Get examples for the emotion
    examples = reasoning_examples.get(emotion, [])
    
    # Format with reasoning chains
    formatted = []
    for ex in examples:
        formatted.append(f"Comment: {ex['comment']}\n{ex['reasoning']}\nAnswer: {ex['answer']}")
    
    return "\n\n".join(formatted)

def get_auto_cot_prediction(text: str,
                          subreddit: str,
                          author: str, 
                          client,
                          emotion: str) -> Optional[bool]:
    """
    Get Auto-CoT prediction for a single emotion.
    
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
    
    # Get auto-generated CoT examples
    cot_examples = generate_auto_cot_examples(emotion)
    
    # Create Auto-CoT prompt
    prompt = f"""You are classifying emotions in Reddit comments.

Emotion: {emotion}
Definition: {prompt_descriptions[emotion]}

Here are some examples with reasoning:

{cot_examples}

Now classify this comment using the same step-by-step reasoning:

Subreddit: {subreddit}
Author: {author}
Comment: {text}

Let's think step by step."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Always think step by step and end with 'Answer: true' or 'Answer: false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract answer from reasoning
        if "answer: true" in result or "answer is true" in result or result.endswith("true"):
            return True
        elif "answer: false" in result or "answer is false" in result or result.endswith("false"):
            return False
        else:
            print(f"Could not extract answer from Auto-CoT response for {emotion}: {result}")
            return None
            
    except Exception as e:
        print(f"Error getting Auto-CoT prediction for {emotion}: {str(e)}")
        return None


def get_auto_cot_prediction_all_emotions(text: str,
                                       subreddit: str,
                                       author: str,
                                       client) -> List[str]:
    """
    Get Auto-CoT predictions for all GoEmotions categories.
    
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
        prediction = get_auto_cot_prediction(
            text, subreddit, author, client, emotion
        )
        
        if prediction is True:
            assigned_emotions.append(emotion)
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_emotions