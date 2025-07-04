# goemotions_evaluation/prompting/few_shot.py

"""
Few-shot prompting for GoEmotions emotion classification.
Provides examples before asking for classification.
"""

import time
from typing import List, Optional
from utils.emotion_rubric import GoEmotionsRubric

def get_few_shot_examples(emotion: str) -> str:
    """Get few-shot examples for each emotion."""
    examples = {
        'admiration': [
            ("That was an incredible performance!", "true"),
            ("I really respect what you did there", "true"),
            ("What time is it?", "false")
        ],
        'amusement': [
            ("Haha that made me laugh out loud", "true"),
            ("This is hilarious 😂", "true"),
            ("I'm really sad about this", "false")
        ],
        'anger': [
            ("This makes me absolutely furious!", "true"),
            ("I'm so angry I could scream", "true"),
            ("Thanks for the help", "false")
        ],
        'annoyance': [
            ("This is really getting on my nerves", "true"),
            ("How annoying can this be", "true"),
            ("I love this so much", "false")
        ],
        'approval': [
            ("I completely agree with this", "true"),
            ("Yes, that's exactly right", "true"),
            ("I have no idea what this means", "false")
        ],
        'caring': [
            ("I hope you feel better soon", "true"),
            ("Take care of yourself", "true"),
            ("I don't care at all", "false")
        ],
        'confusion': [
            ("I don't understand this at all", "true"),
            ("This is really confusing to me", "true"),
            ("That makes perfect sense", "false")
        ],
        'curiosity': [
            ("I wonder what will happen next", "true"),
            ("Can you tell me more about this?", "true"),
            ("I already know everything about this", "false")
        ],
        'desire': [
            ("I really want to try that", "true"),
            ("I wish I could have that", "true"),
            ("I have no interest in this", "false")
        ],
        'disappointment': [
            ("I expected so much better than this", "true"),
            ("This is really disappointing", "true"),
            ("This exceeded my expectations", "false")
        ],
        'disapproval': [
            ("I don't agree with this at all", "true"),
            ("This is completely wrong", "true"),
            ("I fully support this decision", "false")
        ],
        'disgust': [
            ("That's absolutely disgusting", "true"),
            ("This makes me feel sick", "true"),
            ("That looks delicious", "false")
        ],
        'embarrassment': [
            ("I feel so embarrassed about this", "true"),
            ("That was really awkward", "true"),
            ("I'm proud of what I did", "false")
        ],
        'excitement': [
            ("I can't wait for this to happen!", "true"),
            ("This is so exciting!", "true"),
            ("This is really boring", "false")
        ],
        'fear': [
            ("I'm really scared about this", "true"),
            ("This makes me very worried", "true"),
            ("I feel completely safe", "false")
        ],
        'gratitude': [
            ("Thank you so much for this", "true"),
            ("I really appreciate your help", "true"),
            ("I don't owe you anything", "false")
        ],
        'grief': [
            ("I miss them so much it hurts", "true"),
            ("This loss is devastating", "true"),
            ("I couldn't be happier", "false")
        ],
        'joy': [
            ("I'm so happy about this!", "true"),
            ("This brings me such joy", "true"),
            ("This makes me miserable", "false")
        ],
        'love': [
            ("I love this so much", "true"),
            ("You mean everything to me", "true"),
            ("I hate this completely", "false")
        ],
        'nervousness': [
            ("I'm nervous about tomorrow", "true"),
            ("This makes me really anxious", "true"),
            ("I feel completely relaxed", "false")
        ],
        'optimism': [
            ("Things will definitely get better", "true"),
            ("I'm confident this will work out", "true"),
            ("There's no hope left", "false")
        ],
        'pride': [
            ("I'm so proud of this achievement", "true"),
            ("This makes me feel accomplished", "true"),
            ("I'm ashamed of what I did", "false")
        ],
        'realization': [
            ("I just realized something important", "true"),
            ("Now I finally understand", "true"),
            ("I'm still completely confused", "false")
        ],
        'relief': [
            ("I'm so relieved it's finally over", "true"),
            ("Thank goodness that worked out", "true"),
            ("The stress is just beginning", "false")
        ],
        'remorse': [
            ("I really regret doing that", "true"),
            ("I feel so bad about what happened", "true"),
            ("I'm glad I did that", "false")
        ],
        'sadness': [
            ("This makes me so sad", "true"),
            ("I feel really down about this", "true"),
            ("This makes me extremely happy", "false")
        ],
        'surprise': [
            ("Wow, I didn't expect that at all!", "true"),
            ("What a surprise!", "true"),
            ("That was completely predictable", "false")
        ],
        'neutral': [
            ("The meeting is scheduled for 3pm", "true"),
            ("Here is the requested information", "true"),
            ("I'm absolutely thrilled about this!", "false")
        ]
    }
    
    # Format examples for prompt
    formatted_examples = []
    for comment, label in examples.get(emotion, []):
        formatted_examples.append(f"Comment: {comment}\nAnswer: {label}")
    
    return "\n\n".join(formatted_examples)

def get_few_shot_prediction(text: str,
                          subreddit: str,
                          author: str, 
                          client,
                          emotion: str) -> Optional[bool]:
    """
    Get few-shot prediction for a single emotion.
    
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
    
    # Get examples for this emotion
    examples = get_few_shot_examples(emotion)
    
    # Create few-shot prompt
    prompt = f"""You are classifying emotions in Reddit comments.

Emotion: {emotion}
Definition: {prompt_descriptions[emotion]}

Here are some examples:

{examples}

Now classify this comment:

Subreddit: {subreddit}
Author: {author}
Comment: {text}

Answer:"""

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
        print(f"Error getting few-shot prediction for {emotion}: {str(e)}")
        return None


def get_few_shot_prediction_all_emotions(text: str,
                                       subreddit: str,
                                       author: str,
                                       client) -> List[str]:
    """
    Get few-shot predictions for all GoEmotions categories.
    
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
        prediction = get_few_shot_prediction(
            text, subreddit, author, client, emotion
        )
        
        if prediction is True:
            assigned_emotions.append(emotion)
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_emotions