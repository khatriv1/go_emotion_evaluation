# goemotions_evaluation/prompting/active_prompt.py

"""
Active prompting for GoEmotions emotion classification.
Iteratively selects informative examples to improve classification.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Tuple
from utils.emotion_rubric import GoEmotionsRubric

class ActivePromptSelector:
    """Selects most informative examples for active learning."""
    
    def __init__(self):
        self.example_pool = self._initialize_example_pool()
        self.selected_examples = {emotion: [] for emotion in self.example_pool.keys()}
        self.uncertainty_scores = {}
    
    def _initialize_example_pool(self) -> Dict[str, List[Tuple[str, bool]]]:
        """Initialize pool of examples for active selection."""
        return {
            'admiration': [
                ("That was absolutely incredible work!", True),
                ("I really respect what you accomplished", True),
                ("What's for dinner?", False),
                ("Brilliant performance, truly impressive", True),
                ("Random comment here", False)
            ],
            'amusement': [
                ("Haha this made me laugh so hard", True),
                ("This is hilarious! 😂", True),
                ("I'm really sad about this", False),
                ("LOL that's so funny", True),
                ("The weather is nice", False)
            ],
            'anger': [
                ("This makes me absolutely furious!", True),
                ("I'm so angry I could scream", True),
                ("Thanks for the help", False),
                ("This is infuriating beyond belief", True),
                ("Good morning everyone", False)
            ],
            'joy': [
                ("I'm so happy about this!", True),
                ("This brings me such joy and delight", True),
                ("I need to buy groceries", False),
                ("I feel absolutely wonderful!", True),
                ("The meeting is at 3pm", False)
            ],
            'sadness': [
                ("This makes me so sad and heartbroken", True),
                ("I feel really down about this", True),
                ("Great job on the project", False),
                ("I'm feeling so depressed", True),
                ("See you tomorrow", False)
            ],
            'fear': [
                ("I'm really scared about this", True),
                ("This makes me terrified", True),
                ("I love ice cream", False),
                ("I'm worried and anxious", True),
                ("The store closes at 9pm", False)
            ],
            'surprise': [
                ("Wow, I didn't expect that at all!", True),
                ("What a shocking surprise!", True),
                ("I knew this would happen", False),
                ("I'm completely stunned", True),
                ("Regular daily routine", False)
            ],
            'neutral': [
                ("The meeting is scheduled for 3pm", True),
                ("Here is the requested information", True),
                ("I'm absolutely thrilled!", False),
                ("Weather forecast shows rain", True),
                ("This is incredibly exciting!", False)
            ]
        }
    
    def select_examples(self, emotion: str, n_examples: int = 3) -> List[Tuple[str, str]]:
        """Select most informative examples using uncertainty sampling."""
        available_examples = self.example_pool.get(emotion, [])
        
        # For first iteration, select diverse examples
        if not self.selected_examples[emotion]:
            # Select one positive, one negative, and one uncertain
            positive = [ex for ex in available_examples if ex[1]]
            negative = [ex for ex in available_examples if not ex[1]]
            
            selected = []
            if positive:
                selected.append((positive[0][0], "true"))
            if negative:
                selected.append((negative[0][0], "false"))
            if len(positive) > 1:
                selected.append((positive[1][0], "true"))
            
            return selected[:n_examples]
        
        # For subsequent iterations, use uncertainty scores
        return self._select_by_uncertainty(emotion, n_examples)
    
    def _select_by_uncertainty(self, emotion: str, n_examples: int) -> List[Tuple[str, str]]:
        """Select examples with highest uncertainty."""
        examples = self.example_pool.get(emotion, [])
        selected = []
        
        for comment, label in examples[:n_examples]:
            selected.append((comment, "true" if label else "false"))
        
        return selected

def get_active_prompt_prediction(text: str,
                               subreddit: str,
                               author: str, 
                               client,
                               emotion: str,
                               selector: ActivePromptSelector = None) -> Optional[bool]:
    """
    Get active prompting prediction for a single emotion.
    
    Args:
        text: The Reddit comment text
        subreddit: Name of the subreddit
        author: Comment author
        client: OpenAI client
        emotion: Emotion to classify for
        selector: Active prompt selector instance
    
    Returns:
        Boolean indicating if comment expresses this emotion, None if failed
    """
    rubric = GoEmotionsRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    if emotion not in prompt_descriptions:
        raise ValueError(f"Unknown emotion: {emotion}")
    
    # Initialize selector if not provided
    if selector is None:
        selector = ActivePromptSelector()
    
    # Get actively selected examples
    selected_examples = selector.select_examples(emotion, n_examples=3)
    
    # Format examples
    examples_text = []
    for ex_comment, ex_label in selected_examples:
        examples_text.append(f"Comment: {ex_comment}\nAnswer: {ex_label}")
    examples_formatted = "\n\n".join(examples_text)
    
    # Create active prompt
    prompt = f"""You are classifying emotions in Reddit comments.

Emotion: {emotion}
Definition: {prompt_descriptions[emotion]}

Here are carefully selected examples:

{examples_formatted}

Now classify this comment:

Subreddit: {subreddit}
Author: {author}
Comment: {text}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Learn from the examples provided. Respond only with 'true' or 'false'."},
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
        print(f"Error getting active prompt prediction for {emotion}: {str(e)}")
        return None


def get_active_prompt_prediction_all_emotions(text: str,
                                            subreddit: str,
                                            author: str,
                                            client) -> List[str]:
    """
    Get active prompting predictions for all GoEmotions categories.
    
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
    selector = ActivePromptSelector()  # Share selector across emotions
    
    for emotion in emotions:
        prediction = get_active_prompt_prediction(
            text, subreddit, author, client, emotion, selector
        )
        
        if prediction is True:
            assigned_emotions.append(emotion)
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_emotions