# goemotions_evaluation/prompting/cot.py

import time
import re
from typing import List, Optional
from utils.emotion_rubric import GoEmotionsRubric

def get_cot_prediction(text: str,
                       subreddit: str,
                       author: str, 
                       client,
                       emotion: str) -> Optional[bool]:
    """
    Get Chain of Thought prediction for a single emotion.
    Ensures a final dedicated line with 'true' or 'false' for reliable parsing.
    """
    rubric = GoEmotionsRubric()
    descriptions = rubric.get_prompt_descriptions()
    if emotion not in descriptions:
        raise ValueError(f"Unknown emotion: {emotion}")

    # Build prompt with an explicit final-answer line
    prompt = f"""Consider the Reddit comment below:

Subreddit: {subreddit}
Author: {author}
Comment: {text}

Emotion to check: {emotion}
Definition: {descriptions[emotion]}

Step 1: What is the main content/sentiment of this comment?
Step 2: Does it express {emotion} per the definition?
Step 3: What words or phrases support your decision?

Finally, on its own line **write exactly** `Final Answer: true`  
or `Final Answer: false`, and nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing emotions in text. Think step by step and conclude with a single Final Answer line."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        result = response.choices[0].message.content.strip().lower()

        # Look for the last occurrence of 'true' or 'false'
        for line in reversed(result.splitlines()):
            m = re.search(r"\b(final answer:\s*)(true|false)\b", line)
            if m:
                return (m.group(2) == "true")

        # Fallback: any standalone true/false token
        m2 = re.findall(r"\b(true|false)\b", result)
        if m2:
            return (m2[-1] == "true")

        print(f"Could not extract true/false from CoT response for {emotion}: {result}")
        return None

    except Exception as e:
        print(f"Error getting CoT prediction for {emotion}: {e}")
        return None


def get_cot_prediction_all_emotions(text: str,
                                    subreddit: str,
                                    author: str,
                                    client) -> List[str]:
    """
    Get CoT predictions for all emotions.
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    assigned = []
    for emo in emotions:
        pred = get_cot_prediction(text, subreddit, author, client, emo)
        if pred:
            assigned.append(emo)
        time.sleep(0.5)
    return assigned
