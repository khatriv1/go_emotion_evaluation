import time
import re
from typing import List, Optional

from utils.emotion_rubric import GoEmotionsRubric


def get_contrastive_examples(emotion: str) -> str:
    """
    Retrieve and format contrastive examples for this emotion.
    Expects the rubric to return a tuple of (positive_examples, negative_examples).
    """
    rubric = GoEmotionsRubric()
    examples = rubric.get_contrastive_examples(emotion)
    try:
        pos_examples, neg_examples = examples
    except Exception:
        raise ValueError(f"Expected contrastive examples for '{emotion}' as a tuple of (pos, neg), got: {examples}")

    lines = ["Contrastive Examples:"]
    lines += [f"- {ex}" for ex in pos_examples]
    lines.append("")  # blank line between pos and neg
    lines += [f"- {ex}" for ex in neg_examples]
    return "\n".join(lines)


def get_contrastive_cot_prediction(
    text: str,
    subreddit: str,
    author: str,
    client,
    emotion: str
) -> Optional[bool]:
    """
    Perform a contrastive Chain-of-Thought analysis for a given emotion,
    forcing a final `Final Answer: true` or `false` line for reliable parsing.
    """
    rubric = GoEmotionsRubric()
    desc = rubric.get_prompt_descriptions().get(emotion)
    if not desc:
        raise ValueError(f"Unknown emotion: {emotion}")

    examples_block = get_contrastive_examples(emotion)
    prompt = f"""Emotion: {emotion}
Definition: {desc}

{examples_block}

Now analyze this comment:

Subreddit: {subreddit}
Author: {author}
Comment: {text}

1) Positive reasoning (why it MIGHT express {emotion})  
2) Negative reasoning (why it MIGHT NOT express {emotion})  

On its own line **write exactly** `Final Answer: true` or `Final Answer: false`."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at emotion classification. Use contrastive reasoning and end with a single Final Answer line."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0,
            max_tokens=250
        )
        result = response.choices[0].message.content.strip().lower()

        # Scan backwards for the explicit Final Answer line
        for line in reversed(result.splitlines()):
            m = re.search(r"\bfinal answer:\s*(true|false)\b", line)
            if m:
                return (m.group(1) == "true")

        # Fallback: pick the last standalone true/false token
        tokens = re.findall(r"\b(true|false)\b", result)
        if tokens:
            return (tokens[-1] == "true")

        print(f"Could not extract answer from Contrastive CoT for {emotion}: {result}")
        return None

    except Exception as e:
        print(f"Error in Contrastive CoT for {emotion}: {e}")
        return None


def get_contrastive_cot_prediction_all_emotions(
    text: str,
    subreddit: str,
    author: str,
    client
) -> List[str]:
    """
    Run contrastive CoT over all emotions and return those predicted True.
    """
    rubric = GoEmotionsRubric()
    emotions = list(rubric.get_prompt_descriptions().keys())

    assigned = []
    for emo in emotions:
        pred = get_contrastive_cot_prediction(text, subreddit, author, client, emo)
        if pred:
            assigned.append(emo)
        time.sleep(0.5)
    return assigned
