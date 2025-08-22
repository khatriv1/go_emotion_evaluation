# goemotions_evaluation/prompting/zero_shot.py
"""
Zero-shot prompting for GoEmotions emotion classification.
FIXED: Consistent with Bloom's structure
"""

import time
import logging
from typing import List
import re
import ast
import config
import openai

logger = logging.getLogger(__name__)

def parse_emotion_response(response_text: str, valid_emotions: List[str]) -> List[str]:
    """Parse emotion response from LLM output"""
    response_text = response_text.strip()
    
    try:
        # Look for list pattern like ['emotion1', 'emotion2']
        list_match = re.search(r'\[([^\]]+)\]', response_text)
        if list_match:
            list_str = '[' + list_match.group(1) + ']'
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                emotions = [str(item).strip().strip("'\"") for item in parsed]
                # Filter to valid emotions only
                return [e for e in emotions if e in valid_emotions]
    except:
        pass
    
    # Try to find emotions mentioned in the text
    found_emotions = []
    response_lower = response_text.lower()
    
    for emotion in valid_emotions:
        if emotion.lower() in response_lower:
            found_emotions.append(emotion)
    
    return found_emotions if found_emotions else ['neutral']

def get_zero_shot_prediction_all_emotions(text: str,
                                        subreddit: str, 
                                        author: str,
                                        client) -> List[str]:
    """
    Get zero-shot predictions using multi-label approach.
    FIXED: Single comprehensive prompt instead of 28 binary questions
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # Create comprehensive multi-label prompt
    prompt = f"""Classify this Reddit comment into the most appropriate emotions.

Available emotions: {', '.join(emotions)}

Comment: "{text}"
Subreddit: {subreddit}
Author: {author}

Instructions:
- Select only the PRIMARY emotions clearly expressed
- Most comments have 1-2 emotions maximum
- Be selective - don't over-predict
- If no clear emotion, select 'neutral'
- Consider the context and tone carefully

Respond with a Python list of emotions: ['emotion1', 'emotion2']

Response:"""

    max_retries = getattr(config, 'MAX_RETRIES', 3)
    retry_delay = getattr(config, 'RETRY_DELAY', 1.0)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.MODEL_ID,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing emotions in text. Be selective and only choose emotions that are clearly expressed. Respond with a Python list."},
                    {"role": "user", "content": prompt}
                ],
                temperature=getattr(config, 'OPENAI_TEMPERATURE', 0.0),
                max_tokens=getattr(config, 'OPENAI_MAX_TOKENS', 100)
            )
            
            result = response.choices[0].message.content.strip()
            predicted_emotions = parse_emotion_response(result, emotions)
            
            # Fallback to neutral if no emotions found
            if not predicted_emotions:
                predicted_emotions = ['neutral']
                
            return predicted_emotions
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries exceeded for rate limit")
                return ['neutral']
                
        except openai.APIError as e:
            logger.error(f"OpenAI API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("Max retries exceeded for API error")
                return ['neutral']
                
        except openai.APIConnectionError as e:
            logger.error(f"Connection error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("Max retries exceeded for connection error")
                return ['neutral']
                
        except Exception as e:
            logger.error(f"Unexpected error in zero-shot prediction: {e}")
            return ['neutral']
    
    # If we get here, all retries failed
    logger.error("All retry attempts failed for zero-shot prediction")
    return ['neutral']