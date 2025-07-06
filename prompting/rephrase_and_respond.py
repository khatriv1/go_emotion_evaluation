# goemotions_evaluation/prompting/rephrase_and_respond.py

"""
Rephrase and Respond prompting for GoEmotions emotion classification.
FIXED: Multi-label approach with rephrasing step
"""

import time
import re
import ast
import logging
from typing import List, Optional
import config
import openai

# Set up logging
logger = logging.getLogger(__name__)

def parse_emotion_response(response_text: str, valid_emotions: List[str]) -> List[str]:
    """Parse emotion response from LLM output"""
    response_text = response_text.strip()
    
    # Try to parse as Python list first
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
    
    return found_emotions

def rephrase_comment(comment: str, client) -> Optional[str]:
    """
    Rephrase the comment to clarify its meaning.
    """
    prompt = f"""Rephrase the following Reddit comment to make its emotional intent and meaning clearer, while preserving all important information:

Original comment: "{comment}"

Rephrased comment:"""

    max_retries = getattr(config, 'MAX_RETRIES', 3)
    retry_delay = getattr(config, 'RETRY_DELAY', 1.0)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert at understanding and clarifying the emotional meaning of comments. Rephrase to make emotional intent clear while preserving all information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded during rephrasing on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
            else:
                logger.error("Max retries exceeded for rephrasing")
                return None
                
        except openai.APIError as e:
            logger.error(f"API error during rephrasing on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error rephrasing comment: {e}")
            return None
    
    return None

def get_rephrase_respond_prediction_all_emotions(text: str,
                                               subreddit: str,
                                               author: str,
                                               client) -> List[str]:
    """
    Get Rephrase and Respond predictions using multi-label approach.
    FIXED: Single comprehensive prompt after rephrasing
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # Step 1: Rephrase the comment
    rephrased = rephrase_comment(text, client)
    if rephrased is None:
        logger.warning("Failed to rephrase comment, using original")
        rephrased = text
    
    # Step 2: Classify using both original and rephrased versions
    prompt = f"""Classify this Reddit comment into the most appropriate emotions.

Available emotions: {', '.join(emotions)}

Original comment: "{text}"
Rephrased for clarity: "{rephrased}"
Subreddit: {subreddit}
Author: {author}

Instructions:
- Consider both the original and rephrased versions
- Select only PRIMARY emotions clearly expressed
- Most comments have 1-2 emotions maximum
- Be selective - don't over-predict
- If no clear emotion, select 'neutral'

Response as Python list: ['emotion1', 'emotion2']

Response:"""

    max_retries = getattr(config, 'MAX_RETRIES', 3)
    retry_delay = getattr(config, 'RETRY_DELAY', 1.0)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing emotions in text. Consider both the original and rephrased versions to make accurate classifications. Be selective."},
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
            logger.error(f"Unexpected error in rephrase-respond prediction: {e}")
            return ['neutral']
    
    # If we get here, all retries failed
    logger.error("All retry attempts failed for rephrase-respond prediction")
    return ['neutral']