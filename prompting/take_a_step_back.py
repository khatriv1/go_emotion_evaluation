# goemotions_evaluation/prompting/take_a_step_back.py

"""
Take a Step Back prompting for GoEmotions emotion classification.
FIXED: Multi-label approach with high-level principles
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

def derive_classification_principles(client) -> str:
    """
    Derive high-level principles for emotion classification.
    """
    prompt = f"""Take a step back and think about the fundamental principles for identifying emotions in Reddit comments.

What are the key characteristics, patterns, and principles that would help identify emotions in text? 

List 5-7 high-level principles for emotion classification:"""

    max_retries = getattr(config, 'MAX_RETRIES', 3)
    retry_delay = getattr(config, 'RETRY_DELAY', 1.0)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing emotional patterns in text. Derive clear, high-level principles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded during principle derivation on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
            else:
                logger.error("Max retries exceeded for principle derivation")
                return "Consider the explicit emotional words, context, and tone of the comment."
                
        except openai.APIError as e:
            logger.error(f"API error during principle derivation on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return "Consider the explicit emotional words, context, and tone of the comment."
                
        except Exception as e:
            logger.error(f"Error deriving principles: {e}")
            return "Consider the explicit emotional words, context, and tone of the comment."
    
    return "Consider the explicit emotional words, context, and tone of the comment."

def get_take_step_back_prediction_all_emotions(text: str,
                                             subreddit: str,
                                             author: str,
                                             client) -> List[str]:
    """
    Get Take a Step Back predictions using multi-label approach.
    FIXED: Single comprehensive prompt with high-level principles
    """
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # Step 1: Derive high-level principles (cached for efficiency)
    logger.info("Deriving classification principles...")
    principles = derive_classification_principles(client)
    
    # Step 2: Apply principles to classify
    prompt = f"""Classify this Reddit comment using high-level emotion classification principles.

High-level principles for emotion classification:
{principles}

Available emotions: {', '.join(emotions)}

Comment: "{text}"
Subreddit: {subreddit}
Author: {author}

Apply the principles above to classify this comment:
1. What high-level patterns do you see?
2. What emotions are clearly expressed based on the principles?
3. Be selective - most comments have 1-2 emotions maximum

Instructions:
- Select only PRIMARY emotions clearly expressed
- Don't over-predict - follow the principles
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
                    {"role": "system", "content": "You are an expert at analyzing emotions in text. Apply the given principles to make accurate classifications. Be selective."},
                    {"role": "user", "content": prompt}
                ],
                temperature=getattr(config, 'OPENAI_TEMPERATURE', 0.0),
                max_tokens=getattr(config, 'OPENAI_MAX_TOKENS', 150)
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
            logger.error(f"Unexpected error in take-step-back prediction: {e}")
            return ['neutral']
    
    # If we get here, all retries failed
    logger.error("All retry attempts failed for take-step-back prediction")
    return ['neutral']