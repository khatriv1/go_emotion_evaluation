"""
Active prompting evaluation for GoEmotions emotion classification
FIXED: Multi-label approach instead of binary classification
"""

import os
import sys
import time
import logging
import re
import ast
from typing import Dict, List, Tuple, Optional
import pandas as pd
import openai
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.emotion_rubric import GoEmotionsRubric
from utils.metrics import GoEmotionsMetrics, create_detailed_results_dataframe
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
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

def get_active_prompt_prediction_all_emotions(text: str, subreddit: str, author: str, client, model="gpt-3.5-turbo") -> List[str]:
    """
    Get active prompt prediction for all emotions in a comment
    FIXED: Multi-label approach with uncertainty-based refinement
    
    Args:
        text: The text to classify (matches evaluation file parameter name)
        subreddit: Name of the subreddit
        author: Comment author
        client: OpenAI client
        model: Model to use for prediction
        
    Returns:
        List of emotions assigned to the comment
    """
    max_retries = getattr(config, 'MAX_RETRIES', 3)
    retry_delay = getattr(config, 'RETRY_DELAY', 1.0)
    
    try:
        # Initialize emotion rubric
        emotion_rubric = GoEmotionsRubric()
        emotions = emotion_rubric.get_all_emotions()
        
        # Step 1: Get initial prediction to identify uncertain emotions
        initial_prompt = f"""Classify this Reddit comment into the most appropriate emotions.

Available emotions: {emotion_rubric.format_emotions_for_prompt()}

Comment: "{text}"

Instructions:
- Select only PRIMARY emotions clearly expressed
- Most comments have 1-2 emotions maximum
- Be selective - don't over-predict
- If no clear emotion, select 'neutral'

Respond with a Python list of emotions: ['emotion1', 'emotion2']

Response:"""
        
        # Try initial prediction with error handling
        initial_emotions = ['neutral']  # fallback
        for attempt in range(max_retries):
            try:
                initial_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert emotion classifier. Always respond with a Python list of emotions."},
                        {"role": "user", "content": initial_prompt}
                    ],
                    max_tokens=getattr(config, 'OPENAI_MAX_TOKENS', 150),
                    temperature=getattr(config, 'OPENAI_TEMPERATURE', 0.1)
                )
                
                # Parse initial response
                initial_emotions = parse_emotion_response(initial_response.choices[0].message.content, emotions)
                if not initial_emotions:
                    initial_emotions = ['neutral']
                break
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit in initial prediction, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries exceeded for initial prediction")
                    break
                    
            except openai.APIError as e:
                logger.error(f"API error in initial prediction, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Unexpected error in initial prediction: {e}")
                break
        
        # Step 2: Active prompting for refinement with examples
        active_prompt = create_active_prompt(text, emotion_rubric, initial_emotions)
        
        # Try active prediction with error handling
        for attempt in range(max_retries):
            try:
                active_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert emotion classifier. Carefully consider the examples and respond with a Python list of emotions."},
                        {"role": "user", "content": active_prompt}
                    ],
                    max_tokens=getattr(config, 'OPENAI_MAX_TOKENS', 150),
                    temperature=getattr(config, 'OPENAI_TEMPERATURE', 0.1)
                )
                
                # Parse final response
                predicted_emotions = parse_emotion_response(active_response.choices[0].message.content, emotions)
                
                # Fallback to initial prediction if parsing fails
                if not predicted_emotions:
                    predicted_emotions = initial_emotions if initial_emotions else ['neutral']
                
                return predicted_emotions
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit in active prediction, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries exceeded for active prediction")
                    return initial_emotions if initial_emotions else ['neutral']
                    
            except openai.APIError as e:
                logger.error(f"API error in active prediction, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return initial_emotions if initial_emotions else ['neutral']
                    
            except Exception as e:
                logger.error(f"Unexpected error in active prediction: {e}")
                return initial_emotions if initial_emotions else ['neutral']
        
        # If we get here, return initial emotions as fallback
        return initial_emotions if initial_emotions else ['neutral']
        
    except Exception as e:
        logger.error(f"Error in get_active_prompt_prediction_all_emotions: {str(e)}")
        return ['neutral']

def create_active_prompt(text: str, emotion_rubric: GoEmotionsRubric, uncertain_emotions: List[str] = None) -> str:
    """
    Create an active prompt that focuses on uncertain emotions
    
    Args:
        text: The text to classify
        emotion_rubric: GoEmotionsRubric instance  
        uncertain_emotions: List of emotions to focus on (if any)
        
    Returns:
        Active prompt string
    """
    if uncertain_emotions is None:
        uncertain_emotions = []
    
    prompt = f"""Classify this Reddit comment into the most appropriate emotions.

Available emotions: {emotion_rubric.format_emotions_for_prompt()}

"""
    
    # Add focused examples for uncertain emotions
    if uncertain_emotions:
        prompt += f"""Based on initial analysis, these emotions might be present: {', '.join(uncertain_emotions)}

Here are some clarifying examples:
"""
        for emotion in uncertain_emotions[:3]:  # Limit to 3 emotions to avoid long prompts
            examples = emotion_rubric.get_emotion_examples(emotion, 1)
            if examples:
                prompt += f"\n{emotion.title()}: \"{examples[0]}\""
        
        prompt += "\n\n"
    
    prompt += f"""Now classify this comment:
Comment: "{text}"

Think step by step:
1. What emotions are clearly expressed?
2. Are there any subtle emotions that might be missed?
3. Consider the context and tone carefully
4. Be selective - most comments have 1-2 emotions maximum

Instructions:
- Select only PRIMARY emotions clearly expressed
- Don't over-predict - be conservative
- If no clear emotion, select 'neutral'

Respond with a Python list of emotions: ['emotion1', 'emotion2']

Response:"""
    
    return prompt

def evaluate_active_prompt(df: pd.DataFrame, emotions_list: List[str], output_dir: str, data_loader=None) -> Dict:
    """
    Evaluate using Active Prompting technique
    FIXED: Properly handle DataFrame structure and avoid ID-related errors
    
    Args:
        df: DataFrame with comments and emotions
        emotions_list: List of all emotions
        output_dir: Directory to save results
        data_loader: Data loader instance (optional)
        
    Returns:
        Dictionary containing evaluation results
    """
    try:
        # Initialize components
        rubric = GoEmotionsRubric()
        metrics_calculator = GoEmotionsMetrics(emotions_list)
        client = openai.OpenAI(api_key=getattr(config, 'OPENAI_API_KEY', os.getenv('OPENAI_API_KEY')))
        
        predictions = []
        
        print(f"Evaluating on {len(df)} comments")
        
        for idx, (index, row) in enumerate(df.iterrows(), 1):
            print(f"Processing comment {idx}/{len(df)}")
            
            try:
                comment = row['text']
                true_emotions = row['emotions'] if isinstance(row['emotions'], list) else []
                
                # Create a safe comment ID - FIXED: Use DataFrame index instead of undefined column
                comment_id = f"comment_{index}" if 'id' not in row else str(row['id'])
                
                print(f"Comment: {comment[:50]}...")
                
                # Get prediction using the new function
                pred_emotions = get_active_prompt_prediction_all_emotions(
                    comment, 
                    row.get('subreddit', ''), 
                    row.get('author', ''), 
                    client
                )
                
                print(f"Human: {true_emotions}")
                print(f"Model: {pred_emotions}")
                
                # Check exact match
                exact_match = set(true_emotions) == set(pred_emotions)
                print(f"Match: {exact_match}")
                print()
                
                # Store prediction
                prediction = {
                    'id': comment_id,
                    'text': comment,
                    'human_emotions': true_emotions,
                    'predicted_emotions': pred_emotions,
                    'exact_match': exact_match
                }
                predictions.append(prediction)
                
                # Add small delay to avoid rate limiting
                time.sleep(getattr(config, 'REQUEST_DELAY', 1.0))
                
            except Exception as e:
                print(f"Error processing comment {comment_id}: {e}")
                # Continue with next comment instead of failing completely
                continue
        
        # Calculate metrics if we have predictions
        if not predictions:
            print("No successful predictions to evaluate")
            return None
        
        # Extract true and predicted emotions for metrics
        y_true = [pred['human_emotions'] for pred in predictions]
        y_pred = [pred['predicted_emotions'] for pred in predictions]
        
        # Calculate comprehensive metrics
        results = metrics_calculator.comprehensive_evaluation(y_true, y_pred)
        
        # Print results
        metrics_calculator.print_results(results, "Active Prompting")
        
        # Create detailed results DataFrame
        detailed_df = create_detailed_results_dataframe(predictions, emotions_list)
        
        # Save detailed results
        detailed_path = os.path.join(output_dir, "detailed_results.csv")
        detailed_df.to_csv(detailed_path, index=False)
        
        # Add detailed results to return dict
        results['detailed_results'] = detailed_df
        results['predictions'] = predictions
        
        print(f"Detailed results saved to {detailed_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Active Prompting evaluation failed: {e}")
        print(f"✗ Active Prompting failed: {e}")
        return None