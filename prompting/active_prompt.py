# goemotions_evaluation/prompting/active_prompt.py
"""
Active Prompting for GoEmotions emotion classification.
FIXED: 12 examples total (2 per main emotion) WITHOUT self-consistency
"""

import os
import sys
import time
import logging
import re
import ast
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
import openai
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def parse_emotion_response(response_text: str, valid_emotions: List[str]) -> List[str]:
    """Parse emotion response from LLM output"""
    response_text = response_text.strip()
    
    try:
        list_match = re.search(r'\[([^\]]+)\]', response_text)
        if list_match:
            list_str = '[' + list_match.group(1) + ']'
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                emotions = [str(item).strip().strip("'\"") for item in parsed]
                return [e for e in emotions if e in valid_emotions]
    except:
        pass
    
    found_emotions = []
    response_lower = response_text.lower()
    
    for emotion in valid_emotions:
        if emotion.lower() in response_lower:
            found_emotions.append(emotion)
    
    return found_emotions if found_emotions else ['neutral']

class ActivePromptSelector:
    """Implements Active Prompting methodology with uncertainty + wrong selection"""
    
    def __init__(self, pool_size: int = 20, k_samples: int = 5):
        self.pool_size = pool_size
        self.k_samples = k_samples
    
    def estimate_uncertainty_and_wrong(self, texts: List[str], client, emotion: str, 
                                      ground_truth_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Estimate uncertainty AND wrongness for a SPECIFIC emotion"""
        print(f"Estimating uncertainty and wrongness for emotion: {emotion}")
        
        uncertainty_scores = {}
        wrong_scores = {}
        
        for i, text in enumerate(texts):
            if (i + 1) % 5 == 0:
                print(f"Processing text {i + 1}/{len(texts)}")
            
            predictions = []
            for sample_idx in range(self.k_samples):
                pred = self._get_single_prediction(text, client, emotion)
                if pred is not None:
                    predictions.append(pred)
                time.sleep(0.05)
            
            if predictions:
                # Calculate uncertainty (disagreement)
                unique_predictions = len(set(predictions))
                uncertainty = unique_predictions / len(predictions)
                uncertainty_scores[text] = uncertainty
                
                # Get ground truth
                ground_truth = self._get_ground_truth(text, emotion, ground_truth_df)
                
                # Calculate wrongness (error rate)
                wrong_rate = sum(1 for p in predictions if emotion in p != ground_truth) / len(predictions)
                wrong_scores[text] = wrong_rate
            else:
                uncertainty_scores[text] = 0.0
                wrong_scores[text] = 0.0
        
        return uncertainty_scores, wrong_scores
    
    def _get_single_prediction(self, text: str, client, emotion: str) -> Optional[List[str]]:
        """Get a single emotion prediction"""
        prompt = f"""Does this text express {emotion.upper()} emotion?

Text: "{text[:200]}"

Answer with a Python list of emotions present.

Answer:"""

        try:
            import config
            response = client.chat.completions.create(
                model=config.MODEL_ID,
                messages=[
                    {"role": "system", "content": f"You are an expert at detecting emotions. Be precise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=20,
                timeout=8
            )
            
            result = response.choices[0].message.content.strip()
            # Simple parsing
            if emotion.lower() in result.lower():
                return [emotion]
            return []
            
        except Exception as e:
            pass
            
        return None
    
    def _get_ground_truth(self, text: str, emotion: str, df: pd.DataFrame) -> bool:
        """Get ground truth for a text and emotion"""
        matching_rows = df[df['text'] == text]
        
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            if emotion in row and row[emotion] == 1:
                return True
        return False
    
    def select_top_uncertain_and_wrong(self, uncertainty_scores: Dict[str, float], 
                                      wrong_scores: Dict[str, float]) -> Tuple[str, str]:
        """Select top 1 uncertain and top 1 wrong text"""
        top_uncertain = max(uncertainty_scores.items(), key=lambda x: x[1])[0] if uncertainty_scores else None
        top_wrong = max(wrong_scores.items(), key=lambda x: x[1])[0] if wrong_scores else None
        
        print(f"  Top uncertain: {top_uncertain[:60] if top_uncertain else 'None'}...")
        print(f"  Top wrong: {top_wrong[:60] if top_wrong else 'None'}...")
        
        return top_uncertain, top_wrong

def prepare_active_prompting_data(df: pd.DataFrame, client, n_examples: int = 12) -> List[Tuple[str, List[str], str]]:
    """Prepare 12 active prompting examples (2 per main emotion)"""
    print("Preparing Active Prompting data (12 EXAMPLES VERSION)...")
    
    main_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
    all_examples = []  # Will contain 12 examples total
    
    # Get 20 sample texts
    selector = ActivePromptSelector(pool_size=20, k_samples=5)
    sample_df = df.sample(n=min(len(df), 20), random_state=42)
    sample_texts = sample_df['text'].tolist()
    
    # Find uncertain and wrong examples for EACH EMOTION
    for emotion in main_emotions:
        print(f"\nProcessing emotion: {emotion}")
        
        try:
            # Get uncertainty and wrong scores
            uncertainty_scores, wrong_scores = selector.estimate_uncertainty_and_wrong(
                sample_texts, client, emotion, sample_df
            )
            
            # Select top 1 uncertain and top 1 wrong
            top_uncertain, top_wrong = selector.select_top_uncertain_and_wrong(
                uncertainty_scores, wrong_scores
            )
            
            # Add uncertain example
            if top_uncertain:
                matching_rows = sample_df[sample_df['text'] == top_uncertain]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    true_emotions = [e for e in main_emotions if e in row and row[e] == 1]
                    if not true_emotions:
                        true_emotions = ['neutral']
                    reasoning = f"This text expresses {', '.join(true_emotions)}."
                    all_examples.append((top_uncertain, true_emotions, reasoning))
            
            # Add wrong example (if different from uncertain)
            if top_wrong and top_wrong != top_uncertain:
                matching_rows = sample_df[sample_df['text'] == top_wrong]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    true_emotions = [e for e in main_emotions if e in row and row[e] == 1]
                    if not true_emotions:
                        true_emotions = ['neutral']
                    reasoning = f"This text expresses {', '.join(true_emotions)}."
                    all_examples.append((top_wrong, true_emotions, reasoning))
            
            print(f"✓ Selected examples for {emotion}")
            
        except Exception as e:
            print(f"⚠ Error in {emotion}: {e}")
    
    print(f"\n✓ Total examples collected: {len(all_examples)}")
    return all_examples[:12]  # Ensure max 12 examples

def get_active_prompt_prediction_all_emotions(text: str, subreddit: str, author: str, client, 
                                            uncertainty_examples: List[Tuple[str, List[str], str]] = None,
                                            model=None,
                                            use_self_consistency: bool = False,
                                            consistency_samples: int = 5) -> List[str]:
    """Get active prompt prediction WITHOUT self-consistency (like Bloom's)"""
    
    main_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
    all_emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # Build examples text from all 12 examples
    examples_text = ""
    if uncertainty_examples:
        examples_text = "EXAMPLES from uncertain and wrong cases:\n"
        for i, (ex_text, ex_emotions, ex_reasoning) in enumerate(uncertainty_examples[:12], 1):
            examples_text += f'Example {i}:\n'
            examples_text += f'Text: "{ex_text[:100]}..."\n'
            examples_text += f'Emotions: {ex_emotions}\n\n'
    
    prompt = f"""Classify this Reddit comment into emotions.

Available emotions: {', '.join(all_emotions)}

{examples_text}

Now classify this text, please think step by step:
Text: "{text[:150]}"
Subreddit: {subreddit}
Author: {author}

INSTRUCTIONS:
1. Learn from the examples provided above
2. Analyze the text carefully
3. Look for emotional words and tone
4. Most texts have 1-2 emotions maximum
5. Be conservative - only select clear emotions
6. If no clear emotion, choose 'neutral'

Answer with Python list like ['joy'] or ['anger', 'sadness'] or ['neutral']:"""

    try:
        import config
        response = client.chat.completions.create(
            model=config.MODEL_ID if model is None else model,
            messages=[
                {"role": "system", "content": "You are an emotion classifier. Think step by step and be selective. Answer with a Python list of emotions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=30,
            timeout=10
        )
        
        predicted_emotions = parse_emotion_response(response.choices[0].message.content, all_emotions)
        
        if not predicted_emotions:
            predicted_emotions = ['neutral']
        
        return predicted_emotions
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return ['neutral']