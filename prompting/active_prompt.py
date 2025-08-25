# goemotions_evaluation/prompting/active_prompt.py
"""
Active Prompting for GoEmotions
- Generates 56 candidates (2 per emotion × 28 emotions)
- Selects TOP 12 examples with highest uncertainty/wrong scores
- Uses selected 12 as few-shot examples
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

# All 28 emotions
ALL_EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness',
    'surprise', 'neutral'
]

def parse_emotion_response(response_text: str, valid_emotions: List[str]) -> List[str]:
    """Parse emotion response from LLM output"""
    response_text = response_text.strip()
    
    try:
        # Try to find Python list format
        list_match = re.search(r'\[([^\]]+)\]', response_text)
        if list_match:
            list_str = '[' + list_match.group(1) + ']'
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                emotions = [str(item).strip().strip("'\"") for item in parsed]
                return [e for e in emotions if e in valid_emotions]
    except:
        pass
    
    # Fallback: search for emotion words
    found_emotions = []
    response_lower = response_text.lower()
    
    for emotion in valid_emotions:
        if emotion.lower() in response_lower:
            found_emotions.append(emotion)
    
    return found_emotions if found_emotions else ['neutral']

class ActivePromptSelector:
    """Generate and select high-quality examples for Active Prompting"""
    
    def __init__(self, k_samples: int = 5):
        self.k_samples = k_samples
    
    def calculate_uncertainty_for_emotion(
        self, 
        text: str, 
        emotion: str, 
        client,
        n_samples: int = 5
    ) -> float:
        """Calculate uncertainty for a specific emotion prediction"""
        
        predictions = []
        for _ in range(n_samples):
            prompt = f"""Does this text express {emotion}?
Text: "{text[:200]}"

Answer with Python list of emotions. Include '{emotion}' if present."""

            try:
                import config
                response = client.chat.completions.create(
                    model=config.MODEL_ID,
                    messages=[
                        {"role": "system", "content": "You are an emotion detector. Be precise."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=30,
                    timeout=8
                )
                
                result = parse_emotion_response(response.choices[0].message.content, ALL_EMOTIONS)
                predictions.append(emotion in result)
                
            except Exception as e:
                print(f"Error in prediction: {e}")
                predictions.append(False)
            
            time.sleep(0.05)
        
        # Calculate uncertainty (entropy-based)
        if predictions:
            pos_rate = sum(predictions) / len(predictions)
            # Maximum uncertainty when pos_rate = 0.5
            if pos_rate == 0 or pos_rate == 1:
                uncertainty = 0.0
            else:
                uncertainty = -pos_rate * np.log(pos_rate) - (1-pos_rate) * np.log(1-pos_rate)
        else:
            uncertainty = 0.0
        
        return uncertainty
    
    def calculate_wrongness_for_emotion(
        self,
        text: str,
        emotion: str,
        ground_truth: bool,
        client,
        n_samples: int = 5
    ) -> float:
        """Calculate wrongness rate for a specific emotion"""
        
        wrong_count = 0
        for _ in range(n_samples):
            prompt = f"""Does this text express {emotion}?
Text: "{text[:200]}"

Answer with Python list of emotions."""

            try:
                import config
                response = client.chat.completions.create(
                    model=config.MODEL_ID,
                    messages=[
                        {"role": "system", "content": "You are an emotion detector."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=30,
                    timeout=8
                )
                
                result = parse_emotion_response(response.choices[0].message.content, ALL_EMOTIONS)
                prediction = emotion in result
                
                if prediction != ground_truth:
                    wrong_count += 1
                    
            except Exception as e:
                print(f"Error: {e}")
                wrong_count += 1
            
            time.sleep(0.05)
        
        return wrong_count / n_samples if n_samples > 0 else 0.0

def prepare_active_prompting_data(
    df: pd.DataFrame, 
    client, 
    n_examples: int = 12
) -> List[Tuple[str, List[str], str]]:
    """
    Enhanced Active Prompting:
    1. Generate 56 candidates (2 per emotion)
    2. Select TOP 12 with highest scores
    """
    print("\n" + "="*60)
    print("ACTIVE PROMPTING: 56 Candidates → Top 12 Selection")
    print("="*60)
    
    selector = ActivePromptSelector(k_samples=5)
    all_candidates = []
    used_texts = set()
    
    # Ensure we have enough data
    pool_size = min(len(df), 90)
    pool_df = df.head(pool_size)
    print(f"Using pool of {pool_size} samples")
    
    # PHASE 1: Generate candidates for each emotion
    print("\n📊 PHASE 1: Generating 56 candidates (2 per emotion)...")
    
    for emotion_idx, emotion in enumerate(ALL_EMOTIONS, 1):
        print(f"\n[{emotion_idx}/28] Processing: {emotion}")
        
        emotion_candidates = []
        
        # Calculate scores for all texts for this emotion
        for idx, row in pool_df.iterrows():
            text = row['text']
            
            # Skip if already used
            if text in used_texts:
                continue
            
            # Get ground truth for this emotion
            ground_truth = False
            if emotion in row.index:
                try:
                    value = row[emotion]
                    ground_truth = pd.notna(value) and int(value) == 1
                except:
                    pass
            
            # Calculate uncertainty
            uncertainty = selector.calculate_uncertainty_for_emotion(
                text, emotion, client, n_samples=5
            )
            
            # Calculate wrongness
            wrongness = selector.calculate_wrongness_for_emotion(
                text, emotion, ground_truth, client, n_samples=5
            )
            
            # Get all true emotions for this text
            true_emotions = []
            for em in ALL_EMOTIONS:
                if em in row.index:
                    try:
                        if pd.notna(row[em]) and int(row[em]) == 1:
                            true_emotions.append(em)
                    except:
                        pass
            
            if not true_emotions:
                true_emotions = ['neutral']
            
            # Store candidate with scores
            emotion_candidates.append({
                'text': text,
                'emotions': true_emotions,
                'target_emotion': emotion,
                'uncertainty': uncertainty,
                'wrongness': wrongness,
                'combined_score': uncertainty + wrongness,  # Combined metric
                'ground_truth': ground_truth
            })
        
        # Sort by scores and select top 2 for this emotion
        emotion_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Take top uncertain example
        if len(emotion_candidates) > 0:
            best_uncertain = max(emotion_candidates, key=lambda x: x['uncertainty'])
            if best_uncertain['text'] not in used_texts:
                all_candidates.append(best_uncertain)
                used_texts.add(best_uncertain['text'])
                print(f"  ✓ Uncertain example: score={best_uncertain['uncertainty']:.3f}")
        
        # Take top wrong example (different from uncertain)
        for candidate in sorted(emotion_candidates, key=lambda x: x['wrongness'], reverse=True):
            if candidate['text'] not in used_texts:
                all_candidates.append(candidate)
                used_texts.add(candidate['text'])
                print(f"  ✓ Wrong example: score={candidate['wrongness']:.3f}")
                break
    
    print(f"\n📊 Total candidates generated: {len(all_candidates)}")
    
    # PHASE 2: Select TOP 12 examples from all candidates
    print("\n📊 PHASE 2: Selecting TOP 12 examples...")
    
    # Sort all candidates by combined score
    all_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Select top 12
    selected_examples = []
    emotions_covered = set()
    
    for candidate in all_candidates[:12]:
        text = candidate['text']
        emotions = candidate['emotions']
        target = candidate['target_emotion']
        score = candidate['combined_score']
        
        # Create reasoning
        reasoning = f"This text expresses {', '.join(emotions)}."
        if target in emotions:
            reasoning += f" It clearly shows {target}."
        else:
            reasoning += f" It does not show {target}."
        
        selected_examples.append((text, emotions, reasoning))
        emotions_covered.update(emotions)
        
        print(f"  Selected: {target} | Score: {score:.3f} | Emotions: {emotions}")
    
    print(f"\n✅ Selected 12 examples covering {len(emotions_covered)} unique emotions")
    print(f"Emotions covered: {sorted(emotions_covered)}")
    
    # If we don't have exactly 12, add more from pool
    while len(selected_examples) < 12:
        for idx, row in pool_df.iterrows():
            if row['text'] not in used_texts and len(selected_examples) < 12:
                true_emotions = []
                for em in ALL_EMOTIONS:
                    if em in row.index:
                        try:
                            if pd.notna(row[em]) and int(row[em]) == 1:
                                true_emotions.append(em)
                        except:
                            pass
                
                if not true_emotions:
                    true_emotions = ['neutral']
                
                reasoning = f"This text expresses {', '.join(true_emotions)}."
                selected_examples.append((row['text'], true_emotions, reasoning))
                used_texts.add(row['text'])
                print(f"  Added padding example: {true_emotions}")
    
    print(f"\n✅ Final: {len(selected_examples)} examples ready for few-shot prompting")
    
    return selected_examples[:12]

def get_active_prompt_prediction_all_emotions(
    text: str, 
    subreddit: str, 
    author: str, 
    client, 
    uncertainty_examples: List[Tuple[str, List[str], str]] = None,
    model=None,
    use_self_consistency: bool = False,
    consistency_samples: int = 5
) -> List[str]:
    """
    Get prediction using Active Prompting with TOP 12 examples
    """
    
    # Build few-shot examples
    examples_text = ""
    if uncertainty_examples:
        examples_text = "Learn from these 12 high-uncertainty examples:\n\n"
        
        for i, (ex_text, ex_emotions, ex_reasoning) in enumerate(uncertainty_examples, 1):
            examples_text += f'Example {i}:\n'
            examples_text += f'Text: "{ex_text[:100]}..."\n'
            examples_text += f'Emotions: {ex_emotions}\n'
            examples_text += f'Reasoning: {ex_reasoning}\n\n'
    
    # Main prompt
    prompt = f"""Classify the emotions in this Reddit comment.

Available emotions: {', '.join(ALL_EMOTIONS)}

{examples_text}

Now classify this text. Think step by step:
1. What is the overall tone?
2. Which specific emotions are present?
3. Can multiple emotions coexist?

Text: "{text[:200]}"
Subreddit: {subreddit}
Author: {author}

Answer with a Python list of emotions like ['joy'] or ['anger', 'sadness'] or ['neutral']:"""

    try:
        import config
        
        if use_self_consistency:
            # Multiple samples for self-consistency
            all_predictions = []
            for _ in range(consistency_samples):
                response = client.chat.completions.create(
                    model=config.MODEL_ID if model is None else model,
                    messages=[
                        {"role": "system", "content": "You are an expert emotion classifier. Use the examples to guide your analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=100,
                    timeout=10
                )
                
                predicted = parse_emotion_response(response.choices[0].message.content, ALL_EMOTIONS)
                all_predictions.append(predicted)
            
            # Majority voting
            emotion_counts = Counter()
            for pred in all_predictions:
                for emotion in pred:
                    emotion_counts[emotion] += 1
            
            # Select emotions that appear in majority
            threshold = consistency_samples / 2
            predicted_emotions = [em for em, count in emotion_counts.items() if count > threshold]
            
        else:
            # Single prediction
            response = client.chat.completions.create(
                model=config.MODEL_ID if model is None else model,
                messages=[
                    {"role": "system", "content": "You are an expert emotion classifier. Use the examples carefully."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=100,
                timeout=10
            )
            
            predicted_emotions = parse_emotion_response(response.choices[0].message.content, ALL_EMOTIONS)
        
        if not predicted_emotions:
            predicted_emotions = ['neutral']
        
        return predicted_emotions
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return ['neutral']