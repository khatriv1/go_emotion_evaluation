# goemotions_evaluation/prompting/active_prompt.py
# FIXED: Added Self-Consistency to final predictions

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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.emotion_rubric import GoEmotionsRubric
from utils.metrics import GoEmotionsMetrics, create_detailed_results_dataframe
import config

class ActivePromptSelector:
    """Implements Active Prompting methodology with MINIMAL API calls"""
    
    def __init__(self, pool_size: int = 20, k_samples: int = 2, consistency_samples: int = 5):
        self.pool_size = pool_size
        self.k_samples = k_samples
        self.consistency_samples = consistency_samples  # NEW: For self-consistency
        self.uncertainty_scores = {}
        
    def estimate_uncertainty_for_emotions(self, texts: List[str], subreddits: List[str], 
                                        authors: List[str], client) -> Dict[str, float]:
        """Estimate uncertainty with MINIMAL API calls (unchanged)"""
        print(f"Estimating uncertainty for {len(texts)} texts")
        
        uncertainty_scores = {}
        
        for i, (text, subreddit, author) in enumerate(zip(texts, subreddits, authors)):
            if (i + 1) % 5 == 0:
                print(f"Processing text {i + 1}/{len(texts)}")
            
            predictions = []
            for sample_idx in range(self.k_samples):
                pred = self._get_single_prediction(text, subreddit, author, client)
                if pred is not None:
                    predictions.append(tuple(sorted(pred)))
                time.sleep(0.05)
            
            if predictions:
                unique_predictions = len(set(predictions))
                disagreement = unique_predictions / len(predictions)
                uncertainty_scores[text] = disagreement
            else:
                uncertainty_scores[text] = 0.0
        
        return uncertainty_scores
    
    def _get_single_prediction(self, text: str, subreddit: str, author: str, client) -> Optional[List[str]]:
        """Get a single emotion prediction with SIMPLE prompt (unchanged)"""
        
        main_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
        
        prompt = f"""What emotions are in this text? Pick from: {', '.join(main_emotions)}

Text: "{text[:100]}"

Answer with list like: ['joy', 'anger'] or ['neutral']:"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "Answer with a Python list of emotions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=20,
                timeout=8
            )
            
            result = response.choices[0].message.content.strip()
            return parse_emotion_response(result, main_emotions)
            
        except Exception as e:
            pass
            
        return None
    
    def select_uncertain_texts(self, uncertainty_scores: Dict[str, float], n_select: int = 3) -> List[str]:
        """Select the most uncertain texts"""
        sorted_texts = sorted(uncertainty_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [text for text, score in sorted_texts[:n_select]]
        
        print(f"Selected {len(selected)} most uncertain texts:")
        for i, text in enumerate(selected):
            score = uncertainty_scores[text]
            print(f"  {i+1}. (uncertainty: {score:.3f}) {text[:40]}...")
        
        return selected

def create_cot_reasoning_emotions(text: str, emotions: List[str], basic_reasoning: str) -> str:
    """Create detailed Chain-of-Thought reasoning for emotion classification examples"""
    
    text_lower = text.lower()
    
    # Emotion keywords
    emotion_indicators = {
        'joy': ['happy', 'excited', 'great', 'awesome', 'love', 'amazing', 'wonderful', 'fantastic'],
        'sadness': ['sad', 'disappointed', 'depressed', 'down', 'upset', 'hurt', 'crying'],
        'anger': ['angry', 'mad', 'furious', 'hate', 'annoyed', 'irritated', 'pissed'],
        'fear': ['scared', 'afraid', 'worried', 'nervous', 'anxious', 'frightened', 'terrified'],
        'surprise': ['wow', 'amazing', 'unexpected', 'shocked', 'surprised', 'incredible'],
        'neutral': ['okay', 'fine', 'normal', 'regular', 'standard', 'typical']
    }
    
    # Find emotion indicators in text
    found_indicators = {}
    for emotion in emotions:
        indicators = [word for word in emotion_indicators.get(emotion, []) if word in text_lower]
        if indicators:
            found_indicators[emotion] = indicators
    
    if len(emotions) == 1 and emotions[0] == 'neutral':
        detailed_reasoning = f"Let me analyze this step by step: 1) I scan the text for emotional language and tone indicators, 2) The text lacks strong positive or negative emotional markers, 3) The overall tone is factual or matter-of-fact without emotional coloring, 4) No specific emotion-triggering words or phrases are present. Therefore, this text is neutral."
    
    elif len(emotions) == 1:
        emotion = emotions[0]
        if found_indicators.get(emotion):
            detailed_reasoning = f"Let me analyze this step by step: 1) I identify emotional language including '{found_indicators[emotion][0]}' which indicates {emotion}, 2) The overall tone and context support this emotional interpretation, 3) The text clearly expresses {emotion} through both explicit words and implicit meaning, 4) No conflicting emotions are strongly present. Therefore, this text expresses {emotion}."
        else:
            detailed_reasoning = f"Let me analyze this step by step: 1) While explicit {emotion} words aren't present, the overall tone conveys {emotion}, 2) The context and implied meaning suggest {emotion}, 3) The emotional undertone is consistent with {emotion} expression, 4) The text's impact and message align with {emotion}. Therefore, this text expresses {emotion}."
    
    else:  # Multiple emotions
        emotion_list = ', '.join(emotions[:-1]) + f' and {emotions[-1]}'
        detailed_reasoning = f"Let me analyze this step by step: 1) I identify multiple emotional indicators in the text, 2) The text contains elements that trigger {emotion_list}, 3) These emotions can coexist as the text has complex emotional content, 4) Each emotion is justified by different parts or interpretations of the text. Therefore, this text expresses multiple emotions: {emotion_list}."
    
    return detailed_reasoning

def parse_emotion_response(response_text: str, valid_emotions: List[str]) -> List[str]:
    """Parse emotion response from LLM output (unchanged)"""
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

def create_active_examples(selected_texts: List[str], ground_truth_data: pd.DataFrame) -> List[Tuple[str, List[str], str]]:
    """Create examples from selected uncertain texts using CORRECT column names (unchanged)"""
    examples = []
    
    all_emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    main_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
    
    for text in selected_texts:
        matching_rows = ground_truth_data[ground_truth_data['text'] == text]
        
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            
            true_emotions = []
            for emotion in main_emotions:
                if emotion in row and row[emotion] == 1:
                    true_emotions.append(emotion)
            
            if not true_emotions:
                true_emotions = ['neutral']
            
            reasoning = f"This text expresses {', '.join(true_emotions)} based on the emotional content."
            examples.append((text, true_emotions, reasoning))
    
    return examples

def get_active_prompt_prediction_all_emotions(text: str, subreddit: str, author: str, client, 
                                            uncertainty_examples: List[Tuple[str, List[str], str]] = None,
                                            model="gpt-3.5-turbo",
                                            use_self_consistency: bool = True,
                                            consistency_samples: int = 5) -> List[str]:
    """Get active prompt prediction with SELF-CONSISTENCY"""
    
    main_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
    
    # Create BETTER active prompt
    prompt = f"""Classify this Reddit comment into emotions: {', '.join(main_emotions)}

"""
    
    # Add examples if available with CoT reasoning
    if uncertainty_examples:
        prompt += "Examples from uncertain cases:\n"
        for ex_text, ex_emotions, ex_reasoning in uncertainty_examples[:2]:
            # Create detailed CoT reasoning
            detailed_reasoning = create_cot_reasoning_emotions(ex_text, ex_emotions, ex_reasoning)
            prompt += f'Text: "{ex_text[:60]}..."\n'
            prompt += f'Reasoning: {detailed_reasoning}\n'
            prompt += f'Emotions: {ex_emotions}\n\n'
    
    prompt += f"""Now classify this text:
Text: "{text[:150]}"

Instructions:
- Look for emotional words and tone
- Most texts have 1-2 emotions maximum  
- Be conservative - only select clear emotions
- If no clear emotion, choose 'neutral'

Answer with Python list like ['joy'] or ['anger', 'sadness'] or ['neutral']:"""

    if not use_self_consistency:
        # Single prediction (original method)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an emotion classifier. Be conservative and selective. Answer with a Python list of emotions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=30,
                temperature=0.1,
                timeout=10
            )
            
            predicted_emotions = parse_emotion_response(response.choices[0].message.content, main_emotions)
            
            if not predicted_emotions:
                predicted_emotions = ['neutral']
            
            return predicted_emotions
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return ['neutral']
    
    else:
        # NEW: SELF-CONSISTENCY - Multiple predictions + most common answer
        all_predictions = []
        
        for i in range(consistency_samples):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an emotion classifier. Be conservative and selective. Answer with a Python list of emotions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=30,
                    temperature=0.7,  # Higher temperature for diversity
                    timeout=10
                )
                
                predicted_emotions = parse_emotion_response(response.choices[0].message.content, main_emotions)
                
                if not predicted_emotions:
                    predicted_emotions = ['neutral']
                
                # Convert to tuple for hashing in Counter
                all_predictions.append(tuple(sorted(predicted_emotions)))
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in self-consistency sample {i+1}: {e}")
                continue
        
        if not all_predictions:
            return ['neutral']
        
        # Take most common emotion combination (SELF-CONSISTENCY)
        counter = Counter(all_predictions)
        most_common_emotions = counter.most_common(1)[0][0]
        
        print(f"Self-consistency for emotions: {all_predictions} → {list(most_common_emotions)}")
        return list(most_common_emotions)

def prepare_active_prompting_data(df: pd.DataFrame, client, n_examples: int = 3) -> List[Tuple[str, List[str], str]]:
    """Prepare active prompting examples with MINIMAL uncertainty estimation (unchanged)"""
    print("Preparing Active Prompting data (MINIMAL VERSION)...")
    
    max_samples = min(len(df), 10)
    sample_df = df.sample(n=max_samples, random_state=42)
    
    texts = sample_df['text'].tolist()
    subreddits = sample_df['subreddit'].tolist()
    authors = sample_df['author'].tolist()
    
    try:
        selector = ActivePromptSelector(k_samples=2)
        
        uncertainty_scores = selector.estimate_uncertainty_for_emotions(texts, subreddits, authors, client)
        selected_texts = selector.select_uncertain_texts(uncertainty_scores, n_examples)
        examples = create_active_examples(selected_texts, sample_df)
        
        print(f"Created {len(examples)} active prompting examples")
        return examples
        
    except Exception as e:
        print(f"Error in uncertainty estimation: {e}")
        print("Using fallback examples...")
        
        main_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
        
        examples = []
        sample_texts = texts[:n_examples] if len(texts) >= n_examples else texts
        
        for text in sample_texts:
            matching_rows = sample_df[sample_df['text'] == text]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                true_emotions = []
                for emotion in main_emotions:
                    if emotion in row and row.get(emotion, 0) == 1:
                        true_emotions.append(emotion)
                
                if not true_emotions:
                    true_emotions = ['neutral']
                
                reasoning = f"This text expresses {', '.join(true_emotions)}."
                examples.append((text, true_emotions, reasoning))
        
        print(f"Created {len(examples)} fallback examples")
        return examples

def evaluate_active_prompt(df: pd.DataFrame, emotions_list: List[str], output_dir: str, 
                         uncertainty_examples: List[Tuple[str, List[str], str]] = None,
                         data_loader=None,
                         use_self_consistency: bool = True,
                         consistency_samples: int = 5) -> Dict:
    """Evaluate using Active Prompting with SELF-CONSISTENCY"""
    
    try:
        from utils.emotion_rubric import GoEmotionsRubric
        from utils.metrics import GoEmotionsMetrics, create_detailed_results_dataframe
        
        main_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
        
        metrics_calculator = GoEmotionsMetrics(main_emotions)
        client = openai.OpenAI(api_key=getattr(config, 'OPENAI_API_KEY', os.getenv('OPENAI_API_KEY')))
        
        predictions = []
        
        print(f"Evaluating on {len(df)} comments using Active Prompting with Self-Consistency: {use_self_consistency}")
        
        for idx, (index, row) in enumerate(df.iterrows(), 1):
            print(f"Processing comment {idx}/{len(df)}")
            
            try:
                comment = row['text']
                true_emotions = row['emotions'] if isinstance(row['emotions'], list) else []
                
                true_emotions = [e for e in true_emotions if e in main_emotions]
                if not true_emotions:
                    true_emotions = ['neutral']
                
                comment_id = f"comment_{index}"
                
                print(f"Comment: {comment[:50]}...")
                
                # Get prediction with self-consistency
                pred_emotions = get_active_prompt_prediction_all_emotions(
                    comment, 
                    row.get('subreddit', ''), 
                    row.get('author', ''), 
                    client,
                    uncertainty_examples,
                    use_self_consistency=use_self_consistency,
                    consistency_samples=consistency_samples
                )
                
                print(f"Human: {true_emotions}")
                print(f"Model: {pred_emotions}")
                
                exact_match = set(true_emotions) == set(pred_emotions)
                print(f"Match: {exact_match}\n")
                
                prediction = {
                    'id': comment_id,
                    'text': comment,
                    'human_emotions': true_emotions,
                    'predicted_emotions': pred_emotions,
                    'exact_match': exact_match
                }
                predictions.append(prediction)
                
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error processing comment: {e}")
                continue
        
        if not predictions:
            print("No successful predictions")
            return None
        
        # Calculate metrics
        y_true = [pred['human_emotions'] for pred in predictions]
        y_pred = [pred['predicted_emotions'] for pred in predictions]
        
        results = metrics_calculator.comprehensive_evaluation(y_true, y_pred)
        
        # Print results
        technique_name = "Active Prompting (Self-Consistency)" if use_self_consistency else "Active Prompting (Single)"
        metrics_calculator.print_results(results, technique_name)
        
        # Save results
        detailed_df = create_detailed_results_dataframe(predictions, main_emotions)
        detailed_path = os.path.join(output_dir, "detailed_results_active_prompt.csv")
        detailed_df.to_csv(detailed_path, index=False)
        
        results['detailed_results'] = detailed_df
        results['predictions'] = predictions
        results['uncertainty_examples'] = uncertainty_examples
        results['self_consistency_used'] = use_self_consistency
        
        print(f"Results saved to {detailed_path}")
        
        return results
        
    except Exception as e:
        print(f"Active Prompting evaluation failed: {e}")
        return None