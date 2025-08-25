# goemotions_evaluation/apo_optimization/goemotions_apo_system.py
"""
GoEmotions Rubric APO System
Optimizes emotion definitions in emotion_rubric.py while keeping prompting strategies at baseline
"""

import pandas as pd
import numpy as np
import openai
from typing import Dict, List, Tuple, Any, Optional
import json
import time
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import sys
import os
import shutil
from collections import Counter

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

@dataclass
class RubricCandidate:
    """Represents a candidate rubric with its performance metrics"""
    rubric_definitions: Dict[str, str]
    performance_scores: Dict[str, float]  
    average_score: float
    detailed_metrics: Dict[str, Dict[str, float]]

class GoEmotionsRubricAPO:
    """
    Automatic Prompt Optimization for GoEmotions Classification
    Optimizes EMOTION DEFINITIONS in emotion_rubric.py
    """
    
    def __init__(self, api_key: str, data_path: str = "../data/holdout_150.csv",
                 validation_sample_size: int = 100,
                 evaluation_sample_size: int = 50):
        """Initialize GoEmotions Rubric APO system"""
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set OpenAI API key
        openai.api_key = api_key
        self.client = self._get_openai_client()
        
        self.data_path = data_path
        self.validation_sample_size = validation_sample_size
        self.evaluation_sample_size = evaluation_sample_size
        
        # All 28 emotions
        self.emotions = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness',
            'surprise', 'neutral'
        ]
        
        # Directory setup
        self.prompting_dir = "../prompting"  
        self.utils_dir = "../utils"  
        self.apo_copies_dir = "./apo_copies"  
        
        # Create APO copies directory
        self._setup_apo_copies()
        
        # Get baseline rubric definitions from emotion_rubric.py
        self.baseline_rubric = self._get_baseline_rubric()
        
        # Techniques to test
        self.techniques = ['zero_shot', 'few_shot', 'auto_cot', 'self_consistency', 'active_prompting']
        
        # Load and prepare data
        self.logger.info("Loading and preparing data...")
        
        # Check for existing splits
        holdout_file = '../data/holdout_150.csv'
        apo_file = '../data/apo_training_100.csv'
        
        if os.path.exists(holdout_file) and os.path.exists(apo_file):
            self.logger.info("✓ Found existing data splits, using them...")
            self.holdout_set = pd.read_csv(holdout_file)
            self.apo_set = pd.read_csv(apo_file)
            self.logger.info(f"  Loaded {len(self.holdout_set)} holdout samples")
            self.logger.info(f"  Loaded {len(self.apo_set)} APO samples")
        else:
            self.logger.error("Data splits not found! Please ensure holdout_150.csv and apo_training_100.csv exist")
            raise FileNotFoundError("Required data files not found")
        
        # Check Active Prompting pool
        self.active_pool_path = '../data/active_prompting_pool_20.csv'
        if os.path.exists(self.active_pool_path):
            self.logger.info(f"✓ Found Active Prompting pool")
            self.active_pool_df = pd.read_csv(self.active_pool_path)
        else:
            self.logger.warning(f"⚠ Active Prompting pool not found")
            self.active_pool_df = None
        
        # Prepare validation data based on evaluation_sample_size
        if evaluation_sample_size <= len(self.apo_set):
            self.validation_data = self._prepare_validation_data(self.apo_set.iloc[:evaluation_sample_size])
        else:
            self.validation_data = self._prepare_validation_data(self.apo_set)
        
        self.logger.info(f"Using {len(self.validation_data)} samples for validation")
        
        # Store original emotion_rubric.py content
        self.original_rubric_content = self._store_original_rubric()
        
        # Cache for Active Prompting
        self._active_prompting_cache = None
    
    def _get_openai_client(self):
        """Initialize OpenAI client"""
        from openai import OpenAI
        return OpenAI(api_key=openai.api_key)
    
    def _setup_apo_copies(self):
        """Create APO copies directory with prompting and utils folders"""
        
        # Create apo_copies directory
        if os.path.exists(self.apo_copies_dir):
            shutil.rmtree(self.apo_copies_dir)
        os.makedirs(self.apo_copies_dir)
        
        # Copy prompting folder
        src_prompting = self.prompting_dir
        dst_prompting = os.path.join(self.apo_copies_dir, "prompting")
        if os.path.exists(src_prompting):
            shutil.copytree(src_prompting, dst_prompting)
            self.logger.info(f"Copied prompting folder to {dst_prompting}")
        
        # Copy utils folder (contains emotion_rubric.py)
        src_utils = self.utils_dir
        dst_utils = os.path.join(self.apo_copies_dir, "utils")
        if os.path.exists(src_utils):
            shutil.copytree(src_utils, dst_utils)
            self.logger.info(f"Copied utils folder to {dst_utils}")
        
        self.logger.info(f"APO copies created in {self.apo_copies_dir}")
    
    def _store_original_rubric(self):
        """Store original emotion_rubric.py content"""
        rubric_file = os.path.join(self.apo_copies_dir, "utils", "emotion_rubric.py")
        if os.path.exists(rubric_file):
            with open(rubric_file, 'r') as f:
                return f.read()
        return None
    
    def _get_baseline_rubric(self) -> Dict[str, str]:
        """Get baseline emotion definitions from emotion_rubric.py"""
        return {
            'admiration': 'Finding something impressive or worthy of respect',
            'amusement': 'Finding something funny or being entertained',
            'anger': 'A strong feeling of displeasure or antagonism',
            'annoyance': 'Mild anger, irritation',
            'approval': 'Having or expressing a favorable opinion',
            'caring': 'Displaying kindness and concern for others',
            'confusion': 'Lack of understanding, uncertainty',
            'curiosity': 'A strong desire to know or learn something',
            'desire': 'A strong feeling of wanting something or wishing for something to happen',
            'disappointment': "Sadness or displeasure caused by the nonfulfillment of one's hopes or expectations",
            'disapproval': 'Having or expressing an unfavorable opinion',
            'disgust': 'Revulsion or strong disapproval aroused by something unpleasant or offensive',
            'embarrassment': 'Self-consciousness, shame, or awkwardness',
            'excitement': 'Feeling of great enthusiasm and eagerness',
            'fear': 'Being afraid or worried',
            'gratitude': 'A feeling of thankfulness and appreciation',
            'grief': "Intense sorrow, especially caused by someone's death",
            'joy': 'A feeling of pleasure and happiness',
            'love': 'A strong positive emotion of regard and affection',
            'nervousness': 'Apprehension, worry, anxiety',
            'optimism': 'Hopefulness and confidence about the future or the success of something',
            'pride': "Pleasure or satisfaction due to one's own achievements or the achievements of those with whom one is closely associated",
            'realization': 'Becoming aware of something',
            'relief': 'Reassurance and relaxation following release from anxiety or distress',
            'remorse': 'Regret or guilty feeling',
            'sadness': 'Emotional pain, sorrow',
            'surprise': 'Feeling astonished, startled by something unexpected',
            'neutral': 'Not expressing strong emotion; balanced'
        }
    
    def _prepare_validation_data(self, df: pd.DataFrame) -> List[Dict]:
        """Prepare validation data from dataframe"""
        validation_data = []
        
        for _, row in df.iterrows():
            text = row['text']
            
            # Get emotions for this text
            emotions_present = []
            for emotion in self.emotions:
                if emotion in row and row[emotion] == 1:
                    emotions_present.append(emotion)
            
            if not emotions_present:
                emotions_present = ['neutral']
            
            validation_data.append({
                'text': text,
                'ground_truth': emotions_present
            })
        
        return validation_data
    
    def _generate_rubric_variations(self) -> List[Dict[str, str]]:
        """Generate 5 variations of emotion definitions using GPT-3.5"""
        
        prompt = f"""You are an expert in emotion classification. I need you to create 5 variations of emotion definitions for the GoEmotions dataset.

Current definitions:
{json.dumps(self.baseline_rubric, indent=2)}

Create 5 different variations of these definitions. Each variation should:
1. Be clear and precise for an LLM to understand
2. Maintain the core meaning of each emotion
3. Use different wording/phrasing than the original
4. Be concise (1-2 sentences max)

Return EXACTLY 5 variations as a JSON array. Each variation should contain all 28 emotions.

Format:
[
  {{
    "admiration": "new definition here",
    "amusement": "new definition here",
    ... (all 28 emotions)
  }},
  ... (5 variations total)
]

Return ONLY the JSON array, no other text."""

        try:
            self.logger.info("Generating emotion rubric variations using GPT-3.5...")
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at creating clear emotion definitions. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.7
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON
            try:
                if '```json' in result.lower():
                    result = result.split('```json')[1].split('```')[0]
                elif '```' in result:
                    result = result.split('```')[1].split('```')[0]
                
                variations = json.loads(result)
                
                if isinstance(variations, list) and len(variations) >= 5:
                    self.logger.info(f"✓ Successfully generated {len(variations)} variations!")
                    return variations[:5]
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {e}")
            
            # Fallback: create simple variations
            self.logger.warning("Using fallback variations")
            return self._create_fallback_variations()
            
        except Exception as e:
            self.logger.error(f"Error generating variations: {e}")
            return self._create_fallback_variations()
    
    def _create_fallback_variations(self) -> List[Dict[str, str]]:
        """Create fallback variations if GPT fails"""
        variations = []
        
        for i in range(5):
            variation = {}
            for emotion, definition in self.baseline_rubric.items():
                # Create slight variations
                if i == 0:
                    variation[emotion] = definition.replace("feeling", "emotion")
                elif i == 1:
                    variation[emotion] = definition.replace("strong", "intense")
                elif i == 2:
                    variation[emotion] = definition.lower()
                elif i == 3:
                    variation[emotion] = definition.replace("or", "and")
                else:
                    variation[emotion] = definition
            variations.append(variation)
        
        return variations
    
    def _update_rubric_file(self, rubric_definitions: Dict[str, str]):
        """Update emotion_rubric.py with new definitions"""
        
        rubric_file = os.path.join(self.apo_copies_dir, "utils", "emotion_rubric.py")
        
        # Read current content
        with open(rubric_file, 'r') as f:
            content = f.read()
        
        # Create new emotion_definitions dictionary string
        new_definitions = "self.emotion_definitions = {\n"
        for emotion, definition in rubric_definitions.items():
            # Escape quotes in definition
            escaped_def = definition.replace("'", "\\'")
            new_definitions += f"            '{emotion}': '{escaped_def}',\n"
        new_definitions = new_definitions.rstrip(',\n') + "\n        }"
        
        # Replace the emotion_definitions in the file
        pattern = r'self\.emotion_definitions = \{[^}]+\}'
        content = re.sub(pattern, new_definitions, content, flags=re.DOTALL)
        
        # Write back
        with open(rubric_file, 'w') as f:
            f.write(content)
        
        self.logger.debug(f"Updated emotion definitions in {rubric_file}")
    
    def _restore_original_rubric(self):
        """Restore original emotion_rubric.py"""
        rubric_file = os.path.join(self.apo_copies_dir, "utils", "emotion_rubric.py")
        if self.original_rubric_content:
            with open(rubric_file, 'w') as f:
                f.write(self.original_rubric_content)
    
    def evaluate_rubric_with_all_techniques(self, rubric_definitions: Dict[str, str], sample_data: List[Dict]) -> Dict[str, Dict]:
        """Test a rubric with ALL prompting techniques"""
        
        # Update the rubric file
        self._update_rubric_file(rubric_definitions)
        
        # Clear module cache
        modules_to_remove = []
        for mod_name in list(sys.modules.keys()):
            if 'prompting' in mod_name or 'utils' in mod_name or 'emotion_rubric' in mod_name:
                modules_to_remove.append(mod_name)
        
        for mod_name in modules_to_remove:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        
        # Ensure apo_copies is first in path
        copies_path = self.apo_copies_dir
        if copies_path in sys.path:
            sys.path.remove(copies_path)
        sys.path.insert(0, copies_path)
        
        results = {}
        
        # Test each technique
        for technique in self.techniques:
            self.logger.info(f"  Testing {technique}...")
            
            try:
                # Import the appropriate function
                if technique == 'zero_shot':
                    from prompting.zero_shot import get_zero_shot_prediction_all_emotions
                    get_prediction_func = get_zero_shot_prediction_all_emotions
                    
                elif technique == 'few_shot':
                    from prompting.few_shot import get_few_shot_prediction_all_emotions
                    get_prediction_func = get_few_shot_prediction_all_emotions
                    
                elif technique == 'auto_cot':
                    from prompting.auto_cot import get_auto_cot_prediction_all_emotions
                    get_prediction_func = get_auto_cot_prediction_all_emotions
                    
                elif technique == 'self_consistency':
                    from prompting.self_consistency import get_self_consistency_prediction_all_emotions
                    get_prediction_func = get_self_consistency_prediction_all_emotions
                    
                elif technique == 'active_prompting':
                    from prompting.active_prompt import get_active_prompt_prediction_all_emotions
                    get_prediction_func = get_active_prompt_prediction_all_emotions
                
                # Evaluate
                metrics = self._evaluate_technique(technique, get_prediction_func, sample_data)
                results[technique] = metrics
                self.logger.info(f"    {technique} accuracy: {metrics['accuracy']:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {technique}: {e}")
                results[technique] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        # Restore original
        self._restore_original_rubric()
        
        # Clear cache again
        for mod_name in modules_to_remove:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        
        return results
    
    def _evaluate_technique(self, technique: str, get_prediction_func, sample_data: List[Dict]) -> Dict[str, float]:
        """Evaluate a single technique"""
        
        # Special handling for active prompting
        uncertainty_data = None
        if technique == 'active_prompting' and self.active_pool_df is not None:
            if self._active_prompting_cache is None:
                self.logger.info("    Preparing Active Prompting data...")
                from prompting.active_prompt import prepare_active_prompting_data
                self._active_prompting_cache = prepare_active_prompting_data(self.active_pool_df, self.client)
            uncertainty_data = self._active_prompting_cache
        
        # Test each sample
        correct = 0
        total = 0
        
        for data_point in sample_data:
            text = data_point['text']
            ground_truth = set(data_point['ground_truth'])
            
            try:
                # Get prediction
                if technique == 'active_prompting' and uncertainty_data:
                    predictions = get_prediction_func(
                        text, '', '', self.client, 
                        uncertainty_examples=uncertainty_data
                    )
                else:
                    predictions = get_prediction_func(text, '', '', self.client)
                
                # Convert to set for comparison
                predictions = set(predictions) if isinstance(predictions, list) else {predictions}
                
                # Check exact match
                if predictions == ground_truth:
                    correct += 1
                total += 1
                
            except Exception as e:
                self.logger.error(f"Prediction error: {e}")
                total += 1
            
            time.sleep(0.1)  # Rate limiting
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': accuracy,  # Simplified for now
            'recall': accuracy,
            'f1': accuracy
        }
    
    def optimize_rubrics(self) -> RubricCandidate:
        """Main optimization function - test baseline + 5 variations"""
        
        self.logger.info("="*60)
        self.logger.info("STARTING GOEMOTIONS RUBRIC APO OPTIMIZATION")
        self.logger.info(f"Testing 6 rubrics (baseline + 5 variations)")
        self.logger.info(f"Evaluation sample size: {self.evaluation_sample_size}")
        self.logger.info("="*60)
        
        # Generate 5 variations
        self.logger.info("\nGenerating 5 emotion rubric variations...")
        variations = self._generate_rubric_variations()
        
        all_candidates = []
        
        # Test baseline
        self.logger.info("\n" + "="*40)
        self.logger.info("Testing BASELINE rubric...")
        baseline_results = self.evaluate_rubric_with_all_techniques(self.baseline_rubric, self.validation_data)
        
        baseline_candidate = RubricCandidate(
            rubric_definitions=self.baseline_rubric,
            performance_scores={tech: metrics['accuracy'] for tech, metrics in baseline_results.items()},
            average_score=np.mean([metrics['accuracy'] for metrics in baseline_results.values()]),
            detailed_metrics=baseline_results
        )
        all_candidates.append(baseline_candidate)
        self.logger.info(f"Baseline average: {baseline_candidate.average_score:.3f}")
        
        # Test each variation
        for i, variation in enumerate(variations, 1):
            self.logger.info("\n" + "="*40)
            self.logger.info(f"Testing VARIATION {i}/5...")
            
            variation_results = self.evaluate_rubric_with_all_techniques(variation, self.validation_data)
            
            variation_candidate = RubricCandidate(
                rubric_definitions=variation,
                performance_scores={tech: metrics['accuracy'] for tech, metrics in variation_results.items()},
                average_score=np.mean([metrics['accuracy'] for metrics in variation_results.values()]),
                detailed_metrics=variation_results
            )
            all_candidates.append(variation_candidate)
            self.logger.info(f"Variation {i} average: {variation_candidate.average_score:.3f}")
        
        # Find best
        best_candidate = max(all_candidates, key=lambda x: x.average_score)
        
        # Store all variations for saving
        self.all_variations = variations
        
        self.logger.info("\n" + "="*60)
        self.logger.info("OPTIMIZATION COMPLETE!")
        self.logger.info(f"Best average score: {best_candidate.average_score:.3f}")
        self.logger.info("="*60)
        
        return best_candidate