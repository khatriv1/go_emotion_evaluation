# dspy_implementation/dspy_goemotions_classifier.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
import pandas as pd
import numpy as np
import json
from typing import Dict, List
import time

# Import from parent directory
from config import OPENAI_API_KEY
from utils.metrics import calculate_agreement_metrics

# Configure DSPy with OpenAI
print(f"Setting up OpenAI API for GoEmotions...")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# DSPy v3 configuration
lm = dspy.LM('openai/gpt-3.5-turbo', api_key=OPENAI_API_KEY, max_tokens=500, temperature=0)
dspy.configure(lm=lm)

# Test the API connection
print("Testing API connection...")
try:
    test_prompt = "Say 'API working' if you can read this"
    test_response = lm(test_prompt)
    print(f"API Test successful: Connection established")
except Exception as e:
    print(f"WARNING: API test failed: {e}")
    print("Please check your API key and internet connection")

# All 28 GoEmotions
EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness',
    'surprise', 'neutral'
]

class GoEmotionsSignature(dspy.Signature):
    """Classify text into GoEmotions categories (multi-label)."""
    
    text = dspy.InputField(desc="The Reddit comment text to classify")
    admiration = dspy.OutputField(desc="1 if expresses admiration, else 0")
    amusement = dspy.OutputField(desc="1 if expresses amusement, else 0")
    anger = dspy.OutputField(desc="1 if expresses anger, else 0")
    annoyance = dspy.OutputField(desc="1 if expresses annoyance, else 0")
    approval = dspy.OutputField(desc="1 if expresses approval, else 0")
    caring = dspy.OutputField(desc="1 if expresses caring, else 0")
    confusion = dspy.OutputField(desc="1 if expresses confusion, else 0")
    curiosity = dspy.OutputField(desc="1 if expresses curiosity, else 0")
    desire = dspy.OutputField(desc="1 if expresses desire, else 0")
    disappointment = dspy.OutputField(desc="1 if expresses disappointment, else 0")
    disapproval = dspy.OutputField(desc="1 if expresses disapproval, else 0")
    disgust = dspy.OutputField(desc="1 if expresses disgust, else 0")
    embarrassment = dspy.OutputField(desc="1 if expresses embarrassment, else 0")
    excitement = dspy.OutputField(desc="1 if expresses excitement, else 0")
    fear = dspy.OutputField(desc="1 if expresses fear, else 0")
    gratitude = dspy.OutputField(desc="1 if expresses gratitude, else 0")
    grief = dspy.OutputField(desc="1 if expresses grief, else 0")
    joy = dspy.OutputField(desc="1 if expresses joy, else 0")
    love = dspy.OutputField(desc="1 if expresses love, else 0")
    nervousness = dspy.OutputField(desc="1 if expresses nervousness, else 0")
    optimism = dspy.OutputField(desc="1 if expresses optimism, else 0")
    pride = dspy.OutputField(desc="1 if expresses pride, else 0")
    realization = dspy.OutputField(desc="1 if expresses realization, else 0")
    relief = dspy.OutputField(desc="1 if expresses relief, else 0")
    remorse = dspy.OutputField(desc="1 if expresses remorse, else 0")
    sadness = dspy.OutputField(desc="1 if expresses sadness, else 0")
    surprise = dspy.OutputField(desc="1 if expresses surprise, else 0")
    neutral = dspy.OutputField(desc="1 if expresses neutral emotion, else 0")

class GoEmotionsClassifier(dspy.Module):
    """DSPy module for GoEmotions multi-label classification"""
    
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(GoEmotionsSignature)
    
    def forward(self, text):
        prediction = self.prog(text=text)
        return prediction

def load_training_data(csv_path: str, sample_size: int = None):
    """Load and prepare GoEmotions training data for DSPy"""
    df = pd.read_csv(csv_path)
    
    if sample_size:
        df = df.head(sample_size)
    
    print(f"Loading {len(df)} training samples...")
    
    examples = []
    for _, row in df.iterrows():
        # Create example with all 28 emotions
        example_dict = {'text': row['text']}
        
        # Add each emotion as binary (1 or 0)
        for emotion in EMOTIONS:
            if emotion in row.index:
                value = row[emotion]
                example_dict[emotion] = str(1 if pd.notna(value) and value == 1 else 0)
            else:
                example_dict[emotion] = '0'
        
        example = dspy.Example(**example_dict).with_inputs('text')
        examples.append(example)
    
    return examples

def goemotions_metric(example, pred, trace=None):
    """Metric for GoEmotions (exact match or high overlap)"""
    
    matches = 0
    total_emotions = 0
    
    for emotion in EMOTIONS:
        try:
            true_val = str(getattr(example, emotion, '0'))
            pred_val = str(getattr(pred, emotion, '0'))
            
            true_val = '1' if true_val == '1' else '0'
            pred_val = '1' if pred_val == '1' else '0'
            
            if true_val == pred_val:
                matches += 1
            
            # Count true positives
            if true_val == '1':
                total_emotions += 1
                
        except:
            pass
    
    # Success if 80% of emotions match (22+ out of 28)
    return matches >= 22

def train_dspy_module(training_examples, sample_size):
    """Train a DSPy module for GoEmotions"""
    print(f"\nTraining DSPy GoEmotions module with {sample_size} samples...")
    print("DSPy is learning emotion patterns from Reddit comments...")
    
    # Use subset of training examples
    train_set = training_examples[:sample_size]
    
    # Initialize module
    goemotions_module = GoEmotionsClassifier()
    
    # Configure optimizer
    from dspy.teleprompt import BootstrapFewShot
    
    teleprompter = BootstrapFewShot(
        metric=goemotions_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
        max_rounds=1
    )
    
    # Compile the module
    print(f"Optimizing module with {len(train_set)} examples...")
    start_time = time.time()
    
    try:
        optimized_module = teleprompter.compile(goemotions_module, trainset=train_set)
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.1f} seconds")
        
        if elapsed < 10:
            print("Note: Training used cached responses for speed")
        else:
            print("Training with fresh API calls completed")
    except Exception as e:
        print(f"Training error: {e}")
        print("Returning base module without optimization")
        return goemotions_module
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save what DSPy learned
    try:
        learned_file = f'results/goemotions_module_{sample_size}_learned.json'
        
        module_str = str(optimized_module)
        
        learned_data = {
            'task': 'GoEmotions',
            'sample_size': sample_size,
            'training_time': elapsed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'module_structure': {
                'type': 'ChainOfThought',
                'signature': 'GoEmotionsSignature',
                'inputs': ['text'],
                'outputs': EMOTIONS,
                'num_emotions': 28,
                'reasoning': 'Chain-of-thought reasoning enabled'
            },
            'optimization': {
                'method': 'BootstrapFewShot',
                'max_demos': 3,
                'max_labeled': 4,
                'optimization_rounds': 1,
                'examples_evaluated': min(sample_size, 10)
            },
            'note': 'DSPy v3 optimizes prompts internally for multi-label emotion classification',
            'module_string_preview': module_str[:500] if len(module_str) > 500 else module_str
        }
        
        with open(learned_file, 'w') as f:
            json.dump(learned_data, f, indent=2)
        
        print(f"Module configuration saved to {learned_file}")
        
    except Exception as e:
        print(f"Note: Could not save module details: {e}")
    
    print(f"Module trained with {sample_size} samples")
    return optimized_module

def test_dspy_module(module, test_csv_path, module_name):
    """Test DSPy module on GoEmotions holdout set"""
    
    print(f"\nTesting {module_name} on holdout set...")
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    results = []
    
    start_time = time.time()
    api_calls = 0
    
    for idx, row in test_df.iterrows():
        if (idx + 1) % 30 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {idx + 1}/{len(test_df)} - Time: {elapsed:.1f}s")
        
        text = row['text']
        
        # Get human labels (multi-label)
        human_emotions = []
        human_binary = {}
        
        for emotion in EMOTIONS:
            if emotion in row.index:
                value = row[emotion]
                is_present = 1 if pd.notna(value) and value == 1 else 0
                human_binary[emotion] = is_present
                if is_present:
                    human_emotions.append(emotion)
            else:
                human_binary[emotion] = 0
        
        if not human_emotions:
            human_emotions = ['neutral']
        
        try:
            # Get DSPy prediction
            prediction = module(text)
            api_calls += 1
            
            # Convert to binary dict
            model_binary = {}
            model_emotions = []
            
            for emotion in EMOTIONS:
                pred_val = getattr(prediction, emotion, '0')
                is_present = 1 if str(pred_val) == '1' else 0
                model_binary[emotion] = is_present
                if is_present:
                    model_emotions.append(emotion)
            
            if not model_emotions:
                model_emotions = ['neutral']
            
        except Exception as e:
            print(f"Error predicting item {idx}: {e}")
            model_binary = {emotion: 0 for emotion in EMOTIONS}
            model_emotions = ['neutral']
        
        # Calculate metrics
        binary_matches = sum(1 for emotion in EMOTIONS 
                           if human_binary[emotion] == model_binary[emotion])
        binary_accuracy = binary_matches / 28
        
        # Exact match (same emotions)
        exact_match = set(human_emotions) == set(model_emotions)
        
        # Append result
        results.append({
            'text_id': idx + 1,
            'text': text[:100],  # First 100 chars
            'human_emotions': ', '.join(human_emotions),
            'model_emotions': ', '.join(model_emotions),
            'exact_match': exact_match,
            'human_binary': str(human_binary),
            'model_binary': str(model_binary),
            'binary_accuracy': binary_accuracy,
            'binary_matches': binary_matches,
            'num_human_emotions': len(human_emotions),
            'num_model_emotions': len(model_emotions),
            'Technique': module_name
        })
        
        # Rate limiting
        time.sleep(0.5)
    
    total_time = time.time() - start_time
    print(f"Testing completed in {total_time:.1f} seconds ({api_calls} API calls)")
    
    return pd.DataFrame(results)

def test_api_connection():
    """Test if API is working properly with GoEmotions"""
    print("\n" + "="*60)
    print("TESTING API CONNECTION FOR GOEMOTIONS")
    print("="*60)
    
    test_text = "I'm so happy about this news, it made my day!"
    print(f"Test text: {test_text}")
    
    try:
        classifier = GoEmotionsClassifier()
        start = time.time()
        result = classifier(test_text)
        elapsed = time.time() - start
        
        print(f"API call took: {elapsed:.2f} seconds")
        print(f"\nDetected emotions:")
        
        detected = []
        for emotion in EMOTIONS:
            value = getattr(result, emotion, '0')
            if str(value) == '1':
                detected.append(emotion)
                print(f"  ✓ {emotion}")
        
        if not detected:
            print("  No emotions detected (or neutral)")
        
        print(f"\nTotal emotions detected: {len(detected)}")
        
        if elapsed < 0.5:
            print("Note: Using cached responses for speed")
        else:
            print("SUCCESS: Fresh API call completed!")
            
    except Exception as e:
        print(f"ERROR: API test failed - {e}")
        print("Check your API key in config.py")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    test_api_connection()