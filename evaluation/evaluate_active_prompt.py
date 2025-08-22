# goemotions_evaluation/evaluation/evaluate_active_prompt.py

import sys
import os
import pandas as pd
import openai
import time
from typing import Dict, List, Tuple

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.active_prompt import get_active_prompt_prediction_all_emotions, prepare_active_prompting_data
from utils.data_loader import load_and_preprocess_goemotions_data, get_comment_emotions, filter_annotated_comments
from utils.metrics import calculate_agreement_metrics, plot_emotion_performance, print_detailed_results

def evaluate_active_prompt(data_path: str, api_key: str, output_dir: str = "results/active_prompt", limit: int = None):
    """Evaluate Active Prompting technique on GoEmotions dataset (12 examples, no self-consistency)."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    try:
        df = load_and_preprocess_goemotions_data(data_path)
        if df.empty:
            raise Exception("No valid examples found in the data file")
        
        # Filter to comments with emotion annotations
        df = filter_annotated_comments(df, min_emotions=1, annotator='expert')
        
        if limit:
            df = df.head(limit)
        print(f"\nTotal samples for evaluation: {len(df)}")
        
    except Exception as e:
        raise Exception(f"Error loading or processing data: {str(e)}")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # GoEmotions categories
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # ========== PREPARE ACTIVE PROMPTING DATA ==========
    print("\n" + "="*60)
    print("PHASE 1: PREPARING ACTIVE PROMPTING EXAMPLES")
    print("="*60)
    
    # Use first 20 samples for uncertainty pool
    pool_size = min(20, len(df))
    pool_df = df.head(pool_size)
    eval_df = df.iloc[pool_size:]  # Remaining samples for evaluation
    
    print(f"Pool size: {len(pool_df)} samples")
    print(f"Evaluation size: {len(eval_df)} samples")
    
    # Get 12 examples (2 per main emotion)
    uncertainty_examples = prepare_active_prompting_data(pool_df, client, n_examples=12)
    
    if not uncertainty_examples:
        print("Warning: No uncertainty data prepared. Using empty examples.")
        uncertainty_examples = []
    else:
        print(f"\n✓ Prepared {len(uncertainty_examples)} examples for Active Prompting")
        
        # Display the examples
        print("\nActive Prompting Examples:")
        for i, (text, emotions_list, reasoning) in enumerate(uncertainty_examples[:6], 1):
            print(f"  Example {i}: {text[:60]}... → {emotions_list}")
    
    # ========== EVALUATION PHASE ==========
    print("\n" + "="*60)
    print("PHASE 2: EVALUATION WITH ACTIVE PROMPTING")
    print("="*60)
    
    # Store results
    human_labels = {}  # comment_id -> list of emotions
    model_labels = {}  # comment_id -> list of emotions
    detailed_results = []
    
    total = len(eval_df)
    print(f"\nEvaluating on {total} test comments")
    
    for seq, (_, row) in enumerate(eval_df.iterrows(), start=1):
        print(f"\nProcessing comment {seq}/{total}")
        print(f"Comment: {row['text'][:100]}...")
        
        comment_id = str(row['comment_id'])
        
        try:
            # Get human annotations (expert ground truth)
            human_emotions = get_comment_emotions(row, annotator='expert')
            human_labels[comment_id] = human_emotions
            
            # Get model predictions with Active Prompting (WITHOUT self-consistency)
            model_emotions = get_active_prompt_prediction_all_emotions(
                text=row['text'],
                subreddit=row.get('subreddit', ''),
                author=row.get('author', ''),
                client=client,
                uncertainty_examples=uncertainty_examples,
                use_self_consistency=False  # Explicitly NO self-consistency
            )
            model_labels[comment_id] = model_emotions
            
            # Store detailed result
            detailed_results.append({
                'comment_id': comment_id,
                'text': row['text'],
                'subreddit': row.get('subreddit', ''),
                'author': row.get('author', ''),
                'human_emotions': ', '.join(human_emotions),
                'model_emotions': ', '.join(model_emotions),
                'exact_match': set(human_emotions) == set(model_emotions)
            })
            
            print(f"Human: {human_emotions}")
            print(f"Model: {model_emotions}")
            print(f"Match: {set(human_emotions) == set(model_emotions)}")
            
        except Exception as e:
            print(f"Error processing comment {comment_id}: {str(e)}")
            continue
        
        time.sleep(1)  # Rate limiting
    
    if not human_labels:
        raise Exception("No valid predictions were generated")
    
    # Calculate metrics
    metrics = calculate_agreement_metrics(human_labels, model_labels, emotions)
    
    # Create visualization
    try:
        plot_emotion_performance(
            metrics, 
            emotions, 
            'Active Prompting (12 Examples)',
            f"{output_dir}/active_prompt_performance.png"
        )
        print(f"Performance plot saved to {output_dir}/active_prompt_performance.png")
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    # Print results
    print_detailed_results(metrics, emotions, 'Active Prompting (12 Examples)')
    
    # Save detailed results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f"Detailed results saved to {output_dir}/detailed_results.csv")
    
    # Save the uncertainty examples used
    if uncertainty_examples:
        examples_data = []
        for text, emotions_list, reasoning in uncertainty_examples:
            examples_data.append({
                'text': text[:200],
                'emotions': ', '.join(emotions_list),
                'reasoning': reasoning
            })
        examples_df = pd.DataFrame(examples_data)
        examples_df.to_csv(f"{output_dir}/uncertainty_examples.csv", index=False)
        print(f"Uncertainty examples saved to {output_dir}/uncertainty_examples.csv")
    
    # Save metrics summary
    metrics_summary = {
        'technique': 'Active Prompting (12 Examples)',
        'exact_match_accuracy': metrics['accuracy'],
        'kappa': metrics['kappa'],
        'alpha': metrics['alpha'],
        'icc': metrics['icc'],
        'hamming_loss': metrics.get('hamming_loss', 0),
        'subset_accuracy': metrics.get('subset_accuracy', 0),
        'num_uncertainty_examples': len(uncertainty_examples),
        'pool_size': len(pool_df),
        'test_samples': total,
        'self_consistency': False
    }
    
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    print(f"Metrics summary saved to {output_dir}/metrics_summary.csv")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Pool size: {len(pool_df)} samples")
    print(f"Test size: {total} samples")
    print(f"Total examples prepared: {len(uncertainty_examples)}")
    print(f"Accuracy: {metrics['accuracy']:.1f}%")
    
    return detailed_results, metrics

if __name__ == "__main__":
    # Import config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Run evaluation
    try:
        print("\nStarting Active Prompting evaluation on GoEmotions dataset...")
        print(f"Using data file: {config.DATA_PATH}")
        print(f"Using model: {config.MODEL_ID}")
        
        results, metrics = evaluate_active_prompt(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=None  # Set to None for full evaluation or a number for testing
        )
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())