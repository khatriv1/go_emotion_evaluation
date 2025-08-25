# goemotions_evaluation/evaluation/evaluate_active_prompt.py

import sys
import os
import pandas as pd
import openai
import time
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.active_prompt import get_active_prompt_prediction_all_emotions, prepare_active_prompting_data
from utils.data_loader import load_and_preprocess_goemotions_data, get_comment_emotions, filter_annotated_comments
from utils.metrics import calculate_agreement_metrics, plot_emotion_performance, print_detailed_results

def evaluate_active_prompt(
    data_path: str, 
    api_key: str, 
    output_dir: str = "results/active_prompt", 
    limit: int = None
):
    """
    Evaluate Active Prompting with 56→12 selection strategy
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    df = load_and_preprocess_goemotions_data(data_path)
    df = filter_annotated_comments(df, min_emotions=1, annotator='expert')
    
    if limit:
        df = df.head(limit)
    print(f"\nTotal test samples: {len(df)}")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Emotions list
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral'
    ]
    
    # Load pool (90 samples)
    import config
    pool_path = config.ACTIVE_POOL_PATH  # active_prompting_pool_90.csv
    
    if os.path.exists(pool_path):
        print(f"\n✅ Loading pool from: {pool_path}")
        pool_df = pd.read_csv(pool_path)
        print(f"Pool size: {len(pool_df)} samples")
        eval_df = df  # Test on full dataset
    else:
        print(f"\n⚠️ Pool not found, using first 90 from test set")
        pool_df = df.head(90)
        eval_df = df.iloc[90:]
    
    # PHASE 1: Generate 56 candidates → Select TOP 12
    print("\n" + "="*60)
    print("PREPARING ACTIVE PROMPTING EXAMPLES")
    print("="*60)
    
    uncertainty_examples = prepare_active_prompting_data(
        pool_df, 
        client, 
        n_examples=12
    )
    
    print(f"\n✅ Prepared {len(uncertainty_examples)} examples")
    
    # Display selected examples
    print("\nSelected TOP 12 Examples:")
    for i, (text, emotions_list, reasoning) in enumerate(uncertainty_examples[:5], 1):
        print(f"  {i}. {text[:60]}... → {emotions_list}")
    print("  ...")
    
    # Save examples
    examples_data = []
    for text, emotions_list, reasoning in uncertainty_examples:
        examples_data.append({
            'text': text[:200],
            'emotions': ', '.join(emotions_list),
            'reasoning': reasoning
        })
    examples_df = pd.DataFrame(examples_data)
    examples_df.to_csv(f"{output_dir}/top12_examples.csv", index=False)
    print(f"Examples saved to {output_dir}/top12_examples.csv")
    
    # PHASE 2: Evaluation
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    human_labels = {}
    model_labels = {}
    detailed_results = []
    
    total = len(eval_df)
    print(f"\nEvaluating on {total} test samples...")
    
    for seq, (_, row) in enumerate(eval_df.iterrows(), start=1):
        if seq % 10 == 0:
            print(f"Progress: {seq}/{total}")
        
        comment_id = str(row.get('id', seq))
        
        try:
            # Get ground truth
            human_emotions = get_comment_emotions(row, annotator='expert')
            human_labels[comment_id] = human_emotions
            
            # Get model prediction with TOP 12 examples
            model_emotions = get_active_prompt_prediction_all_emotions(
                text=row['text'],
                subreddit=row.get('subreddit', ''),
                author=row.get('author', ''),
                client=client,
                uncertainty_examples=uncertainty_examples,
                use_self_consistency=False
            )
            model_labels[comment_id] = model_emotions
            
            # Store result
            detailed_results.append({
                'comment_id': comment_id,
                'text': row['text'][:100],
                'human_emotions': ', '.join(human_emotions),
                'model_emotions': ', '.join(model_emotions),
                'exact_match': set(human_emotions) == set(model_emotions)
            })
            
            if seq <= 5:
                print(f"  Sample {seq}:")
                print(f"    Human: {human_emotions}")
                print(f"    Model: {model_emotions}")
                print(f"    Match: {set(human_emotions) == set(model_emotions)}")
            
        except Exception as e:
            print(f"Error processing {comment_id}: {e}")
            continue
        
        time.sleep(0.5)
    
    # Calculate metrics
    metrics = calculate_agreement_metrics(human_labels, model_labels, emotions)
    
    # Save results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    
    # Save metrics
    metrics_summary = {
        'technique': 'Active Prompting (56→12 Strategy)',
        'pool_size': len(pool_df),
        'candidates_generated': 56,
        'examples_used': 12,
        'test_samples': total,
        'exact_match_accuracy': metrics['accuracy'],
        'kappa': metrics['kappa'],
        'alpha': metrics['alpha'],
        'icc': metrics['icc'],
        'hamming_loss': metrics.get('hamming_loss', 0),
        'subset_accuracy': metrics.get('subset_accuracy', 0)
    }
    
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Strategy: 56 candidates → TOP 12 selection")
    print(f"Pool: {len(pool_df)} samples")
    print(f"Test: {total} samples")
    print(f"Accuracy: {metrics['accuracy']:.1f}%")
    print(f"Kappa: {metrics['kappa']:.3f}")
    print(f"Results saved to: {output_dir}/")
    
    return detailed_results, metrics

if __name__ == "__main__":
    import config
    
    print("\nStarting Active Prompting Evaluation")
    print("Strategy: Generate 56 candidates → Select TOP 12")
    print(f"Model: {config.MODEL_ID}")
    
    try:
        results, metrics = evaluate_active_prompt(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=None
        )
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()