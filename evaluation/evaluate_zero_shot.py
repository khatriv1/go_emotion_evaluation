# goemotions_evaluation/evaluation/evaluate_zero_shot.py

import sys
import os
import pandas as pd
import openai
import time
from typing import Dict, List

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.zero_shot import get_zero_shot_prediction_all_emotions
from utils.data_loader import load_and_preprocess_goemotions_data, get_comment_emotions, filter_annotated_comments
from utils.metrics import calculate_agreement_metrics, plot_emotion_performance, print_detailed_results

def evaluate_zero_shot(data_path: str, api_key: str, output_dir: str = "results/zero_shot", limit: int = None):
    """Evaluate Zero-shot prompting technique on GoEmotions dataset."""
    
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
        print(f"\nEvaluating on {len(df)} comments")
        
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
    
    # Store results
    human_labels = {}  # comment_id -> list of emotions
    model_labels = {}  # comment_id -> list of emotions
    detailed_results = []
    
    # Process each comment
    total = len(df)
    for seq, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"\nProcessing comment {seq}/{total}")
        print(f"Comment: {row['text'][:100]}...")
        
        comment_id = str(row['comment_id'])
        
        try:
            # Get human annotations (expert ground truth)
            human_emotions = get_comment_emotions(row, annotator='expert')
            human_labels[comment_id] = human_emotions
            
            # Get model predictions
            model_emotions = get_zero_shot_prediction_all_emotions(
                text=row['text'],
                subreddit=row.get('subreddit', ''),
                author=row.get('author', ''),
                client=client
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
    plot_emotion_performance(
        metrics, 
        emotions, 
        'Zero-shot',
        f"{output_dir}/zero_shot_performance.png"
    )
    
    # Print results
    print_detailed_results(metrics, emotions, 'Zero-shot')
    
    # Save detailed results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f"\nDetailed results saved to {output_dir}/detailed_results.csv")
    
    # Save metrics summary
    metrics_summary = {
        'technique': 'Zero-shot',
        'exact_match_accuracy': metrics['accuracy'],
        'kappa': metrics['kappa'],
        'alpha': metrics['alpha'],
        'icc': metrics['icc'],
        'hamming_loss': metrics.get('hamming_loss', 0),
        'subset_accuracy': metrics.get('subset_accuracy', 0)
    }
    
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    
    return detailed_results, metrics

if __name__ == "__main__":
    # Import config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Run evaluation
    try:
        print("\nStarting Zero-shot evaluation on GoEmotions dataset...")
        print(f"Using data file: {config.DATA_PATH}")
        
        results, metrics = evaluate_zero_shot(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=10  # Set to small number for testing
        )
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())