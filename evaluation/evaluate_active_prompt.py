# goemotions_evaluation/evaluation/evaluate_active_prompt.py
# MINIMAL VERSION: Reduced parameters to match active_prompt.py

import sys
import os
import pandas as pd
import openai
import time
from typing import Dict, List

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.active_prompt import get_active_prompt_prediction_all_emotions, prepare_active_prompting_data
from utils.data_loader import load_and_preprocess_goemotions_data, get_comment_emotions, filter_annotated_comments
from utils.metrics import calculate_agreement_metrics, plot_emotion_performance, print_detailed_results

def evaluate_active_prompt(data_path: str, api_key: str, output_dir: str = "results/active_prompt", limit: int = None):
    """Evaluate Active Prompting technique on GoEmotions dataset with MINIMAL parameters."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    try:
        df = load_and_preprocess_goemotions_data(data_path)
        if df.empty:
            raise Exception("No valid examples found in the data file")
        
        # Filter to comments with emotion annotations
        df = filter_annotated_comments(df, min_emotions=1, annotator='expert')
        
        print(f"Loaded {len(df)} total comments with emotions")
        
        if limit:
            # REDUCED uncertainty pool size
            uncertainty_size = min(20, max(10, len(df) // 4))  # REDUCED from min(1000, max(200, len(df) // 3))
            eval_size = min(limit, 15)  # Cap evaluation size
            
            # Sample for uncertainty estimation (smaller pool)
            uncertainty_df = df.sample(n=uncertainty_size, random_state=42)
            
            # Sample for evaluation
            remaining_df = df.drop(uncertainty_df.index) if len(df) > uncertainty_size + eval_size else df
            if len(remaining_df) >= eval_size:
                eval_df = remaining_df.sample(n=eval_size, random_state=43)
            else:
                eval_df = df.sample(n=eval_size, random_state=43)
                
            print(f"Using {len(uncertainty_df)} comments for uncertainty estimation (REDUCED)")
            print(f"Evaluating on {len(eval_df)} comments")
        else:
            # Default small sizes for testing
            uncertainty_size = min(20, len(df))
            eval_size = min(10, len(df))
            
            uncertainty_df = df.sample(n=uncertainty_size, random_state=42)
            eval_df = df.sample(n=eval_size, random_state=43)
            
            print(f"Using {len(uncertainty_df)} comments for uncertainty estimation")
            print(f"Evaluating on {len(eval_df)} comments")
        
    except Exception as e:
        raise Exception(f"Error loading or processing data: {str(e)}")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Use MAIN emotions only (reduced from 28 to 6)
    emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
    
    print("\n" + "="*60)
    print("ACTIVE PROMPTING: UNCERTAINTY ESTIMATION PHASE (MINIMAL)")
    print("="*60)
    print("Optimized parameters:")
    print(f"  • Pool size: {len(uncertainty_df)} comments")
    print(f"  • k_samples: 2 per comment")
    print(f"  • Emotions: 6 main emotions (vs 28)")
    print(f"  • Examples: 3 total")
    print(f"  • Expected API calls: ~{len(uncertainty_df) * 2} for uncertainty")
    print("="*60)
    
    # STAGE 1-3: Uncertainty Estimation, Selection, and Annotation
    try:
        uncertainty_examples = prepare_active_prompting_data(uncertainty_df, client, n_examples=3)  # REDUCED from 8 to 3
        print(f"Active Prompting preparation completed with {len(uncertainty_examples)} examples")
        
        # Show what was created
        if uncertainty_examples:
            print(f"  Created examples from uncertain texts:")
            for i, (text, emotions_list, reasoning) in enumerate(uncertainty_examples[:3]):
                print(f"    {i+1}. {emotions_list} - {text[:40]}...")
        
    except Exception as e:
        print(f"Uncertainty estimation failed: {e}")
        print("Using fallback method...")
        
        # Fallback: random selection with simple examples
        sample_comments = uncertainty_df.sample(n=min(3, len(uncertainty_df)), random_state=42)
        uncertainty_examples = []
        
        for _, row in sample_comments.iterrows():
            # Get ground truth emotions (simplified to main emotions)
            try:
                true_emotions = get_comment_emotions(row, annotator='expert')
                # Filter to main emotions only
                main_true_emotions = [e for e in true_emotions if e in emotions]
                if not main_true_emotions:
                    main_true_emotions = ['neutral']
            except:
                main_true_emotions = ['neutral']
            
            # Create simple reasoning
            reasoning = f"This text expresses {', '.join(main_true_emotions)} based on the language used."
            uncertainty_examples.append((row['text'], main_true_emotions, reasoning))
        
        print(f"Fallback preparation completed with {len(uncertainty_examples)} examples")
    
    print("\n" + "="*60)
    print("ACTIVE PROMPTING: EVALUATION PHASE (MINIMAL)")
    print("="*60)
    
    # Store results
    human_labels = {}
    model_labels = {}
    detailed_results = []
    
    # STAGE 4: Inference with Selected Examples
    total = len(eval_df)
    successful_predictions = 0
    
    for seq, (_, row) in enumerate(eval_df.iterrows(), start=1):
        print(f"\nProcessing comment {seq}/{total}")
        print(f"Comment: {row['text'][:80]}...")
        
        comment_id = str(row.get('comment_id', seq))
        
        try:
            # Get human annotations (filter to main emotions)
            try:
                human_emotions = get_comment_emotions(row, annotator='expert')
                # Filter to main emotions only
                human_emotions = [e for e in human_emotions if e in emotions]
                if not human_emotions:
                    human_emotions = ['neutral']
            except:
                human_emotions = ['neutral']
                
            human_labels[comment_id] = human_emotions
            
            # Get model predictions with Active Prompting
            model_emotions = get_active_prompt_prediction_all_emotions(
                text=row['text'],
                subreddit=row.get('subreddit', ''),
                author=row.get('author', ''),
                client=client,
                uncertainty_examples=uncertainty_examples
            )
            model_labels[comment_id] = model_emotions
            successful_predictions += 1
            
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
        
        time.sleep(0.3)  # Faster rate limiting
    
    if not human_labels:
        raise Exception("No valid predictions were generated")
    
    print(f"\nSuccessfully processed {successful_predictions}/{total} comments")
    
    # Calculate metrics
    metrics = calculate_agreement_metrics(human_labels, model_labels, emotions)
    
    # Create visualization
    try:
        plot_emotion_performance(
            metrics, 
            emotions, 
            'Active Prompting (MINIMAL)',
            f"{output_dir}/active_prompt_performance.png"
        )
        print(f"📊 Performance plot saved to {output_dir}/active_prompt_performance.png")
    except Exception as e:
        print(f"⚠️ Could not create plot: {e}")
    
    # Print results
    print_detailed_results(metrics, emotions, 'Active Prompting (MINIMAL)')
    
    # Save detailed results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f"Detailed results saved to {output_dir}/detailed_results.csv")
    
    # Save metrics summary
    metrics_summary = {
        'technique': 'Active Prompting (MINIMAL)',
        'exact_match_accuracy': metrics['accuracy'],
        'kappa': metrics['kappa'],
        'alpha': metrics['alpha'],
        'icc': metrics['icc'],
        'hamming_loss': metrics.get('hamming_loss', 0),
        'subset_accuracy': metrics.get('subset_accuracy', 0),
        'uncertainty_examples_used': len(uncertainty_examples),
        'emotions_evaluated': len(emotions),
        'uncertainty_pool_size': len(uncertainty_df),
        'successful_predictions': successful_predictions
    }
    
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    print(f"📊 Metrics summary saved to {output_dir}/metrics_summary.csv")
    
    return detailed_results, metrics

if __name__ == "__main__":
    # Import config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Run evaluation
    try:
        print("Starting MINIMAL Active Prompting evaluation on GoEmotions dataset...")
        print("Optimized for fast testing:")
        print("   • Small uncertainty pool (20 comments)")
        print("   • Fewer samples (k=2 vs k=10)")
        print("   • Main emotions only (6 vs 28)")
        print("   • Fewer examples (3 vs 8)")
        print("   • Simplified prompts")
        print(f"📁 Using data file: {config.DATA_PATH}")
        
        # Ask user for limit
        try:
            limit_input = input("\n📝 Enter number of comments to evaluate (or press Enter for 10): ").strip()
            limit = int(limit_input) if limit_input else 10
            
            if limit > 15:
                print(f"Reducing limit from {limit} to 15 for faster testing")
                limit = 15
                
        except ValueError:
            limit = 10
        
        print(f"\n🎯 Configuration:")
        print(f"   • Uncertainty pool: ~20 comments")
        print(f"   • Evaluation set: {limit} comments")
        print(f"   • Expected API calls: ~40 for uncertainty + ~{limit} for evaluation")
        print(f"   • Estimated time: 2-3 minutes")
        
        # Confirm before starting
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes', '']:
            print("Cancelled")
            exit()
        
        start_time = time.time()
        
        results, metrics = evaluate_active_prompt(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=limit
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("MINIMAL ACTIVE PROMPTING EVALUATION COMPLETED!")
        print("="*60)
        print(f" Total time: {duration:.1f} seconds")
        print(f" Accuracy: {metrics['accuracy']:.1%}")
        print(f" Cohen's Kappa: {metrics['kappa']:.3f}")
        print(f" Results saved in: results/active_prompt/")
        
    except KeyboardInterrupt:
        print("\n\n Evaluation stopped by user")
        
    except Exception as e:
        print(f"\n Error during evaluation: {str(e)}")
        print("\n Troubleshooting tips:")
        print("   • Check your OpenAI API key")
        print("   • Ensure data file exists")
        print("   • Try with smaller limit (e.g., 5)")
        
        import traceback
        print("\n Full error traceback:")
        print(traceback.format_exc())