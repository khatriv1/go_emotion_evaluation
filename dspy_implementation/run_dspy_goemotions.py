# dspy_implementation/run_dspy_goemotions.py

import sys
import os
import shutil
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CLEAR DSPY CACHE FUNCTION
def clear_dspy_cache():
    """Clear ALL DSPy cache locations"""
    import glob
    
    cache_dirs = [
        os.path.expanduser("~/.cache/dspy"),
        os.path.expanduser("~/.dspy_cache"),
        os.path.expanduser("~/.dspy"),
        os.path.expanduser("~/dspy_cache"),
        ".dspy_cache",
        "dspy_cache",
        "/tmp/dspy*",
        "/var/tmp/dspy*",
        os.path.expanduser("~/.cache/litellm"),
        os.path.expanduser("~/.litellm"),
    ]
    
    for pattern in ["*.dspy", ".dspy*", "dspy_cache*"]:
        cache_dirs.extend(glob.glob(pattern))
    
    cleared_count = 0
    for cache_dir in cache_dirs:
        if '*' in cache_dir:
            for path in glob.glob(cache_dir):
                if os.path.exists(path):
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
                        print(f"✓ Cleared: {path}")
                        cleared_count += 1
                    except Exception as e:
                        print(f"Could not clear {path}: {e}")
        else:
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    print(f"✓ Cleared: {cache_dir}")
                    cleared_count += 1
                except Exception as e:
                    print(f"Could not clear {cache_dir}: {e}")
    
    if cleared_count == 0:
        print("No cache found - starting fresh")
    else:
        print(f"Cleared {cleared_count} cache location(s)")
    
    os.environ['DSPY_CACHEDIR'] = '/tmp/dspy_no_cache_' + str(time.time())
    os.environ['LITELLM_CACHE'] = 'FALSE'
    
    print("="*60)

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from dspy_goemotions_classifier import (
    load_training_data,
    train_dspy_module,
    test_dspy_module,
    test_api_connection,
    EMOTIONS
)

def calculate_goemotions_metrics(results_df):
    """Calculate metrics specific to GoEmotions multi-label classification"""
    
    metrics = {
        'exact_match_accuracy': results_df['exact_match'].mean() * 100,
        'binary_accuracy': results_df['binary_accuracy'].mean() * 100,
        'avg_emotions_per_text': results_df['num_human_emotions'].mean(),
        'avg_predicted_emotions': results_df['num_model_emotions'].mean()
    }
    
    # Per-emotion accuracy
    emotion_accuracies = {}
    for emotion in EMOTIONS:
        correct = 0
        total = len(results_df)
        
        for _, row in results_df.iterrows():
            human = eval(row['human_binary'])[emotion]
            model = eval(row['model_binary'])[emotion]
            if human == model:
                correct += 1
        
        emotion_accuracies[emotion] = (correct / total) * 100
    
    metrics['per_emotion_accuracy'] = emotion_accuracies
    
    # Find best and worst emotions
    sorted_emotions = sorted(emotion_accuracies.items(), key=lambda x: x[1], reverse=True)
    metrics['best_emotions'] = sorted_emotions[:5]
    metrics['worst_emotions'] = sorted_emotions[-5:]
    
    return metrics

def create_comparison_charts(summary_results, output_dir='results'):
    """Create comparison charts for GoEmotions DSPy results"""
    
    # Prepare data
    techniques = list(summary_results.keys())
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
    
    # Top chart - Main Metrics Comparison
    metrics_names = ['Exact Match (%)', 'Binary Accuracy (%)', 'Avg Emotions/Text']
    x = np.arange(len(techniques))
    width = 0.25
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, metric in enumerate(metrics_names):
        if metric == 'Exact Match (%)':
            values = [summary_results[t]['exact_match_accuracy'] for t in techniques]
        elif metric == 'Binary Accuracy (%)':
            values = [summary_results[t]['binary_accuracy'] for t in techniques]
        else:  # Avg Emotions
            values = [summary_results[t]['avg_predicted_emotions'] for t in techniques]
        
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('DSPy Training Size', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('GoEmotions Multi-Label Classification: DSPy Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.replace('DSPy_GoEmotions_', '') + ' samples' for t in techniques])
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Middle chart - Binary Accuracy Trend
    sizes = [100, 200, 300]
    binary_acc = [summary_results[f'DSPy_GoEmotions_{s}']['binary_accuracy'] for s in sizes]
    exact_match = [summary_results[f'DSPy_GoEmotions_{s}']['exact_match_accuracy'] for s in sizes]
    
    ax2.plot(sizes, binary_acc, 'o-', label='Binary Accuracy (%)', linewidth=2, markersize=8, color='#2ecc71')
    ax2.plot(sizes, exact_match, 's-', label='Exact Match (%)', linewidth=2, markersize=8, color='#3498db')
    
    for i, size in enumerate(sizes):
        ax2.text(size, binary_acc[i] + 1, f'{binary_acc[i]:.1f}', ha='center', fontsize=9)
        ax2.text(size, exact_match[i] - 2, f'{exact_match[i]:.1f}', ha='center', fontsize=9)
    
    ax2.set_xlabel('Training Sample Size', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('DSPy GoEmotions: Performance vs Training Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(sizes)
    ax2.set_xlim(80, 320)
    ax2.set_ylim(0, 100)
    
    # Bottom chart - Top 10 Emotion Accuracies (for best model)
    best_model = f'DSPy_GoEmotions_300'
    emotion_accs = summary_results[best_model]['per_emotion_accuracy']
    sorted_emotions = sorted(emotion_accs.items(), key=lambda x: x[1], reverse=True)[:10]
    
    emotions = [e[0] for e in sorted_emotions]
    accuracies = [e[1] for e in sorted_emotions]
    
    bars = ax3.barh(range(len(emotions)), accuracies, color='#9b59b6', alpha=0.8)
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax3.text(acc + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}%', va='center', fontsize=9)
    
    ax3.set_yticks(range(len(emotions)))
    ax3.set_yticklabels(emotions, fontsize=10)
    ax3.set_xlabel('Binary Accuracy (%)', fontsize=12)
    ax3.set_title(f'Top 10 Emotions by Accuracy (300 samples)', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 100)
    ax3.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(output_dir, 'dspy_goemotions_comparison_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison chart saved to {chart_path}")
    plt.close()

def run_all_experiments(clear_cache=False):
    """Run DSPy experiments for GoEmotions with 100, 200, 300 samples"""
    
    # CLEAR CACHE IF REQUESTED
    if clear_cache:
        print("="*60)
        print("CLEARING DSPY CACHE")
        print("="*60)
        clear_dspy_cache()
    
    print("="*60)
    print("DSPY EXPERIMENTS FOR GOEMOTIONS")
    print("="*60)
    print("Task: Multi-label emotion classification (28 emotions)")
    print("Model: GPT-3.5-turbo")
    print("Training sizes: 100, 200, 300 samples")
    print("Test set: 150 holdout samples")
    print("="*60)
    
    # Test API connection
    test_api_connection()
    
    print("\nDSPy will learn emotion patterns from Reddit comments")
    print("NO emotion_rubric.py needed - DSPy learns from data!")
    if not clear_cache:
        print("Using cached responses for speed (add --clear-cache for fresh calls)")
    print("="*60)
    
    # Paths
    training_path = 'dspy_300.csv'  # 300-sample training file
    test_path = 'holdout_150.csv'   # 150-sample test file
    
    # Check if files exist
    if not os.path.exists(training_path):
        print(f"ERROR: {training_path} not found!")
        return
    
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found!")
        return
    
    # Load all 300 training samples
    print(f"\nLoading training data from {training_path}...")
    all_training_examples = load_training_data(training_path, sample_size=300)
    print(f"✓ Loaded {len(all_training_examples)} training examples")
    
    # Sample sizes: 100, 200, 300
    sample_sizes = [100, 200, 300]
    
    all_results = []
    summary_results = {}
    
    for size in sample_sizes:
        print("\n" + "="*60)
        print(f"EXPERIMENT: DSPy GoEmotions Module with {size} training samples")
        print("="*60)
        
        # Train module
        module = train_dspy_module(all_training_examples, size)
        
        # Test module
        module_name = f'DSPy_GoEmotions_{size}'
        results_df = test_dspy_module(module, test_path, module_name)
        
        # Save individual results
        os.makedirs('results', exist_ok=True)
        results_file = f'results/dspy_goemotions_results_{size}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
        
        # Append to all results
        all_results.append(results_df)
        
        # Calculate metrics
        metrics = calculate_goemotions_metrics(results_df)
        
        # Store summary
        summary_results[module_name] = metrics
        
        # Print results
        print(f"\n{module_name} Results:")
        print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.1f}%")
        print(f"  Binary Accuracy: {metrics['binary_accuracy']:.1f}%")
        print(f"  Avg emotions per text: {metrics['avg_emotions_per_text']:.2f}")
        print(f"  Avg predicted emotions: {metrics['avg_predicted_emotions']:.2f}")
        print(f"  Best emotion: {metrics['best_emotions'][0][0]} ({metrics['best_emotions'][0][1]:.1f}%)")
        print(f"  Worst emotion: {metrics['worst_emotions'][0][0]} ({metrics['worst_emotions'][0][1]:.1f}%)")
    
    # Combine all results
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv('results/dspy_goemotions_all_results.csv', index=False)
    
    # Save summary JSON
    summary_with_metadata = {
        'task': 'GoEmotions Multi-Label Classification',
        'model': 'GPT-3.5-turbo',
        'num_emotions': 28,
        'training_sizes': sample_sizes,
        'test_size': 150,
        'timestamp': datetime.now().isoformat(),
        'results': summary_results
    }
    
    with open('results/dspy_goemotions_summary.json', 'w') as f:
        json.dump(summary_with_metadata, f, indent=2, default=str)
    print("\nSummary saved to results/dspy_goemotions_summary.json")
    
    # Generate comparison charts
    print("\n" + "="*60)
    print("Generating comparison charts...")
    create_comparison_charts(summary_results)
    
    # Print final summary table
    print("\n" + "="*60)
    print("DSPY GOEMOTIONS RESULTS SUMMARY")
    print("="*60)
    print(f"{'Module':<25} {'Exact Match':<15} {'Binary Acc':<12}")
    print("-"*52)
    for module, metrics in summary_results.items():
        name = module.replace('DSPy_GoEmotions_', '') + ' samples'
        print(f"{name:<25} {metrics['exact_match_accuracy']:>7.1f}%      {metrics['binary_accuracy']:>7.1f}%")
    
    # Show improvement
    print("\n" + "="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)
    improvement = summary_results['DSPy_GoEmotions_300']['binary_accuracy'] - summary_results['DSPy_GoEmotions_100']['binary_accuracy']
    print(f"Binary Accuracy Improvement (100→300): {improvement:+.1f}%")
    
    # Best performing emotions across all models
    print("\nConsistently Best Emotions:")
    all_best = {}
    for module in summary_results.values():
        for emotion, acc in module['best_emotions']:
            if emotion not in all_best:
                all_best[emotion] = []
            all_best[emotion].append(acc)
    
    for emotion, accs in sorted(all_best.items(), key=lambda x: np.mean(x[1]), reverse=True)[:5]:
        print(f"  {emotion}: {np.mean(accs):.1f}%")
    
    print("\n" + "="*60)
    print("DSPY GoEmotions experiments complete!")
    print("\nFiles saved in results/:")
    print("  - dspy_goemotions_results_100.csv, _200.csv, _300.csv")
    print("  - dspy_goemotions_all_results.csv")
    print("  - dspy_goemotions_summary.json")
    print("  - goemotions_module_100_learned.json, _200, _300")
    print("  - dspy_goemotions_comparison_chart.png")
    print("="*60)

if __name__ == "__main__":
    # Check for command line argument
    import sys
    clear_cache = '--clear-cache' in sys.argv
    run_all_experiments(clear_cache=clear_cache)