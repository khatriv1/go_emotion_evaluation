#!/usr/bin/env python3
"""
GoEmotions Multi-Label Emotion Classification: Main Evaluation File
Fixed to work with existing evaluation structure + Comprehensive Analysis
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging

# Disable HTTP request logging from OpenAI
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Keep only your application logs
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Import the existing evaluation functions that your files expect
try:
    from evaluation.evaluate_zero_shot import evaluate_zero_shot
    from evaluation.evaluate_cot import evaluate_cot
    from evaluation.evaluate_few_shot import evaluate_few_shot
    from evaluation.evaluate_active_prompt import evaluate_active_prompt
    from evaluation.evaluate_auto_cot import evaluate_auto_cot
    from evaluation.evaluate_contrastive_cot import evaluate_contrastive_cot
    from evaluation.evaluate_rephrase_respond import evaluate_rephrase_respond
    from evaluation.evaluate_self_consistency import evaluate_self_consistency
    from evaluation.evaluate_take_step_back import evaluate_take_step_back
    
    print("✓ All evaluation modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure all evaluation modules are in the evaluation/ directory")
    sys.exit(1)

# Import config
try:
    import config
    print(f"✓ Config imported - API key configured: {bool(config.OPENAI_API_KEY != 'your-api-key-here')}")
except ImportError:
    print("✗ config.py not found")
    sys.exit(1)

def print_header():
    """Print evaluation suite header"""
    print("=" * 70)
    print("GoEmotions Multi-Label Emotion Classification: Comprehensive Evaluation Suite")
    print("=" * 70)
    print(f"Using GoEmotions dataset: {config.DATA_PATH}")
    print()

def get_user_inputs() -> Tuple[int, List[int]]:
    """Get user inputs for evaluation parameters"""
    
    # Get number of comments to evaluate
    while True:
        try:
            num_input = input("Enter number of comments to evaluate (recommended: 10-50 for testing, press Enter for 10): ").strip()
            if num_input == "":
                num_comments = 10
                print(f"Using default sample size of {num_comments} comments")
                break
            else:
                num_comments = int(num_input)
                if num_comments < 5:
                    print("Warning: Less than 5 comments may not provide reliable results.")
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                break
        except ValueError:
            print("Please enter a valid number or press Enter for default (10 comments).")
    
    # Get techniques to evaluate
    print("\nAvailable prompting techniques:")
    techniques = [
        "Zero-shot",
        "Chain of Thought", 
        "Few-shot",
        "Active Prompting",
        "Auto-CoT",
        "Contrastive CoT",
        "Rephrase and Respond",
        "Self-Consistency",
        "Take a Step Back",
        "All techniques"
    ]
    
    for i, technique in enumerate(techniques, 1):
        print(f"{i}. {technique}")
    
    while True:
        try:
            technique_input = input("\nEnter technique numbers (comma-separated) or 10 for all: ").strip()
            if technique_input == "10":
                selected_techniques = list(range(1, 10))
                break
            else:
                selected_techniques = [int(x.strip()) for x in technique_input.split(",")]
                if all(1 <= t <= 9 for t in selected_techniques):
                    break
                else:
                    print("Please enter numbers between 1-9 or 10 for all techniques.")
        except ValueError:
            print("Please enter valid numbers separated by commas.")
    
    return num_comments, selected_techniques

def setup_results_directory() -> str:
    """Create timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/goemotions_evaluation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def run_evaluations(num_comments: int, selected_techniques: List[int], results_dir: str) -> Dict:
    """Run selected evaluation techniques"""
    
    # Map technique numbers to evaluation functions
    technique_map = {
        1: ("Zero-shot", evaluate_zero_shot),
        2: ("Chain of Thought", evaluate_cot), 
        3: ("Few-shot", evaluate_few_shot),
        4: ("Active Prompting", evaluate_active_prompt),
        5: ("Auto-CoT", evaluate_auto_cot),
        6: ("Contrastive CoT", evaluate_contrastive_cot),
        7: ("Rephrase and Respond", evaluate_rephrase_respond),
        8: ("Self-Consistency", evaluate_self_consistency),
        9: ("Take a Step Back", evaluate_take_step_back)
    }
    
    all_results = {}
    
    for technique_num in selected_techniques:
        technique_name, evaluate_func = technique_map[technique_num]
        
        print("=" * 60)
        print(f"Running {technique_name} evaluation...")
        print("=" * 60)
        
        try:
            # Create technique-specific directory
            technique_dir = os.path.join(results_dir, technique_name.lower().replace(" ", "_"))
            os.makedirs(technique_dir, exist_ok=True)
            
            # Run evaluation using the existing function signature
            start_time = time.time()
            detailed_results, metrics = evaluate_func(
                data_path=config.DATA_PATH,
                api_key=config.OPENAI_API_KEY,
                output_dir=technique_dir,
                limit=num_comments
            )
            
            end_time = time.time()
            
            if metrics and detailed_results:
                # Add metadata
                metrics['technique'] = technique_name
                metrics['evaluation_time'] = end_time - start_time
                metrics['num_comments'] = len(detailed_results)
                
                all_results[technique_name] = {
                    'metrics': metrics,
                    'detailed_results': detailed_results,
                    'evaluation_time': end_time - start_time,
                    'num_comments': len(detailed_results)
                }
                print(f"✓ {technique_name} completed successfully")
                print(f"  - Exact Match Accuracy: {metrics.get('exact_match_accuracy', metrics.get('accuracy', 'N/A')):.1f}%")
                print(f"  - Cohen's Kappa: {metrics.get('kappa', 'N/A'):.3f}")
                print(f"  - Evaluation time: {end_time - start_time:.1f} seconds")
            else:
                print(f"✗ {technique_name} failed: No results returned")
                
        except Exception as e:
            print(f"✗ {technique_name} failed: {str(e)}")
            # Log the error for debugging
            with open(os.path.join(results_dir, f"{technique_name}_error.log"), 'w') as f:
                f.write(f"Error in {technique_name}: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
        
        print()  # Add spacing between evaluations
    
    return all_results

def create_comprehensive_analysis(all_results: Dict, results_dir: str):
    """Create clean comprehensive analysis like Bloom project"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        print("\n" + "=" * 70)
        print("CREATING COMPREHENSIVE ANALYSIS & VISUALIZATIONS")
        print("=" * 70)
        
        if not all_results:
            print("No results to analyze")
            return
        
        # Prepare data
        comparison_data = []
        for technique_name, results in all_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Technique': technique_name,
                'Exact_Match_Accuracy': metrics.get('exact_match_accuracy', metrics.get('accuracy', 0)),
                'Cohens_Kappa': metrics.get('kappa', 0),
                'Krippendorffs_Alpha': metrics.get('alpha', 0),
                'ICC': metrics.get('icc', 0),
                'Hamming_Loss': metrics.get('hamming_loss', 0),
                'Evaluation_Time': results.get('evaluation_time', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # CLEAN CHART DESIGN (like Bloom)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Set clean style
        plt.style.use('default')
        
        # Chart 1: Clean 4-Metric Comparison (like Bloom)
        x = np.arange(len(df))
        width = 0.2
        
        # Use clean colors like Bloom chart
        colors = ['#90EE90', '#87CEEB', '#DDA0DD', '#F08080']  # Light green, sky blue, plum, light coral
        
        # Create clean bars with proper spacing
        bars1 = ax1.bar(x - 1.5*width, df['Exact_Match_Accuracy'], width, 
                        label='Accuracy (%)', color=colors[0], alpha=0.8)
        bars2 = ax1.bar(x - 0.5*width, df['Cohens_Kappa']*100, width, 
                        label="Cohen's κ", color=colors[1], alpha=0.8)
        bars3 = ax1.bar(x + 0.5*width, df['Krippendorffs_Alpha']*100, width, 
                        label="Krippendorff's α", color=colors[2], alpha=0.8)
        bars4 = ax1.bar(x + 1.5*width, df['ICC']*100, width, 
                        label='ICC', color=colors[3], alpha=0.8)
        
        # Clean formatting
        ax1.set_xlabel('Prompting Technique', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax1.set_title('GoEmotions Multi-Label Emotion Classification: 4 Metrics Comparison', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Technique'], rotation=45, ha='right', fontsize=10)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.set_ylim(0, 105)
        
        # Add clean value labels (only on top of bars)
        for i, technique in enumerate(df['Technique']):
            # Accuracy
            height1 = df.iloc[i]['Exact_Match_Accuracy']
            ax1.text(i - 1.5*width, height1 + 1, f'{height1:.1f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Kappa
            height2 = df.iloc[i]['Cohens_Kappa']*100
            ax1.text(i - 0.5*width, height2 + 1, f'{df.iloc[i]["Cohens_Kappa"]:.3f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Alpha  
            height3 = df.iloc[i]['Krippendorffs_Alpha']*100
            ax1.text(i + 0.5*width, height3 + 1, f'{df.iloc[i]["Krippendorffs_Alpha"]:.3f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # ICC
            height4 = df.iloc[i]['ICC']*100
            ax1.text(i + 1.5*width, height4 + 1, f'{df.iloc[i]["ICC"]:.3f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Chart 2: Clean Ranking by Kappa (like Bloom)
        sorted_df = df.sort_values('Cohens_Kappa', ascending=True)
        
        # Create gradient colors (like Bloom)
        colors_gradient = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_df)))
        
        # Highlight best performer in green
        colors_final = []
        for i, kappa in enumerate(sorted_df['Cohens_Kappa']):
            if kappa == sorted_df['Cohens_Kappa'].max():
                colors_final.append('#90EE90')  # Green for best
            else:
                colors_final.append(colors_gradient[i])
        
        bars = ax2.barh(sorted_df['Technique'], sorted_df['Cohens_Kappa'], 
                       color=colors_final, alpha=0.8)
        
        ax2.set_xlabel("Cohen's Kappa (κ)", fontsize=12, fontweight='bold')
        ax2.set_title("Techniques Ranked by Agreement (Cohen's κ)", 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, max(sorted_df['Cohens_Kappa']) * 1.1)
        
        # Add clean value labels
        for bar, kappa in zip(bars, sorted_df['Cohens_Kappa']):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{kappa:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)
        
        # Clean layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        # Save with high quality
        chart_path = os.path.join(results_dir, 'goemotions_metrics_comparison.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create comprehensive report (same as before)
        report_path = os.path.join(results_dir, 'goemotions_comprehensive_report.txt')
        with open(report_path, 'w') as f:
            f.write("=== GoEmotions Multi-Label Emotion Classification: Comprehensive Evaluation Report ===\n\n")
            f.write("Dataset: GoEmotions Reddit Comments with Multi-Label Emotion Annotations\n")
            f.write("Task: Multi-label classification - AI predicts list of emotions for each comment\n")
            f.write("Human Labels: List of emotions (from expert annotations)\n")
            f.write("AI Predictions: List of emotions (from LLM classification)\n")
            f.write(f"Number of techniques evaluated: {len(df)}\n\n")
            
            f.write("=== Overall Metrics Comparison ===\n")
            f.write("Metrics Used:\n")
            f.write("- Exact Match Accuracy: How often AI predicts exact same emotion set as humans\n")
            f.write("- Cohen's Kappa (κ): Agreement beyond chance (0 to 1, higher is better)\n")
            f.write("- Krippendorff's Alpha (α): Reliability measure (0 to 1, higher is better)\n")
            f.write("- ICC: Intraclass Correlation Coefficient (0 to 1, higher is better)\n\n")
            
            # Clean formatted table
            f.write(f"{'Technique':<25} {'Accuracy':<10} {'Kappa':<8} {'Alpha':<8} {'ICC':<8}\n")
            f.write("-" * 65 + "\n")
            for _, row in df.iterrows():
                f.write(f"{row['Technique']:<25} {row['Exact_Match_Accuracy']:<10.1f} "
                       f"{row['Cohens_Kappa']:<8.3f} {row['Krippendorffs_Alpha']:<8.3f} "
                       f"{row['ICC']:<8.3f}\n")
            
            f.write("\n=== Best Performing Techniques by Metric ===\n")
            if len(df) > 0:
                best_exact = df.loc[df['Exact_Match_Accuracy'].idxmax()]
                best_kappa = df.loc[df['Cohens_Kappa'].idxmax()]
                best_alpha = df.loc[df['Krippendorffs_Alpha'].idxmax()]
                best_icc = df.loc[df['ICC'].idxmax()]
                
                f.write(f"Exact Match Accuracy: {best_exact['Technique']} ({best_exact['Exact_Match_Accuracy']:.1f}%)\n")
                f.write(f"Cohen's Kappa (κ): {best_kappa['Technique']} ({best_kappa['Cohens_Kappa']:.3f})\n")
                f.write(f"Krippendorff's Alpha (α): {best_alpha['Technique']} ({best_alpha['Krippendorffs_Alpha']:.3f})\n")
                f.write(f"Intraclass Correlation (ICC): {best_icc['Technique']} ({best_icc['ICC']:.3f})\n")
        
        # Save clean CSV
        csv_path = os.path.join(results_dir, 'goemotions_all_techniques_comparison.csv')
        df.round(3).to_csv(csv_path, index=False)
        
        print("✓ Clean metrics comparison chart created")
        print("✓ Comprehensive report created") 
        print("✓ CSV file created")
        print(f"\n🎉 CLEAN ANALYSIS COMPLETE!")
        print(f"📊 Files created in {results_dir}:")
        print("  📈 goemotions_metrics_comparison.png (CLEAN VERSION)")
        print("  📋 goemotions_comprehensive_report.txt")
        print("  📊 goemotions_all_techniques_comparison.csv")
        
    except ImportError:
        print("⚠️  matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
    except Exception as e:
        print(f"⚠️  Error creating analysis: {e}")
        import traceback
        print(traceback.format_exc())

def generate_summary_report(all_results: Dict, results_dir: str):
    """Generate summary comparison report"""
    if not all_results:
        print("No successful evaluations to compare.")
        return
    
    print("\n" + "=" * 70)
    print("GOEMOTIONS EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Results saved in: {results_dir}")
    print()
    
    # Create comparison dataframe
    comparison_data = []
    
    print(f"{'Technique':<25} {'Exact Match':<12} {'Kappa':<8} {'Alpha':<8} {'ICC':<8} {'Time':<8}")
    print("-" * 75)
    
    for technique_name, results in all_results.items():
        metrics = results['metrics']
        
        comparison_row = {
            'Technique': technique_name,
            'Exact_Match_Accuracy': metrics.get('exact_match_accuracy', metrics.get('accuracy', 0)),
            'Cohens_Kappa': metrics.get('kappa', 0),
            'Krippendorffs_Alpha': metrics.get('alpha', 0),
            'ICC': metrics.get('icc', 0),
            'Hamming_Loss': metrics.get('hamming_loss', 0),
            'Subset_Accuracy': metrics.get('subset_accuracy', 0),
            'Num_Comments': results.get('num_comments', 0),
            'Evaluation_Time': results.get('evaluation_time', 0)
        }
        comparison_data.append(comparison_row)
        
        # Print summary row
        exact_match = metrics.get('exact_match_accuracy', metrics.get('accuracy', 0))
        print(f"{technique_name:<25} {exact_match:<12.1f} "
              f"{metrics.get('kappa', 0):<8.3f} {metrics.get('alpha', 0):<8.3f} "
              f"{metrics.get('icc', 0):<8.3f} {results.get('evaluation_time', 0):<8.1f}")
    
    # Save comparison results
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(results_dir, "technique_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison results saved to: {comparison_path}")
        
        # Show top performers
        if len(comparison_df) > 1:
            print("\nTop Performers:")
            
            # Best by Cohen's Kappa
            best_kappa = comparison_df.loc[comparison_df['Cohens_Kappa'].idxmax()]
            print(f"Best Cohen's Kappa: {best_kappa['Technique']} (κ={best_kappa['Cohens_Kappa']:.3f})")
            
            # Best by Exact Match
            exact_match_col = 'Exact_Match_Accuracy'
            best_exact = comparison_df.loc[comparison_df[exact_match_col].idxmax()]
            print(f"Best Exact Match: {best_exact['Technique']} ({best_exact[exact_match_col]:.1f}%)")
    
    print("\nEvaluation completed successfully!")
    print(f"All results saved in: {results_dir}")
    
    # ADD COMPREHENSIVE ANALYSIS
    create_comprehensive_analysis(all_results, results_dir)

def main():
    """Main evaluation function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GoEmotions Emotion Classification Evaluation')
    parser.add_argument('--num-comments', type=int, default=10, help='Number of comments to evaluate (default: 10)')
    parser.add_argument('--techniques', type=str, help='Comma-separated technique numbers (1-9) or "all"')
    parser.add_argument('--output-dir', type=str, help='Custom output directory')
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Check configuration
    if config.OPENAI_API_KEY == "your-api-key-here" or not config.OPENAI_API_KEY:
        print("ERROR: OpenAI API key not configured!")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='your-actual-api-key-here'")
        print("Or update config.py with your API key")
        sys.exit(1)
    
    # Check if data file exists
    if not os.path.exists(config.DATA_PATH):
        print(f"ERROR: Data file not found: {config.DATA_PATH}")
        print("Please ensure the GoEmotions dataset is available at the specified path")
        sys.exit(1)
    
    # Get evaluation parameters
    if args.techniques is not None:
        # Use command line arguments
        num_comments = args.num_comments
        if args.techniques.lower() == "all":
            selected_techniques = list(range(1, 10))
        else:
            selected_techniques = [int(x.strip()) for x in args.techniques.split(",")]
    else:
        # Interactive mode
        num_comments, selected_techniques = get_user_inputs()
    
    # Setup results directory
    if args.output_dir:
        results_dir = args.output_dir
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = setup_results_directory()
    
    print(f"\nStarting evaluation with {num_comments} comments...")
    print(f"Results will be saved to: {results_dir}")
    print()
    
    # Run evaluations
    all_results = run_evaluations(num_comments, selected_techniques, results_dir)
    
    # Generate summary report (includes comprehensive analysis)
    generate_summary_report(all_results, results_dir)

if __name__ == "__main__":
    main()