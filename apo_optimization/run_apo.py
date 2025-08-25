# goemotions_evaluation/apo_optimization/run_apo.py
"""
GoEmotions Rubric APO Runner
Runs APO with 4 different evaluation sizes: 10, 20, 50, 100
Generates 5 variations per run = 20 total variations
"""

import sys
import os
import time
from datetime import datetime
import json

# Add paths
sys.path.append('..')
sys.path.append('.')

from goemotions_apo_system import GoEmotionsRubricAPO

# Configuration
from config import OPENAI_API_KEY
DATA_PATH = "../data/holdout_150.csv"
EVALUATION_SAMPLE_SIZES = [10, 20, 50, 100]
RESULTS_DIR = "results"

def run_automated_goemotions_apo():
    """Run GoEmotions Rubric APO for all evaluation sizes"""
    
    print("="*70)
    print("STARTING AUTOMATED GOEMOTIONS RUBRIC APO")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: GPT-3.5-turbo")
    print(f"Evaluation sizes: {EVALUATION_SAMPLE_SIZES}")
    print(f"Variations per run: 5")
    print(f"Total variations: 20")
    print("="*70)
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = {}
    
    # Run for each evaluation size
    for run_num, eval_size in enumerate(EVALUATION_SAMPLE_SIZES, 1):
        print("\n" + "="*70)
        print(f"RUN {run_num}/4: EVALUATION_SIZE = {eval_size}")
        print("="*70)
        
        try:
            # Initialize APO
            apo = GoEmotionsRubricAPO(
                api_key=OPENAI_API_KEY,
                data_path=DATA_PATH,
                validation_sample_size=100,
                evaluation_sample_size=eval_size
            )
            
            # Run optimization
            best_candidate = apo.optimize_rubrics()
            
            # Prepare output data with ALL variations
            output_data = {
                'evaluation_sample_size': eval_size,
                'baseline_rubric': apo.baseline_rubric,
                'best_rubric': best_candidate.rubric_definitions,
                'all_variations': apo.all_variations,  # All 5 variations
                'technique_scores': best_candidate.performance_scores,
                'average_score': best_candidate.average_score,
                'detailed_metrics': best_candidate.detailed_metrics
            }
            
            # Save results
            results_file = os.path.join(RESULTS_DIR, f"optimized_rubrics_eval{eval_size}.json")
            with open(results_file, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Results saved to: {results_file}")
            
            # Generate report
            report = generate_report(output_data, eval_size)
            report_file = os.path.join(RESULTS_DIR, f"rubric_comparison_eval{eval_size}.txt")
            with open(report_file, "w") as f:
                f.write(report)
            
            all_results[eval_size] = {
                'average_score': best_candidate.average_score,
                'best_technique': max(best_candidate.performance_scores, 
                                     key=best_candidate.performance_scores.get)
            }
            
        except Exception as e:
            print(f"Error in run {run_num}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*70)
    print("GOEMOTIONS RUBRIC APO COMPLETE!")
    print("="*70)
    print("\nRESULTS SUMMARY:")
    print("Eval Size | Avg Score | Best Technique")
    print("-"*40)
    for eval_size, data in all_results.items():
        print(f"   {eval_size:3d}    |   {data['average_score']:.3f}   | {data['best_technique']}")
    
    print("\nFILES CREATED:")
    for eval_size in EVALUATION_SAMPLE_SIZES:
        print(f"  - optimized_rubrics_eval{eval_size}.json")

def generate_report(output_data: dict, eval_size: int) -> str:
    """Generate comparison report"""
    
    report = f"GOEMOTIONS RUBRIC OPTIMIZATION REPORT - {eval_size} SAMPLES\n"
    report += "="*60 + "\n\n"
    
    report += "PERFORMANCE BY TECHNIQUE:\n"
    report += "-"*40 + "\n"
    for technique, score in output_data['technique_scores'].items():
        report += f"• {technique:20s}: {score:.3f}\n"
    
    report += f"\nAVERAGE SCORE: {output_data['average_score']:.3f}\n"
    
    report += "\nSAMPLE EMOTION FROM BEST RUBRIC:\n"
    report += "-"*40 + "\n"
    # Show first 3 emotions as example
    for i, (emotion, definition) in enumerate(list(output_data['best_rubric'].items())[:3]):
        report += f"• {emotion}: {definition}\n"
    
    return report

if __name__ == "__main__":
    print("GoEmotions Rubric APO System")
    print("This will generate 20 total variations (5 per run × 4 runs)")
    
    response = input("\nReady to start? (y/n): ").lower().strip()
    if response == 'y':
        run_automated_goemotions_apo()
    else:
        print("APO cancelled.")