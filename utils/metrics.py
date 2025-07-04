# goemotions_evaluation/utils/metrics.py

"""
Evaluation metrics for GoEmotions multi-label emotion classification.
Using the 4 specified metrics: Accuracy, Cohen's Kappa, Krippendorff's Alpha, ICC
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy import stats
import krippendorff
from typing import Dict, List, Tuple

def calculate_agreement_metrics(human_labels: Dict[str, List[str]], 
                               model_labels: Dict[str, List[str]], 
                               categories: List[str]) -> Dict[str, float]:
    """
    Calculate the 4 specified metrics for GoEmotions multi-label emotion classification.
    
    Args:
        human_labels: Dict mapping comment_id to list of human-assigned emotions
        model_labels: Dict mapping comment_id to list of model-assigned emotions  
        categories: List of all possible emotion categories (28 emotions)
    
    Returns:
        Dictionary containing the 4 metrics
    """
    # Convert to binary matrices for calculations
    comment_ids = list(human_labels.keys())
    n_comments = len(comment_ids)
    n_categories = len(categories)
    
    human_matrix = np.zeros((n_comments, n_categories), dtype=int)
    model_matrix = np.zeros((n_comments, n_categories), dtype=int)
    
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    for i, comment_id in enumerate(comment_ids):
        # Human labels
        for emotion in human_labels[comment_id]:
            if emotion in category_to_idx:
                human_matrix[i, category_to_idx[emotion]] = 1
        
        # Model labels  
        for emotion in model_labels[comment_id]:
            if emotion in category_to_idx:
                model_matrix[i, category_to_idx[emotion]] = 1
    
    # Flatten for overall metrics
    human_flat = human_matrix.flatten()
    model_flat = model_matrix.flatten()
    
    # 1. ACCURACY - Exact match accuracy (all emotions must match exactly)
    exact_matches = np.all(human_matrix == model_matrix, axis=1)
    accuracy = np.mean(exact_matches) * 100
    
    # 2. COHEN'S KAPPA (κ) - Agreement beyond chance
    kappa = cohen_kappa_score(human_flat, model_flat)
    
    # 3. KRIPPENDORFF'S ALPHA (α) - Reliability measure
    # Prepare data for Krippendorff's alpha
    data = np.array([human_flat, model_flat])
    alpha = krippendorff.alpha(data, level_of_measurement='nominal')
    
    # 4. INTRACLASS CORRELATION (ICC) - Correlation between scores
    # Using ICC(2,1) - two-way random effects, single measurement, absolute agreement
    # For multi-label, we calculate correlation between flattened binary matrices
    if np.var(human_flat) > 0 and np.var(model_flat) > 0:
        correlation = np.corrcoef(human_flat, model_flat)[0, 1]
        icc = correlation
    else:
        icc = 0.0
    
    # Per-emotion metrics
    emotion_metrics = {}
    for i, emotion in enumerate(categories):
        human_emotion = human_matrix[:, i]
        model_emotion = model_matrix[:, i]
        
        if len(np.unique(human_emotion)) > 1 and len(np.unique(model_emotion)) > 1:
            emotion_accuracy = accuracy_score(human_emotion, model_emotion) * 100
            emotion_kappa = cohen_kappa_score(human_emotion, model_emotion)
            
            # Emotion-specific Krippendorff's alpha
            emotion_data = np.array([human_emotion, model_emotion])
            emotion_alpha = krippendorff.alpha(emotion_data, level_of_measurement='nominal')
            
            # Emotion-specific correlation
            emotion_corr = np.corrcoef(human_emotion, model_emotion)[0, 1]
            
        else:
            emotion_accuracy = np.mean(human_emotion == model_emotion) * 100
            emotion_kappa = 0.0
            emotion_alpha = 0.0
            emotion_corr = 0.0
        
        # Calculate support (number of positive examples)
        support = np.sum(human_emotion)
        
        emotion_metrics[emotion] = {
            'accuracy': emotion_accuracy,
            'kappa': emotion_kappa if not np.isnan(emotion_kappa) else 0.0,
            'alpha': emotion_alpha if not np.isnan(emotion_alpha) else 0.0,
            'correlation': emotion_corr if not np.isnan(emotion_corr) else 0.0,
            'support': support
        }
    
    return {
        'accuracy': accuracy,
        'kappa': kappa if not np.isnan(kappa) else 0.0,
        'alpha': alpha if not np.isnan(alpha) else 0.0,
        'icc': icc if not np.isnan(icc) else 0.0,
        'emotion_metrics': emotion_metrics
    }


def plot_emotion_performance(metrics: Dict[str, float], 
                           emotions: List[str], 
                           technique_name: str, 
                           save_path: str = None):
    """
    Create visualization of per-emotion performance using the 4 metrics.
    """
    emotion_metrics = metrics['emotion_metrics']
    
    # Prepare data for plotting - only show top 16 emotions by support
    emotions_with_support = [(emotion, emotion_metrics[emotion]['support']) for emotion in emotions]
    emotions_with_support.sort(key=lambda x: x[1], reverse=True)
    top_emotions = [emotion for emotion, _ in emotions_with_support[:16]]
    
    # Prepare data for plotting
    metric_names = ['Accuracy', 'Cohen\'s κ', 'Krippendorff\'s α', 'Correlation']
    metric_keys = ['accuracy', 'kappa', 'alpha', 'correlation']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
        scores = []
        for emotion in top_emotions:
            if metric_key == 'accuracy':
                scores.append(emotion_metrics[emotion][metric_key] / 100)  # Convert to 0-1 scale
            else:
                scores.append(emotion_metrics[emotion][metric_key])
        
        bars = axes[i].bar(range(len(top_emotions)), scores, alpha=0.7)
        axes[i].set_title(f'{metric_name} by Emotion (Top 16)', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric_name)
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].set_xticks(range(len(top_emotions)))
        axes[i].set_xticklabels(top_emotions, rotation=45, ha='right')
        
        # Add value labels on bars
        for j, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            if metric_key == 'accuracy':
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{score*100:.1f}%', ha='center', va='bottom', fontsize=8)
            else:
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{score:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'GoEmotions Multi-Label Performance: {technique_name}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def print_detailed_results(metrics: Dict[str, float], 
                         emotions: List[str], 
                         technique_name: str):
    """
    Print detailed results summary with the 4 metrics.
    """
    print(f"\n=== {technique_name} Results ===")
    print(f"Overall Multi-Label Metrics:")
    print(f"  Exact Match Accuracy: {metrics['accuracy']:.1f}%")
    print(f"  Cohen's Kappa (κ): {metrics['kappa']:.3f}")
    print(f"  Krippendorff's Alpha (α): {metrics['alpha']:.3f}")
    print(f"  Intraclass Correlation (ICC): {metrics['icc']:.3f}")
    
    print("\nPer-Emotion Results (sorted by support):")
    emotion_metrics = metrics['emotion_metrics']
    
    # Sort emotions by support for better readability
    emotions_with_support = [(emotion, emotion_metrics[emotion]['support']) for emotion in emotions]
    emotions_with_support.sort(key=lambda x: x[1], reverse=True)
    
    for emotion, support in emotions_with_support:
        em = emotion_metrics[emotion]
        print(f"  {emotion:15s}: Acc={em['accuracy']:.1f}%, "
              f"κ={em['kappa']:.3f}, "
              f"α={em['alpha']:.3f}, "
              f"Corr={em['correlation']:.3f}, "
              f"Supp={support}")
    
    # Multi-label specific analysis
    print(f"\nMulti-Label Performance Interpretation:")
    
    exact_match = metrics['accuracy']
    print(f"  Exact Match ({exact_match:.1f}%): ", end="")
    if exact_match > 50:
        print("Excellent - Most emotion sets predicted perfectly")
    elif exact_match > 30:
        print("Good - Many emotion sets predicted correctly")  
    elif exact_match > 15:
        print("Fair - Some emotion sets predicted correctly")
    else:
        print("Poor - Few emotion sets predicted correctly")
    
    kappa = metrics['kappa']
    print(f"  Overall Agreement (κ={kappa:.3f}): ", end="")
    if kappa > 0.8:
        print("Almost Perfect Agreement")
    elif kappa > 0.6:
        print("Substantial Agreement")  
    elif kappa > 0.4:
        print("Moderate Agreement")
    elif kappa > 0.2:
        print("Fair Agreement")
    elif kappa > 0:
        print("Slight Agreement")
    else:
        print("Poor Agreement")
    
    # Emotion group analysis
    from .emotion_rubric import GoEmotionsRubric
    emotion_groups = GoEmotionsRubric.get_emotion_groups()
    
    print(f"\nPerformance by Emotion Groups:")
    for group_name, group_emotions in emotion_groups.items():
        group_kappas = [emotion_metrics[emotion]['kappa'] for emotion in group_emotions if emotion in emotion_metrics]
        avg_kappa = np.mean(group_kappas) if group_kappas else 0.0
        print(f"  {group_name:10s}: Average κ = {avg_kappa:.3f}")


def calculate_multi_label_specific_metrics(human_labels: Dict[str, List[str]], 
                                         model_labels: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculate additional multi-label specific metrics.
    """
    comment_ids = list(human_labels.keys())
    
    # Hamming Loss (average label-wise error)
    total_errors = 0
    total_labels = 0
    
    # Jaccard Score (subset accuracy)
    jaccard_scores = []
    
    for comment_id in comment_ids:
        human_set = set(human_labels[comment_id])
        model_set = set(model_labels[comment_id])
        
        # For Hamming loss
        all_possible_emotions = human_set.union(model_set)
        total_labels += len(all_possible_emotions) if all_possible_emotions else 1
        
        for emotion in all_possible_emotions:
            if (emotion in human_set) != (emotion in model_set):
                total_errors += 1
        
        # For Jaccard score
        if len(human_set.union(model_set)) > 0:
            jaccard = len(human_set.intersection(model_set)) / len(human_set.union(model_set))
        else:
            jaccard = 1.0  # Both sets are empty
        jaccard_scores.append(jaccard)
    
    hamming_loss = total_errors / total_labels if total_labels > 0 else 0.0
    subset_accuracy = np.mean(jaccard_scores)
    
    return {
        'hamming_loss': hamming_loss,
        'subset_accuracy': subset_accuracy
    }


# Keep the original function name for compatibility but use new metrics
def calculate_multilabel_metrics(human_labels: Dict[str, List[str]], 
                                model_labels: Dict[str, List[str]], 
                                categories: List[str]) -> Dict[str, float]:
    """
    Calculate metrics for multi-label emotion classification.
    This now calculates the 4 specified metrics plus multi-label specific ones.
    """
    # Get the main 4 metrics
    main_metrics = calculate_agreement_metrics(human_labels, model_labels, categories)
    
    # Get additional multi-label metrics
    ml_metrics = calculate_multi_label_specific_metrics(human_labels, model_labels)
    
    # Combine results
    main_metrics.update(ml_metrics)
    
    return main_metrics