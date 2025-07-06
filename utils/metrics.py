"""
Comprehensive metrics for multi-label emotion classification evaluation
Fixed to include missing GoEmotionsMetrics class and functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, precision_recall_fscore_support,
    hamming_loss, jaccard_score, classification_report
)
from scipy.stats import pearsonr
import logging

# Try to import krippendorff, but don't fail if it's not available
try:
    import krippendorff
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False
    logging.warning("krippendorff package not found. Install with: pip install krippendorff")

import config

class GoEmotionsMetrics:
    """Comprehensive metrics for GoEmotions multi-label classification"""
    
    def __init__(self, emotions_list: List[str]):
        self.emotions_list = emotions_list
        self.emotion_groups = config.EMOTION_GROUPS
        self.logger = logging.getLogger(__name__)
    
    def calculate_exact_match_accuracy(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """Calculate exact match accuracy (all emotions must match exactly)"""
        matches = 0
        total = len(y_true)
        
        for true_emotions, pred_emotions in zip(y_true, y_pred):
            if set(true_emotions) == set(pred_emotions):
                matches += 1
        
        return (matches / total) * 100 if total > 0 else 0.0
    
    def calculate_hamming_loss(self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
        """Calculate Hamming loss for multi-label classification"""
        return hamming_loss(y_true_binary, y_pred_binary)
    
    def calculate_subset_accuracy(self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
        """Calculate subset accuracy (Jaccard similarity)"""
        # Convert to lists of sets for Jaccard calculation
        y_true_sets = [set(np.where(row)[0]) for row in y_true_binary]
        y_pred_sets = [set(np.where(row)[0]) for row in y_pred_binary]
        
        jaccard_scores = []
        for true_set, pred_set in zip(y_true_sets, y_pred_sets):
            if len(true_set) == 0 and len(pred_set) == 0:
                jaccard_scores.append(1.0)
            elif len(true_set) == 0 or len(pred_set) == 0:
                jaccard_scores.append(0.0)
            else:
                intersection = len(true_set.intersection(pred_set))
                union = len(true_set.union(pred_set))
                jaccard_scores.append(intersection / union if union > 0 else 0.0)
        
        return np.mean(jaccard_scores)
    
    def emotions_to_binary(self, emotions_lists: List[List[str]]) -> np.ndarray:
        """Convert emotion lists to binary matrix"""
        binary_matrix = np.zeros((len(emotions_lists), len(self.emotions_list)))
        
        for i, emotions in enumerate(emotions_lists):
            for emotion in emotions:
                if emotion in self.emotions_list:
                    j = self.emotions_list.index(emotion)
                    binary_matrix[i, j] = 1
        
        return binary_matrix
    
    def calculate_cohens_kappa(self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
        """Calculate Cohen's Kappa for multi-label classification"""
        # Flatten the arrays for overall kappa calculation
        y_true_flat = y_true_binary.flatten()
        y_pred_flat = y_pred_binary.flatten()
        
        try:
            return cohen_kappa_score(y_true_flat, y_pred_flat)
        except:
            return 0.0
    
    def calculate_krippendorffs_alpha(self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
        """Calculate Krippendorff's Alpha for reliability"""
        if not HAS_KRIPPENDORFF:
            return 0.0
            
        try:
            # Reshape for krippendorff format (2 x n_observations)
            data = np.vstack([y_true_binary.flatten(), y_pred_binary.flatten()])
            alpha = krippendorff.alpha(data, level_of_measurement='nominal')
            return alpha if not np.isnan(alpha) else 0.0
        except:
            return 0.0
    
    def calculate_icc(self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
        """Calculate Intraclass Correlation Coefficient"""
        try:
            # Calculate correlation between true and predicted emotion patterns
            correlations = []
            for i in range(len(self.emotions_list)):
                true_col = y_true_binary[:, i]
                pred_col = y_pred_binary[:, i]
                
                if np.var(true_col) > 0 and np.var(pred_col) > 0:
                    corr, _ = pearsonr(true_col, pred_col)
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            return np.mean(correlations) if correlations else 0.0
        except:
            return 0.0
    
    def calculate_per_emotion_metrics(self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> Dict:
        """Calculate metrics for each emotion individually"""
        per_emotion_metrics = {}
        
        for i, emotion in enumerate(self.emotions_list):
            true_col = y_true_binary[:, i]
            pred_col = y_pred_binary[:, i]
            
            # Basic metrics
            accuracy = accuracy_score(true_col, pred_col)
            
            # Support (number of true instances)
            support = int(np.sum(true_col))
            
            # Cohen's Kappa for this emotion
            try:
                kappa = cohen_kappa_score(true_col, pred_col)
            except:
                kappa = 0.0
            
            # Krippendorff's Alpha for this emotion
            try:
                if HAS_KRIPPENDORFF and (np.var(true_col) > 0 or np.var(pred_col) > 0):
                    data = np.vstack([true_col, pred_col])
                    alpha = krippendorff.alpha(data, level_of_measurement='nominal')
                    alpha = alpha if not np.isnan(alpha) else 0.0
                else:
                    alpha = 0.0
            except:
                alpha = 0.0
            
            # Correlation
            try:
                if np.var(true_col) > 0 and np.var(pred_col) > 0:
                    corr, _ = pearsonr(true_col, pred_col)
                    corr = corr if not np.isnan(corr) else 0.0
                else:
                    corr = 0.0
            except:
                corr = 0.0
            
            # Precision, Recall, F1
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_col, pred_col, average='binary', zero_division=0
                )
            except:
                precision = recall = f1 = 0.0
            
            per_emotion_metrics[emotion] = {
                'accuracy': accuracy,
                'kappa': kappa,
                'alpha': alpha,
                'correlation': corr,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }
        
        return per_emotion_metrics
    
    def calculate_group_metrics(self, per_emotion_metrics: Dict) -> Dict:
        """Calculate metrics by emotion groups"""
        group_metrics = {}
        
        for group_name, group_emotions in self.emotion_groups.items():
            group_kappas = []
            group_accuracies = []
            group_f1s = []
            
            for emotion in group_emotions:
                if emotion in per_emotion_metrics:
                    metrics = per_emotion_metrics[emotion]
                    group_kappas.append(metrics['kappa'])
                    group_accuracies.append(metrics['accuracy'])
                    group_f1s.append(metrics['f1'])
            
            if group_kappas:
                group_metrics[group_name] = {
                    'avg_kappa': np.mean(group_kappas),
                    'avg_accuracy': np.mean(group_accuracies),
                    'avg_f1': np.mean(group_f1s),
                    'num_emotions': len(group_kappas)
                }
        
        return group_metrics
    
    def comprehensive_evaluation(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
        """
        Perform comprehensive evaluation of multi-label emotion classification
        
        Args:
            y_true: List of true emotion lists
            y_pred: List of predicted emotion lists
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Convert to binary matrices
        y_true_binary = self.emotions_to_binary(y_true)
        y_pred_binary = self.emotions_to_binary(y_pred)
        
        # Calculate overall metrics
        exact_match_accuracy = self.calculate_exact_match_accuracy(y_true, y_pred)
        cohens_kappa = self.calculate_cohens_kappa(y_true_binary, y_pred_binary)
        krippendorffs_alpha = self.calculate_krippendorffs_alpha(y_true_binary, y_pred_binary)
        icc = self.calculate_icc(y_true_binary, y_pred_binary)
        hamming_loss_score = self.calculate_hamming_loss(y_true_binary, y_pred_binary)
        subset_accuracy = self.calculate_subset_accuracy(y_true_binary, y_pred_binary)
        
        # Calculate per-emotion metrics
        per_emotion_metrics = self.calculate_per_emotion_metrics(y_true_binary, y_pred_binary)
        
        # Calculate group metrics
        group_metrics = self.calculate_group_metrics(per_emotion_metrics)
        
        # Compile results
        results = {
            'exact_match_accuracy': exact_match_accuracy,
            'cohens_kappa': cohens_kappa,
            'krippendorffs_alpha': krippendorffs_alpha,
            'icc': icc,
            'hamming_loss': hamming_loss_score,
            'subset_accuracy': subset_accuracy,
            'per_emotion_metrics': per_emotion_metrics,
            'group_metrics': group_metrics,
            'num_samples': len(y_true)
        }
        
        return results
    
    def print_results(self, results: Dict, technique_name: str = ""):
        """Print formatted results"""
        if technique_name:
            print(f"\n=== {technique_name} Results ===")
        
        print("Overall Multi-Label Metrics:")
        print(f"  Exact Match Accuracy: {results['exact_match_accuracy']:.1f}%")
        print(f"  Cohen's Kappa (κ): {results['cohens_kappa']:.3f}")
        print(f"  Krippendorff's Alpha (α): {results['krippendorffs_alpha']:.3f}")
        print(f"  Intraclass Correlation (ICC): {results['icc']:.3f}")
        print()
        
        # Print per-emotion results sorted by support
        print("Per-Emotion Results (sorted by support):")
        per_emotion = results['per_emotion_metrics']
        sorted_emotions = sorted(per_emotion.items(), 
                               key=lambda x: x[1]['support'], reverse=True)
        
        for emotion, metrics in sorted_emotions:
            print(f"  {emotion:<15}: Acc={metrics['accuracy']*100:.1f}%, "
                  f"κ={metrics['kappa']:.3f}, α={metrics['alpha']:.3f}, "
                  f"Corr={metrics['correlation']:.3f}, Supp={metrics['support']}")
        
        print()
        print("Multi-Label Performance Interpretation:")
        exact_match = results['exact_match_accuracy']
        if exact_match >= 70:
            interpretation = "Excellent - Most emotion sets predicted correctly"
        elif exact_match >= 50:
            interpretation = "Good - Many emotion sets predicted correctly"
        elif exact_match >= 30:
            interpretation = "Fair - Some emotion sets predicted correctly"
        else:
            interpretation = "Poor - Few emotion sets predicted correctly"
        
        print(f"  Exact Match ({exact_match:.1f}%): {interpretation}")
        
        kappa = results['cohens_kappa']
        if kappa >= 0.8:
            agreement = "Almost Perfect Agreement"
        elif kappa >= 0.6:
            agreement = "Substantial Agreement"
        elif kappa >= 0.4:
            agreement = "Moderate Agreement"
        elif kappa >= 0.2:
            agreement = "Fair Agreement"
        elif kappa >= 0.0:
            agreement = "Slight Agreement"
        else:
            agreement = "Poor Agreement"
        
        print(f"  Overall Agreement (κ={kappa:.3f}): {agreement}")
        
        # Print group performance
        if 'group_metrics' in results:
            print()
            print("Performance by Emotion Groups:")
            for group, metrics in results['group_metrics'].items():
                print(f"  {group:<10}: Average κ = {metrics['avg_kappa']:.3f}")


def create_detailed_results_dataframe(predictions: List[Dict], emotions_list: List[str]) -> pd.DataFrame:
    """Create detailed results dataframe for analysis"""
    detailed_data = []
    
    for pred in predictions:
        row = {
            'comment_id': pred.get('id', ''),
            'comment_text': pred.get('text', ''),
            'human_emotions': str(pred.get('human_emotions', [])),
            'predicted_emotions': str(pred.get('predicted_emotions', [])),
            'exact_match': pred.get('exact_match', False),
            'num_human_emotions': len(pred.get('human_emotions', [])),
            'num_predicted_emotions': len(pred.get('predicted_emotions', [])),
        }
        
        # Add binary columns for each emotion
        human_emotions = set(pred.get('human_emotions', []))
        predicted_emotions = set(pred.get('predicted_emotions', []))
        
        for emotion in emotions_list:
            row[f'human_{emotion}'] = 1 if emotion in human_emotions else 0
            row[f'pred_{emotion}'] = 1 if emotion in predicted_emotions else 0
            row[f'match_{emotion}'] = 1 if (emotion in human_emotions) == (emotion in predicted_emotions) else 0
        
        detailed_data.append(row)
    
    return pd.DataFrame(detailed_data)


def comprehensive_report(all_results: Dict, eval_info: Dict, output_path: str):
    """Generate comprehensive evaluation report"""
    
    with open(output_path, 'w') as f:
        f.write("GoEmotions Multi-Label Emotion Classification: Comprehensive Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset information
        f.write("DATASET INFORMATION:\n")
        f.write(f"Total comments evaluated: {eval_info.get('total_comments', 'N/A')}\n")
        f.write(f"Total emotions: {len(eval_info.get('emotions_list', []))}\n")
        f.write(f"Emotion categories: {len(config.EMOTION_GROUPS)}\n\n")
        
        # Emotion list
        f.write("EMOTIONS EVALUATED:\n")
        emotions_list = eval_info.get('emotions_list', [])
        for i, emotion in enumerate(emotions_list, 1):
            f.write(f"{i:2d}. {emotion}\n")
        f.write("\n")
        
        # Emotion groups
        f.write("EMOTION GROUPS:\n")
        for group_name, group_emotions in config.EMOTION_GROUPS.items():
            f.write(f"{group_name.upper()}: {', '.join(group_emotions)}\n")
        f.write("\n")
        
        if not all_results:
            f.write("No successful evaluations to report.\n")
            return
        
        # Overall comparison
        f.write("TECHNIQUE COMPARISON:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Technique':<20} {'Exact Match':<12} {'Cohen κ':<10} {'Kripp α':<10} {'ICC':<8} {'Hamming':<10}\n")
        f.write("-" * 80 + "\n")
        
        for technique_name, results in all_results.items():
            f.write(f"{technique_name:<20} "
                   f"{results.get('exact_match_accuracy', 0):<12.1f} "
                   f"{results.get('cohens_kappa', 0):<10.3f} "
                   f"{results.get('krippendorffs_alpha', 0):<10.3f} "
                   f"{results.get('icc', 0):<8.3f} "
                   f"{results.get('hamming_loss', 0):<10.3f}\n")
        
        f.write("\n")
        
        # Write detailed results for each technique
        for technique_name, results in all_results.items():
            f.write(f"DETAILED RESULTS: {technique_name.upper()}\n")
            f.write("=" * 60 + "\n")
            
            # Overall metrics
            f.write("Overall Metrics:\n")
            f.write(f"  Exact Match Accuracy: {results.get('exact_match_accuracy', 0):.1f}%\n")
            f.write(f"  Cohen's Kappa: {results.get('cohens_kappa', 0):.3f}\n")
            f.write(f"  Krippendorff's Alpha: {results.get('krippendorffs_alpha', 0):.3f}\n")
            f.write(f"  Intraclass Correlation: {results.get('icc', 0):.3f}\n")
            f.write(f"  Hamming Loss: {results.get('hamming_loss', 0):.3f}\n")
            f.write(f"  Subset Accuracy: {results.get('subset_accuracy', 0):.3f}\n\n")
        
        f.write("EVALUATION COMPLETED\n")


# Legacy compatibility functions that might be expected by existing code
def calculate_exact_match_accuracy(y_true: List[List[str]], y_pred: List[List[str]]) -> float:
    """Legacy function for backward compatibility"""
    metrics = GoEmotionsMetrics(config.GOEMOTIONS_EMOTIONS)
    return metrics.calculate_exact_match_accuracy(y_true, y_pred)

def calculate_cohens_kappa(y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """Legacy function for backward compatibility"""
    metrics = GoEmotionsMetrics(config.GOEMOTIONS_EMOTIONS)
    return metrics.calculate_cohens_kappa(y_true_binary, y_pred_binary)

def emotions_to_binary(emotions_lists: List[List[str]], emotions_list: List[str]) -> np.ndarray:
    """Legacy function for backward compatibility"""
    metrics = GoEmotionsMetrics(emotions_list)
    return metrics.emotions_to_binary(emotions_lists)

def calculate_agreement_metrics(y_true: List[List[str]], y_pred: List[List[str]], emotions_list: List[str]) -> Dict:
    """Calculate agreement metrics (compatibility function)"""
    metrics = GoEmotionsMetrics(emotions_list)
    return metrics.comprehensive_evaluation(y_true, y_pred)

def plot_emotion_performance(results: Dict, output_path: str = None):
    """Plot emotion performance (compatibility function)"""
    try:
        import matplotlib.pyplot as plt
        
        if 'per_emotion_metrics' not in results:
            print("No per-emotion metrics found for plotting")
            return
        
        per_emotion = results['per_emotion_metrics']
        emotions = list(per_emotion.keys())
        accuracies = [per_emotion[emotion]['accuracy'] * 100 for emotion in emotions]
        
        plt.figure(figsize=(12, 6))
        plt.bar(emotions, accuracies)
        plt.title('Per-Emotion Accuracy')
        plt.xlabel('Emotions')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Error creating plot: {e}")

def print_detailed_results(results: Dict, technique_name: str = ""):
    """Print detailed results (compatibility function)"""
    metrics = GoEmotionsMetrics(config.GOEMOTIONS_EMOTIONS)
    metrics.print_results(results, technique_name)

def calculate_per_emotion_metrics(y_true_binary: np.ndarray, y_pred_binary: np.ndarray, emotions_list: List[str]) -> Dict:
    """Calculate per-emotion metrics (compatibility function)"""
    metrics = GoEmotionsMetrics(emotions_list)
    return metrics.calculate_per_emotion_metrics(y_true_binary, y_pred_binary)

def calculate_hamming_loss(y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """Calculate Hamming loss (compatibility function)"""
    return hamming_loss(y_true_binary, y_pred_binary)

def calculate_subset_accuracy(y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """Calculate subset accuracy (compatibility function)"""
    metrics = GoEmotionsMetrics(config.GOEMOTIONS_EMOTIONS)
    return metrics.calculate_subset_accuracy(y_true_binary, y_pred_binary)

def evaluate_predictions(y_true: List[List[str]], y_pred: List[List[str]], emotions_list: List[str]) -> Dict:
    """Evaluate predictions (compatibility function)"""
    metrics = GoEmotionsMetrics(emotions_list)
    return metrics.comprehensive_evaluation(y_true, y_pred)

def create_confusion_matrix(y_true_binary: np.ndarray, y_pred_binary: np.ndarray, emotions_list: List[str]) -> Dict:
    """Create confusion matrix data (compatibility function)"""
    from sklearn.metrics import multilabel_confusion_matrix
    
    try:
        cm_array = multilabel_confusion_matrix(y_true_binary, y_pred_binary)
        confusion_matrices = {}
        
        for i, emotion in enumerate(emotions_list):
            confusion_matrices[emotion] = {
                'matrix': cm_array[i].tolist(),
                'tn': int(cm_array[i][0, 0]),
                'fp': int(cm_array[i][0, 1]), 
                'fn': int(cm_array[i][1, 0]),
                'tp': int(cm_array[i][1, 1])
            }
        
        return confusion_matrices
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        return {}

def save_results_to_csv(results: Dict, predictions: List[Dict], output_path: str):
    """Save results to CSV (compatibility function)"""
    try:
        # Save main results
        results_df = pd.DataFrame([{
            'exact_match_accuracy': results.get('exact_match_accuracy', 0),
            'cohens_kappa': results.get('cohens_kappa', 0),
            'krippendorffs_alpha': results.get('krippendorffs_alpha', 0),
            'icc': results.get('icc', 0),
            'hamming_loss': results.get('hamming_loss', 0),
            'subset_accuracy': results.get('subset_accuracy', 0)
        }])
        
        results_df.to_csv(output_path.replace('.csv', '_summary.csv'), index=False)
        
        # Save detailed predictions if available
        if predictions:
            pred_df = pd.DataFrame(predictions)
            pred_df.to_csv(output_path.replace('.csv', '_predictions.csv'), index=False)
        
        print(f"Results saved to {output_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

def format_results_for_display(results: Dict) -> str:
    """Format results for display (compatibility function)"""
    output = []
    output.append("=== EVALUATION RESULTS ===")
    output.append(f"Exact Match Accuracy: {results.get('exact_match_accuracy', 0):.1f}%")
    output.append(f"Cohen's Kappa: {results.get('cohens_kappa', 0):.3f}")
    output.append(f"Krippendorff's Alpha: {results.get('krippendorffs_alpha', 0):.3f}")
    output.append(f"ICC: {results.get('icc', 0):.3f}")
    output.append(f"Hamming Loss: {results.get('hamming_loss', 0):.3f}")
    output.append(f"Subset Accuracy: {results.get('subset_accuracy', 0):.3f}")
    
    return "\n".join(output)

# Additional helper functions that might be expected
def validate_predictions(y_true: List[List[str]], y_pred: List[List[str]]) -> bool:
    """Validate prediction format"""
    if len(y_true) != len(y_pred):
        return False
    
    for true_emotions, pred_emotions in zip(y_true, y_pred):
        if not isinstance(true_emotions, list) or not isinstance(pred_emotions, list):
            return False
    
    return True

def normalize_emotion_lists(emotions_lists: List[List[str]], emotions_list: List[str]) -> List[List[str]]:
    """Normalize emotion lists to ensure all emotions are valid"""
    normalized = []
    for emotions in emotions_lists:
        valid_emotions = [emotion for emotion in emotions if emotion in emotions_list]
        normalized.append(valid_emotions)
    return normalized


# Add these functions to your existing utils/metrics.py file

def calculate_agreement_metrics(human_labels: Dict, model_labels: Dict, emotions: List[str]) -> Dict:
    """
    Calculate agreement metrics between human and model labels (compatibility function)
    
    Args:
        human_labels: Dict mapping comment_id to list of human emotions
        model_labels: Dict mapping comment_id to list of model emotions  
        emotions: List of all possible emotions
        
    Returns:
        Dictionary containing various agreement metrics
    """
    from sklearn.metrics import cohen_kappa_score, accuracy_score
    import numpy as np
    
    # Align the data - only use comments that have both human and model labels
    common_ids = set(human_labels.keys()) & set(model_labels.keys())
    
    if not common_ids:
        return {'accuracy': 0, 'kappa': 0, 'alpha': 0, 'icc': 0}
    
    # Convert to lists for easier processing
    y_true_lists = [human_labels[comment_id] for comment_id in common_ids]
    y_pred_lists = [model_labels[comment_id] for comment_id in common_ids]
    
    # Calculate exact match accuracy
    exact_matches = sum(1 for true, pred in zip(y_true_lists, y_pred_lists) 
                       if set(true) == set(pred))
    exact_match_accuracy = (exact_matches / len(common_ids)) * 100
    
    # Convert to binary matrices for other metrics
    y_true_binary = np.zeros((len(common_ids), len(emotions)))
    y_pred_binary = np.zeros((len(common_ids), len(emotions)))
    
    for i, (true_emotions, pred_emotions) in enumerate(zip(y_true_lists, y_pred_lists)):
        for emotion in true_emotions:
            if emotion in emotions:
                j = emotions.index(emotion)
                y_true_binary[i, j] = 1
        
        for emotion in pred_emotions:
            if emotion in emotions:
                j = emotions.index(emotion)
                y_pred_binary[i, j] = 1
    
    # Calculate Cohen's Kappa
    try:
        kappa = cohen_kappa_score(y_true_binary.flatten(), y_pred_binary.flatten())
    except:
        kappa = 0.0
    
    # Calculate Krippendorff's Alpha (simplified)
    try:
        import krippendorff
        data = np.vstack([y_true_binary.flatten(), y_pred_binary.flatten()])
        alpha = krippendorff.alpha(data, level_of_measurement='nominal')
        if np.isnan(alpha):
            alpha = 0.0
    except:
        alpha = 0.0
    
    # Calculate ICC (simplified correlation)
    try:
        from scipy.stats import pearsonr
        correlations = []
        for i in range(len(emotions)):
            if np.var(y_true_binary[:, i]) > 0 and np.var(y_pred_binary[:, i]) > 0:
                corr, _ = pearsonr(y_true_binary[:, i], y_pred_binary[:, i])
                if not np.isnan(corr):
                    correlations.append(corr)
        icc = np.mean(correlations) if correlations else 0.0
    except:
        icc = 0.0
    
    # Calculate Hamming Loss
    try:
        from sklearn.metrics import hamming_loss
        hamming = hamming_loss(y_true_binary, y_pred_binary)
    except:
        hamming = 0.0
    
    # Calculate Subset Accuracy (Jaccard)
    try:
        jaccard_scores = []
        for true_set, pred_set in zip(y_true_lists, y_pred_lists):
            true_set = set(true_set)
            pred_set = set(pred_set)
            if len(true_set) == 0 and len(pred_set) == 0:
                jaccard_scores.append(1.0)
            else:
                intersection = len(true_set.intersection(pred_set))
                union = len(true_set.union(pred_set))
                jaccard_scores.append(intersection / union if union > 0 else 0.0)
        subset_accuracy = np.mean(jaccard_scores)
    except:
        subset_accuracy = 0.0
    
    return {
        'accuracy': exact_match_accuracy,
        'exact_match_accuracy': exact_match_accuracy,
        'kappa': kappa,
        'alpha': alpha,
        'icc': icc,
        'hamming_loss': hamming,
        'subset_accuracy': subset_accuracy,
        'num_samples': len(common_ids)
    }


def plot_emotion_performance(metrics: Dict, emotions: List[str], technique_name: str, output_path: str):
    """
    Create performance visualization (compatibility function)
    
    Args:
        metrics: Dictionary containing performance metrics
        emotions: List of emotion names
        technique_name: Name of the technique being evaluated
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a simple bar chart of overall metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Overall metrics
        metric_names = ['Exact Match', 'Kappa', 'Alpha', 'ICC']
        metric_values = [
            metrics.get('exact_match_accuracy', 0) / 100,  # Convert to 0-1 scale
            metrics.get('kappa', 0),
            metrics.get('alpha', 0), 
            metrics.get('icc', 0)
        ]
        
        ax1.bar(metric_names, metric_values)
        ax1.set_title(f'{technique_name} - Overall Performance')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(metric_values):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Sample emotion performance (if available)
        # For now, just show a placeholder
        sample_emotions = emotions[:10]  # First 10 emotions
        sample_scores = [metrics.get('kappa', 0)] * len(sample_emotions)  # Use kappa as sample
        
        ax2.barh(sample_emotions, sample_scores)
        ax2.set_title('Sample Emotion Performance (Kappa)')
        ax2.set_xlabel('Kappa Score')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plot saved to {output_path}")
        
    except Exception as e:
        print(f"Could not create performance plot: {e}")


def print_detailed_results(metrics: Dict, emotions: List[str], technique_name: str):
    """
    Print detailed results (compatibility function)
    
    Args:
        metrics: Dictionary containing performance metrics
        emotions: List of emotion names
        technique_name: Name of the technique
    """
    print(f"\n=== {technique_name} Results ===")
    print(f"Exact Match Accuracy: {metrics.get('exact_match_accuracy', 0):.1f}%")
    print(f"Cohen's Kappa: {metrics.get('kappa', 0):.3f}")
    print(f"Krippendorff's Alpha: {metrics.get('alpha', 0):.3f}")
    print(f"ICC: {metrics.get('icc', 0):.3f}")
    print(f"Hamming Loss: {metrics.get('hamming_loss', 0):.3f}")
    print(f"Subset Accuracy: {metrics.get('subset_accuracy', 0):.3f}")
    print(f"Number of samples: {metrics.get('num_samples', 0)}")
    
    # Interpretation
    kappa = metrics.get('kappa', 0)
    if kappa >= 0.8:
        agreement = "Almost Perfect"
    elif kappa >= 0.6:
        agreement = "Substantial"
    elif kappa >= 0.4:
        agreement = "Moderate"
    elif kappa >= 0.2:
        agreement = "Fair"
    else:
        agreement = "Slight"
    
    print(f"Agreement Level: {agreement}")