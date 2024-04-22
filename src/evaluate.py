import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, det_curve

def read_predictions(file_path):
    predictions = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            image_name = parts[0]
            prob = float(parts[1])
            predicted_label = int(parts[2])
            predictions[image_name] = (prob, predicted_label)
    return predictions

def evaluate_predictions(predictions, target_folder):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    total = len(predictions)

    target_files = set(os.listdir(target_folder))

    probs = []
    true_labels = []
    
    for image_name, (prob, predicted_label) in predictions.items():
        actual_label = 1 if image_name in target_files else 0
        true_labels.append(actual_label)
        probs.append(prob)
        if predicted_label == 1 and actual_label == 1:
            true_positives += 1
        elif predicted_label == 0 and actual_label == 0:
            true_negatives += 1
        elif predicted_label == 1 and actual_label == 0:
            false_positives += 1
        elif predicted_label == 0 and actual_label == 1:
            false_negatives += 1

    metrics = {
        'accuracy': (true_positives + true_negatives) / total,
        'precision': true_positives / (true_positives + false_positives) if true_positives + false_positives else 0,
        'recall': true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0,
        'f1_score': 2 * (true_positives / (true_positives + false_positives) * true_positives / (true_positives + false_negatives)) / (true_positives / (true_positives + false_positives) + true_positives / (true_positives + false_negatives)) if true_positives + false_positives and true_positives + false_negatives else 0,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total': total
    }

    # Additional metrics for ROC and Precision-Recall curves
    if len(set(true_labels)) > 1:  # Ensure we have both classes
        fpr, tpr, _ = roc_curve(true_labels, probs)
        precision, recall, _ = precision_recall_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        metrics.update({'roc_auc': roc_auc, 'pr_auc': pr_auc, 'fpr': fpr, 'tpr': tpr, 'precision_curve': precision, 'recall_curve': recall})
    
    return metrics

def plot_results(metrics):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['fpr'], metrics['tpr'], label=f'ROC Curve (area = {metrics["roc_auc"]:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(metrics['recall_curve'], metrics['precision_curve'], label=f'Precision-Recall Curve (area = {metrics["pr_auc"]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction accuracy.")
    parser.add_argument("predictions_path", help="Path to the predictions text file.")
    parser.add_argument("--target_folder", default="./target", help="Folder containing target files")
    parser.add_argument("--plot", action="store_true", help="Plot ROC and Precision-Recall curves")
    args = parser.parse_args()

    predictions = read_predictions(args.predictions_path)
    metrics = evaluate_predictions(predictions, args.target_folder)
    
    print("-"*50)
    print("Evaluation results...")
    print("="*50)
    print(f"Accuracy  : {metrics['accuracy']:.2%}")
    print("-"*50)
    print(f"Precision : {metrics['precision']:.2%}")
    print(f"Recall    : {metrics['recall']:.2%}")
    print(f"F1 Score  : {metrics['f1_score']:.2%}")
    print("-"*50)
    print(f"True Positives  : {metrics['true_positives']}")
    print(f"True Negatives  : {metrics['true_negatives']}")
    print(f"False Positives : {metrics['false_positives']}")
    print(f"False Negatives : {metrics['false_negatives']}")
    print("-"*50)
    print(f"Total images    : {metrics['total']}")
    print("-"*50)

    if args.plot and 'roc_auc' in metrics:
        plot_results(metrics)

if __name__ == "__main__":
    main()