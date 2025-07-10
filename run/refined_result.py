import os
import sys
import argparse
import pandas as pd
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score, brier_score_loss, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the existing save_confusion_matrix if possible
try:
    from utils.plotting import save_confusion_matrix
except ImportError:
    # Define our own if import fails
    def save_confusion_matrix(y_true, y_pred, class_names, output_dir, epoch=None, acc=None, filename_prefix=""):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure and axes
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        
        # Save figure
        plt.tight_layout()
        filename = f"{filename_prefix}confusion_matrix.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
        # Also save with raw counts
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (counts)')
        
        # Save figure
        plt.tight_layout()
        filename = f"{filename_prefix}confusion_matrix_counts.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def refine_prediction(model_pred, yolo_text):
    """
    Refine the predicted class (model_pred) based on YOLO_prediction string.
    Rule list nội bộ:
    rules = [
        {"model_pred": 0, "keyword": "osteophyte", "new_pred": 2},
        {"model_pred": 1, "keyword": "osteophyte", "new_pred": 2},
        {"model_pred": 1, "keyword": "osteophytebig", "new_pred": 2},
        {"model_pred": 3, "keyword": "sclerosis", "new_pred": 4},
        {"model_pred": 0, "keyword": "narrowing", "new_pred": 2},
        {"model_pred": 2, "keyword": "narrowing", "new_pred": 3},
        {"model_pred": 4, "keyword": "sclerosis", "new_pred": 4},
        {"model_pred": 4, "keyword": "osteophyte", "new_pred": 3},
        {"model_pred": 1, "keyword": "osteophytemore", "new_pred": 2},
        {"model_pred": 0, "keyword": "osteophytebig", "new_pred": 2},
        # thêm các rule khác ở đây nếu muốn
    ]
    """
    text = yolo_text.lower() if isinstance(yolo_text, str) else ""
    rules = [
        {"model_pred": 0, "keyword": "osteophyte", "new_pred": 2},
        {"model_pred": 1, "keyword": "osteophyte", "new_pred": 2},
        {"model_pred": 1, "keyword": "osteophytebig", "new_pred": 2},
        {"model_pred": 3, "keyword": "sclerosis", "new_pred": 4},
        {"model_pred": 0, "keyword": "narrowing", "new_pred": 2},
        {"model_pred": 2, "keyword": "narrowing", "new_pred": 3},
        {"model_pred": 4, "keyword": "sclerosis", "new_pred": 4},
        {"model_pred": 4, "keyword": "osteophyte", "new_pred": 3},
        {"model_pred": 1, "keyword": "osteophytemore", "new_pred": 2},
        {"model_pred": 0, "keyword": "osteophytebig", "new_pred": 2},
        # thêm các rule khác ở đây nếu muốn
    ]
    for rule in rules:
        if model_pred == rule["model_pred"] and rule["keyword"] in text:
            return rule["new_pred"]
    return model_pred

def calculate_sensitivity_specificity(y_true, y_pred):
    # Binary conversion: consider classes 0,1 as negative, 2,3,4 as positive
    y_true_bin = np.array(y_true) >= 2
    y_pred_bin = np.array(y_pred) >= 2
    
    tp = np.sum((y_true_bin == True) & (y_pred_bin == True))
    tn = np.sum((y_true_bin == False) & (y_pred_bin == False))
    fp = np.sum((y_true_bin == False) & (y_pred_bin == True))
    fn = np.sum((y_true_bin == True) & (y_pred_bin == False))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity, specificity

def calculate_per_class_metrics(y_true, y_pred, num_classes):
    """Calculate per-class sensitivity and specificity using one-vs-rest."""
    per_class_sensitivity = []
    per_class_specificity = []
    for cls in range(num_classes):
        y_true_cls = np.array(y_true) == cls
        y_pred_cls = np.array(y_pred) == cls
        tp = np.sum(y_true_cls & y_pred_cls)
        tn = np.sum(~(y_true_cls | y_pred_cls))
        fp = np.sum(~y_true_cls & y_pred_cls)
        fn = np.sum(y_true_cls & ~y_pred_cls)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        per_class_sensitivity.append(sens)
        per_class_specificity.append(spec)
    return per_class_sensitivity, per_class_specificity

def main(csv_path, model_name, config='default.yaml'):
    # Load config and CSV
    config_path = os.path.join('config', config)
    config_data = load_config(config_path)
    df = pd.read_csv(csv_path)
    
    # Check necessary columns exist
    if 'ground_truth' not in df.columns:
        print("CSV must include a 'ground_truth' column (from label_column).")
        sys.exit(1)
    if model_name not in df.columns:
        print(f"CSV must include a '{model_name}' column for model predictions.")
        sys.exit(1)
    if 'YOLO_prediction' not in df.columns:
        print("CSV must include a 'YOLO_prediction' column.")
        sys.exit(1)
    
    # Extract original predictions and YOLO predictions
    model_preds = df[model_name].tolist()
    yolo_texts = df['YOLO_prediction'].tolist()
    ground_truth = df['ground_truth'].tolist()
    
    # Apply refinement rules row by row
    refined_preds = []
    for mp, yt in zip(model_preds, yolo_texts):
        refined = refine_prediction(mp, yt)
        refined_preds.append(refined)
    
    df['refined_prediction'] = refined_preds
    
    # Compute metrics
    acc = accuracy_score(ground_truth, refined_preds)
    f1 = f1_score(ground_truth, refined_preds, average='weighted')
    precision = precision_score(ground_truth, refined_preds, average='weighted', zero_division=0)
    recall = recall_score(ground_truth, refined_preds, average='weighted')
    kappa = cohen_kappa_score(ground_truth, refined_preds)
    
    # Sensitivity & Specificity (binary: classes>=2 as positive)
    sensitivity, specificity = calculate_sensitivity_specificity(ground_truth, refined_preds)
    
    # For ROC AUC and Brier Score, perform binary conversion (severity>=2 positive)
    ground_truth_bin = np.array(ground_truth) >= 2
    refined_bin = np.array(refined_preds) >= 2
    try:
        auc = roc_auc_score(ground_truth_bin, refined_bin)
    except Exception as e:
        auc = None
        print("ROC AUC could not be computed:", e)
    
    # Compute Brier score: one-hot encoding
    num_classes = len(config_data['data']['class_labels'])
    gt_onehot = np.zeros((len(ground_truth), num_classes))
    pred_onehot = np.zeros((len(refined_preds), num_classes))
    for i, (gt, pred) in enumerate(zip(ground_truth, refined_preds)):
        gt_onehot[i, int(gt)] = 1
        pred_onehot[i, int(pred)] = 1
    brier = brier_score_loss(gt_onehot.ravel(), pred_onehot.ravel())
    
    # Print metrics and classification report
    print("Classification Report:")
    print(classification_report(ground_truth, refined_preds, target_names=[str(x) for x in config_data['data']['class_names']], zero_division=0))
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Sensitivity (binary): {sensitivity:.4f}")
    print(f"Specificity (binary): {specificity:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    if auc is not None:
        print(f"ROC AUC (binary): {auc:.4f}")
    else:
        print("ROC AUC (binary): Not computed")
    print(f"Brier Score: {brier:.4f}")
    
    # --- FIXED LINE: use ground_truth and refined_preds ---
    per_class_sensitivity, per_class_specificity = calculate_per_class_metrics(ground_truth, refined_preds, num_classes)
    
    for i, class_name in enumerate(config_data['data']['class_names']):
        print(f"Sensitivity {class_name}: {per_class_sensitivity[i]:.4f}")
    
    for i, class_name in enumerate(config_data['data']['class_names']):
        print(f"Specificity {class_name}: {per_class_specificity[i]:.4f}")
    
    # Generate and save confusion matrix
    output_dir = os.path.dirname(csv_path)
    save_confusion_matrix(ground_truth, refined_preds, 
                         config_data['data']['class_names'], 
                         output_dir, 
                         filename_prefix=f"{model_name}_refined_")
    
    # Print confusion matrix to console
    cm = confusion_matrix(ground_truth, refined_preds)
    print("\nConfusion Matrix (counts):")
    print(cm)
    
    # Calculate and print normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    print("\nNormalized Confusion Matrix:")
    print(cm_normalized)
    
    # Optionally, save refined CSV with a new name
    output_csv = os.path.join(os.path.dirname(csv_path), f"refined_results_{model_name}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Refined results saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine model predictions using YOLO output and compute metrics.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to all_models_predictions_yolo.csv")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model column to refine (e.g., 'resnet50')")
    parser.add_argument('--config', type=str, default='default.yaml', help="Name of the configuration file in config folder")
    args = parser.parse_args()
    
    main(csv_path=args.csv_path, model_name=args.model_name, config=args.config)