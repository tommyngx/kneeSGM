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
from tqdm import tqdm  # add tqdm for progress bar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plotting import save_confusion_matrix

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def refine_prediction(model_pred, yolo_text):
    """
    Refine the predicted class (model_pred) based on YOLO_prediction string.
    Rule list nội bộ + bổ sung:
    - Hỗ trợ rule với nhiều keywords (all_required=True: tất cả keywords phải có trong text)
    """
    text = yolo_text.lower() if isinstance(yolo_text, str) else ""

    # Các rule cũ (nếu muốn giữ lại)
    rules = [
        {"model_pred": 0, "keyword": "osteophyte", "new_pred": 2},
        {"model_pred": 1, "keyword": "osteophyte", "new_pred": 2},
        {"model_pred": 1, "keyword": "osteophytebig", "new_pred": 2},
        {"model_pred": 3, "keyword": "sclerosis", "new_pred": 4},
        {"model_pred": 0, "keyword": "narrowing", "new_pred": 2},
        {"model_pred": 0, "keyword": "osteophyte", "new_pred": 3},
        #{"model_pred": 0, "keyword": "osteophyte", "new_pred": 4},
        {"model_pred": 0, "keyword": "osteophytebig", "new_pred": 1},
        #{"model_pred": 0, "keyword": "osteophytebig", "new_pred": 2},
        {"model_pred": 0, "keyword": "osteophytebig", "new_pred": 3},
    ]

    for rule in rules:
        if model_pred == rule["model_pred"] and rule["keyword"] in text:
            return rule["new_pred"]
    
    # Multi-keyword rules (all_required=True)
    multi_rules = [
        {"model_pred": 0, "keywords": ["osteophyte", "osteophytebig"], "new_pred": 2, "all_required": True},
        {"model_pred": 1, "keywords": ["osteophytebig", "healthy"], "new_pred": 2, "all_required": True},
    ]
    for rule in multi_rules:
        if model_pred == rule["model_pred"] and all(kw in text for kw in rule["keywords"]):
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

def save_consolidated_metrics(all_metrics, output_dir, suffix=""):
    """
    Save metrics from all models in a single CSV file with models as columns.
    If suffix is provided, append to filename and column names.
    """
    if not all_metrics:
        print("No metrics to save")
        return

    # List of metrics to save (order matters)
    metric_names = [
        'Accuracy', 'F1_Score', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'Kappa', 'AUC', 'Brier_Score',
        'Sensitivity_None', 'Sensitivity_Doubtful', 'Sensitivity_Mild', 'Sensitivity_Moderate', 'Sensitivity_Severe',
        'Specificity_None', 'Specificity_Doubtful', 'Specificity_Mild', 'Specificity_Moderate', 'Specificity_Severe'
    ]

    # Create DataFrame with metric names as index
    df = pd.DataFrame(index=metric_names)

    # Add a column for each model
    for model_name, metrics in all_metrics.items():
        # Only keep the metrics in metric_names, and add suffix if needed
        if suffix:
            metrics = {k + suffix: v for k, v in metrics.items() if k in metric_names}
            col_name = model_name + suffix
        else:
            metrics = {k: v for k, v in metrics.items() if k in metric_names}
            col_name = model_name
        # Round all metrics to 4 decimal places
        rounded_metrics = {k: round(v, 4) for k, v in metrics.items()}
        df[col_name] = pd.Series(rounded_metrics)

    # Ensure all numeric values are rounded to 4 decimal places in the DataFrame
    df = df.round(4)

    # Save the DataFrame to CSV with float_format to ensure consistent decimal places
    csv_path = os.path.join(output_dir, f"all_models_metrics{suffix}.csv")
    df.to_csv(csv_path, float_format='%.4f')
    print(f"Consolidated metrics saved to {csv_path}")

def compute_detection_metrics(y_true, y_pred):
    """
    Compute binary detection metrics (OA vs None OA).
    OA: class > 1, None OA: class < 2
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score, brier_score_loss, confusion_matrix
    y_true_bin = np.array(y_true) > 1
    y_pred_bin = np.array(y_pred) > 1
    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin)
    kappa = cohen_kappa_score(y_true_bin, y_pred_bin)
    try:
        auc = roc_auc_score(y_true_bin, y_pred_bin)
    except Exception:
        auc = None
    brier = brier_score_loss(y_true_bin, y_pred_bin)
    # Sensitivity and specificity for detection
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    detection_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    detection_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'Detection_Accuracy': acc,
        'Detection_F1_Score': f1,
        'Detection_Precision': precision,
        'Detection_Recall': recall,
        'Detection_Kappa': kappa,
        'Detection_AUC': auc if auc is not None else 0,
        'Detection_Brier_Score': brier,
        'Detection_Sensitivity': detection_sensitivity,
        'Detection_Specificity': detection_specificity,
    }

def save_detection_confusion_matrix(y_true, y_pred, output_dir, filename_prefix=""):
    """
    Save detection (OA vs None OA) confusion matrix as an image.
    OA: class > 1, None OA: class < 2
    """
    import numpy as np
    gt_bin = (np.array(y_true) > 1).astype(int)
    pred_bin = (np.array(y_pred) > 1).astype(int)
    # Use plotting's save_confusion_matrix for detection
    save_confusion_matrix(
        gt_bin,
        pred_bin,
        class_names=["None OA", "OA"],
        output_dir=output_dir,
        filename_prefix=filename_prefix + "detection_"
    )

def main(csv_path, model_name, config='default.yaml'):
    # Load config and CSV
    config_path = os.path.join('config', config)
    config_data = load_config(config_path)
    df = pd.read_csv(csv_path)

    # Check necessary columns exist
    if 'ground_truth' not in df.columns:
        print("CSV must include a 'ground_truth' column (from label_column).")
        sys.exit(1)
    if 'YOLO_prediction' not in df.columns:
        print("CSV must include a 'YOLO_prediction' column.")
        sys.exit(1)

    # Identify all model columns (exclude ground_truth, YOLO_prediction, image columns)
    exclude_cols = {'ground_truth', 'YOLO_prediction', 'image_path', 'image_name'}
    model_cols = [col for col in df.columns if col not in exclude_cols and not col.endswith('_refined')]

    output_dir = os.path.dirname(csv_path)
    refined_df = df.copy()

    # Add refined columns for all models
    for col in tqdm(model_cols, desc="Refining all models"):
        model_preds = df[col].tolist()
        yolo_texts = df['YOLO_prediction'].tolist()
        refined_preds = [refine_prediction(mp, yt) for mp, yt in zip(model_preds, yolo_texts)]
        refined_df[f'{col}_refined'] = refined_preds

    # Now, for each model (original and refined), compare with ground_truth and collect metrics
    all_metrics = {}
    class_names = [str(x) for x in config_data['data']['class_names']]
    num_classes = len(class_names)
    ground_truth = df['ground_truth'].tolist()

    # Evaluate original models
    for col in model_cols:
        preds = df[col].tolist()
        acc = accuracy_score(ground_truth, preds)
        f1 = f1_score(ground_truth, preds, average='weighted')
        precision = precision_score(ground_truth, preds, average='weighted', zero_division=0)
        recall = recall_score(ground_truth, preds, average='weighted')
        kappa = cohen_kappa_score(ground_truth, preds)
        sensitivity, specificity = calculate_sensitivity_specificity(ground_truth, preds)
        ground_truth_bin = np.array(ground_truth) >= 2
        pred_bin = np.array(preds) >= 2
        try:
            auc = roc_auc_score(ground_truth_bin, pred_bin)
        except Exception:
            auc = None
        gt_onehot = np.zeros((len(ground_truth), num_classes))
        pred_onehot = np.zeros((len(preds), num_classes))
        for i, (gt, pred) in enumerate(zip(ground_truth, preds)):
            gt_onehot[i, int(gt)] = 1
            pred_onehot[i, int(pred)] = 1
        brier = brier_score_loss(gt_onehot.ravel(), pred_onehot.ravel())
        per_class_sensitivity, per_class_specificity = calculate_per_class_metrics(ground_truth, preds, num_classes)
        metrics = {
            'Accuracy': acc,
            'F1_Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Kappa': kappa,
            'AUC': auc if auc is not None else 0,
            'Brier_Score': brier,
        }
        for i, cname in enumerate(class_names):
            metrics[f'Sensitivity_{cname}'] = per_class_sensitivity[i]
        for i, cname in enumerate(class_names):
            metrics[f'Specificity_{cname}'] = per_class_specificity[i]
        # Add detection (OA/None OA) metrics
        metrics.update(compute_detection_metrics(ground_truth, preds))
        all_metrics[col] = metrics

    # Evaluate refined models
    for col in model_cols:
        refined_col = f'{col}_refined'
        preds = refined_df[refined_col].tolist()
        acc = accuracy_score(ground_truth, preds)
        f1 = f1_score(ground_truth, preds, average='weighted')
        precision = precision_score(ground_truth, preds, average='weighted', zero_division=0)
        recall = recall_score(ground_truth, preds, average='weighted')
        kappa = cohen_kappa_score(ground_truth, preds)
        sensitivity, specificity = calculate_sensitivity_specificity(ground_truth, preds)
        ground_truth_bin = np.array(ground_truth) >= 2
        pred_bin = np.array(preds) >= 2
        try:
            auc = roc_auc_score(ground_truth_bin, pred_bin)
        except Exception:
            auc = None
        gt_onehot = np.zeros((len(ground_truth), num_classes))
        pred_onehot = np.zeros((len(preds), num_classes))
        for i, (gt, pred) in enumerate(zip(ground_truth, preds)):
            gt_onehot[i, int(gt)] = 1
            pred_onehot[i, int(pred)] = 1
        brier = brier_score_loss(gt_onehot.ravel(), pred_onehot.ravel())
        per_class_sensitivity, per_class_specificity = calculate_per_class_metrics(ground_truth, preds, num_classes)
        metrics = {
            'Accuracy': acc,
            'F1_Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Kappa': kappa,
            'AUC': auc if auc is not None else 0,
            'Brier_Score': brier,
        }
        for i, cname in enumerate(class_names):
            metrics[f'Sensitivity_{cname}'] = per_class_sensitivity[i]
        for i, cname in enumerate(class_names):
            metrics[f'Specificity_{cname}'] = per_class_specificity[i]
        # Add detection (OA/None OA) metrics
        metrics.update(compute_detection_metrics(ground_truth, preds))
        all_metrics[refined_col] = metrics

    # Save all refined predictions to CSV
    output_csv = os.path.join(output_dir, f"refined_results.csv")
    refined_df.to_csv(output_csv, index=False)
    print(f"Refined results saved to: {output_csv}")

    # Save all metrics for all models to a single DataFrame and CSV
    metric_names = [
        'Accuracy', 'F1_Score', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'Kappa', 'AUC', 'Brier_Score',
        'Sensitivity_None', 'Sensitivity_Doubtful', 'Sensitivity_Mild', 'Sensitivity_Moderate', 'Sensitivity_Severe',
        'Specificity_None', 'Specificity_Doubtful', 'Specificity_Mild', 'Specificity_Moderate', 'Specificity_Severe',
        # Detection metrics
        'Detection_Accuracy', 'Detection_F1_Score', 'Detection_Precision', 'Detection_Recall', 'Detection_Kappa', 'Detection_AUC', 'Detection_Brier_Score',
        'Detection_Sensitivity', 'Detection_Specificity'
    ]
    metrics_df = pd.DataFrame(index=metric_names)
    for model, metrics in all_metrics.items():
        rounded_metrics = {k: round(v, 4) for k, v in metrics.items()}
        metrics_df[model] = pd.Series(rounded_metrics)
    metrics_csv = os.path.join(output_dir, "all_models_metrics_refined.csv")
    metrics_df.to_csv(metrics_csv, float_format='%.4f')
    print(f"All models' refined metrics saved to {metrics_csv}")

    # Print only the requested model's metrics (original and refined)
    for suffix in ["", "_refined"]:
        col = model_name + suffix
        if col in all_metrics:
            print(f"\nMetrics for {col}:")
            for k, v in all_metrics[col].items():
                print(f"{k}: {v:.4f}")
            # Save confusion matrix for this model
            y_pred = refined_df[col].tolist() if suffix else df[col].tolist()
            save_confusion_matrix(
                ground_truth,
                y_pred,
                class_names,
                output_dir=os.path.dirname(csv_path),
                filename_prefix=f"{col}_"
            )
            # Save detection confusion matrix
            save_detection_confusion_matrix(
                ground_truth,
                y_pred,
                output_dir=os.path.dirname(csv_path),
                filename_prefix=f"{col}_"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine model predictions using YOLO output and compute metrics.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to all_models_predictions_yolo.csv")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model column to refine (e.g., 'resnet50')")
    parser.add_argument('--config', type=str, default='default.yaml', help="Name of the configuration file in config folder")
    args = parser.parse_args()
    
    main(csv_path=args.csv_path, model_name=args.model_name, config=args.config)