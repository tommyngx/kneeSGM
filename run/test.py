import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class.")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import matplotlib.pyplot as plt
import argparse
import csv
from sklearn.metrics import classification_report, cohen_kappa_score, roc_auc_score, brier_score_loss
from models.model_architectures import get_model
from data.data_loader import get_dataloader
from data.preprocess import get_transforms
from utils.metrics import accuracy, f1, precision, recall
from utils.gradcam import save_random_predictions, get_target_layer
from utils.plotting import save_confusion_matrix, save_roc_curve
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def test(model, dataloader, device):
    model.eval()
    running_acc = 0.0
    running_f1 = 0.0
    running_precision = 0.0
    running_recall = 0.0
    all_preds = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)  # Apply softmax
            running_acc += accuracy(outputs, labels)
            running_f1 += f1(outputs, labels)
            running_precision += precision(outputs, labels)
            running_recall += recall(outputs, labels)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    return running_acc / len(dataloader), running_f1 / len(dataloader), running_precision / len(dataloader), running_recall / len(dataloader), all_preds, all_labels, all_outputs

def calculate_sensitivity_specificity(y_true, y_pred):
    # For multi-class, convert to binary (OA vs no OA)
    # Consider classes 0,1 as negative (no/mild OA) and 2,3,4 as positive (moderate/severe OA)
    y_true_binary = np.array(y_true) >= 2
    y_pred_binary = np.array(y_pred) >= 2
    
    # True positives, etc
    tp = np.sum((y_true_binary == True) & (y_pred_binary == True))
    tn = np.sum((y_true_binary == False) & (y_pred_binary == False))
    fp = np.sum((y_true_binary == False) & (y_pred_binary == True))
    fn = np.sum((y_true_binary == True) & (y_pred_binary == False))
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity

def calculate_per_class_metrics(y_true, y_pred, num_classes):
    """Calculate sensitivity and specificity for each class using one-vs-rest approach"""
    per_class_sensitivity = []
    per_class_specificity = []
    
    for cls in range(num_classes):
        # Convert to binary classification problem (current class vs the rest)
        y_true_binary = np.array(y_true) == cls
        y_pred_binary = np.array(y_pred) == cls
        
        # Calculate TP, TN, FP, FN
        tp = np.sum((y_true_binary == True) & (y_pred_binary == True))
        tn = np.sum((y_true_binary == False) & (y_pred_binary == False))
        fp = np.sum((y_true_binary == False) & (y_pred_binary == True))
        fn = np.sum((y_true_binary == True) & (y_pred_binary == False))
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        per_class_sensitivity.append(sensitivity)
        per_class_specificity.append(specificity)
    
    return per_class_sensitivity, per_class_specificity

def save_metrics_to_csv(metrics_dict, output_dir, model_name):
    """Save metrics to a CSV file with metrics in the first column and model name as the second column header."""
    csv_path = os.path.join(output_dir, "metrics_results.csv")
    
    # Round all metrics to 4 decimal places
    rounded_metrics = {k: round(v, 4) for k, v in metrics_dict.items()}
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Use model_name as column header instead of 'Value' and remove the 'Model' column
        writer.writerow(['Metric', model_name])
        
        for metric_name, metric_value in rounded_metrics.items():
            writer.writerow([metric_name, f"{metric_value:.4f}"])
    
    print(f"Metrics saved to {csv_path}")

def main(config='default.yaml', model_name=None, model_path=None, use_gradcam_plus_plus=False):
    config_path = os.path.join('config', config)
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name is None:
        model_name = config['model']['name']
    
    model = get_model(model_name, config_path=config_path, pretrained=config['model']['pretrained'])
    # Use weights_only=True for better security when loading models
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    _, test_transform = get_transforms(config['data']['image_size'], config_path=config_path)
    test_loader = get_dataloader('test', config['data']['batch_size'], config['data']['num_workers'], transform=test_transform, config_path=config_path)
    
    # Print details before testing
    print(f"Model: {model_name}")
    print(f"Number of classes: {len(config['data']['class_labels'])}")
    print(f"Class names: {config['data']['class_names']}")
    print(f"Number of test images: {len(test_loader.dataset)}")
    
    test_acc, test_f1, test_precision, test_recall, test_preds, test_labels, test_outputs = test(model, test_loader, device)
    
    # Calculate additional metrics
    sensitivity, specificity = calculate_sensitivity_specificity(test_labels, test_preds)
    per_class_sensitivity, per_class_specificity = calculate_per_class_metrics(test_labels, test_preds, len(config['data']['class_labels']))
    
    # Cohen's Kappa - measures agreement between predicted and actual classes
    kappa = cohen_kappa_score(test_labels, test_preds)
    
    # AUC - for multi-class, we compute one-vs-rest ROC AUC
    # Convert to binary labels for AUC calculation (OA severity >= 2 is positive)
    binary_labels = np.array(test_labels) >= 2
    binary_scores = np.array(test_outputs)[:, 2:].sum(axis=1)  # Sum probabilities for classes 2,3,4
    auc = roc_auc_score(binary_labels, binary_scores)
    
    # Brier score - measures accuracy of probabilistic predictions
    # For multi-class, use one-hot encoding and mean Brier score
    n_classes = len(config['data']['class_labels'])
    y_onehot = np.zeros((len(test_labels), n_classes))
    for i, label in enumerate(test_labels):
        y_onehot[i, label] = 1
    brier = brier_score_loss(y_onehot.ravel(), np.array(test_outputs).ravel())
    
    output_dir = os.path.join(config['output_dir'], "final_logs", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Kappa: {kappa:.4f}, AUC: {auc:.4f}, Brier Score: {brier:.4f}")
    
    # Print per-class sensitivity and specificity
    for i, class_name in enumerate(config['data']['class_names']):
        print(f"Class {class_name} - Sensitivity: {per_class_sensitivity[i]:.4f}, Specificity: {per_class_specificity[i]:.4f}")
    
    with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Kappa: {kappa:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Brier Score: {brier:.4f}\n")
        
        # Add per-class metrics to the output file
        #f.write("\nPer-Class Metrics:\n")
        #for i, class_name in enumerate(config['data']['class_names']):
        #    f.write(f"Class {class_name} - Sensitivity: {per_class_sensitivity[i]:.4f}, Specificity: {per_class_specificity[i]:.4f}\n")
        
    # Collect all metrics in a dictionary for CSV export
    metrics_dict = {
        'Accuracy': test_acc,
        'F1_Score': test_f1,
        'Precision': test_precision,
        'Recall': test_recall,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Kappa': kappa,
        'AUC': auc,
        'Brier_Score': brier,
    }
    
    # Add class-specific metrics
    for i, class_name in enumerate(config['data']['class_names']):
        metrics_dict[f'Sensitivity_{class_name}'] = per_class_sensitivity[i]
    
    for i, class_name in enumerate(config['data']['class_names']):
        metrics_dict[f'Specificity_{class_name}'] = per_class_specificity[i]

    # Save metrics to CSV
    save_metrics_to_csv(metrics_dict, output_dir, model_name)
    
    target_layer = get_target_layer(model, model_name)
    
    # Ensure the same parameters are used for save_confusion_matrix
    save_confusion_matrix(test_labels, test_preds, config['data']['class_names'], output_dir, epoch=0, acc=test_acc)
    
    # Calculate risk percentages for ROC curve
    test_outputs = np.array(test_outputs)
    positive_risk = test_outputs[:, 2:].sum(axis=1)  # Sum of class 2, 3, 4
    negative_risk = test_outputs[:, :2].sum(axis=1)  # Sum of class 0, 1
    save_roc_curve(test_labels, positive_risk, config['data']['class_names'], output_dir)
    
    save_random_predictions(model, test_loader, device, output_dir, epoch=0, class_names=config['data']['class_names'], use_gradcam_plus_plus=use_gradcam_plus_plus, target_layer=target_layer, model_name=model_name)
    
    # Save test outputs
    #np.save(os.path.join(output_dir, "test_outputs.npy"), test_outputs)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=config['data']['class_names'], zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a model for knee osteoarthritis classification.')
    parser.add_argument('--config', type=str, default='default.yaml', help='Name of the configuration file.')
    parser.add_argument('--model', type=str, help='Model name to use for testing.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint to load.')
    args = parser.parse_args()
    
    main(config=args.config, model_name=args.model, model_path=args.model_path)
