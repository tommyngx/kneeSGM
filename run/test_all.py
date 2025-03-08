import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class.")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import glob
import matplotlib.pyplot as plt
import argparse
import csv
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, roc_auc_score, brier_score_loss
from models.model_architectures import get_model
from data.data_loader import get_dataloader
from data.preprocess import get_transforms
from utils.metrics import accuracy, f1, precision, recall
from utils.gradcam import save_random_predictions, get_target_layer
from utils.plotting import save_confusion_matrix, save_roc_curve
import numpy as np
import re

# Fix for the import error: Use full module path instead of relative import
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run.test import load_config, test, calculate_sensitivity_specificity, calculate_per_class_metrics

def extract_model_name(filename):
    """Extract the model name from the filename (before _epoch)."""
    match = re.search(r'(.+?)_epoch', os.path.basename(filename))
    if match:
        return match.group(1)
    return None

def get_image_info(dataloader):
    """Extract image names and paths from the test dataloader.
    If the underlying dataset has attributes (data, image_path_column, dataset_based_link),
    use them to construct the full image paths; otherwise, fallback to indices.
    """
    dataset = dataloader.dataset
    # Check for custom attributes in the dataset
    if hasattr(dataset, 'data') and hasattr(dataset, 'image_path_column') and hasattr(dataset, 'dataset_based_link'):
        try:
            # Get the image file names from the DataFrame column
            image_files = dataset.data[dataset.image_path_column].tolist()
            image_paths = [os.path.join(dataset.dataset_based_link, str(fname)) for fname in image_files]
            image_names = [os.path.basename(path) for path in image_paths]
            # Validate the number of images match dataset length
            if len(image_names) == len(dataset):
                return image_names, image_paths
        except Exception as e:
            print(f"Error extracting image info from custom attributes: {e}")
    
    # Fallback: try to use existing dataset attributes if available
    if hasattr(dataset, 'imgs'):
        image_names = [os.path.basename(path) for path, _ in dataset.imgs]
        image_paths = [path for path, _ in dataset.imgs]
        return image_names, image_paths
    elif hasattr(dataset, 'samples'):
        image_names = [os.path.basename(path) for path, _ in dataset.samples]
        image_paths = [path for path, _ in dataset.samples]
        return image_names, image_paths
    elif hasattr(dataset, 'image_paths'):
        image_names = [os.path.basename(path) for path in dataset.image_paths]
        image_paths = list(dataset.image_paths)
        return image_names, image_paths
    
    # Fallback to indices if nothing else works
    print("Warning: Could not extract image paths from dataset. Using indices as fallback.")
    image_names = [f"image_{i}" for i in range(len(dataset))]
    image_paths = [f"index_{i}" for i in range(len(dataset))]
    return image_names, image_paths

def test_model(model_path, model_name, config_path, output_dir, use_gradcam_plus_plus=False):
    """Test a single model and return metrics and predictions."""
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = get_model(model_name, config_path=config_path, pretrained=config['model']['pretrained'])
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Prepare dataloader
    _, test_transform = get_transforms(config['data']['image_size'], config_path=config_path)
    test_loader = get_dataloader('test', config['data']['batch_size'], config['data']['num_workers'], 
                               transform=test_transform, config_path=config_path)
    
    # Print model details
    print(f"\nTesting model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Number of test images: {len(test_loader.dataset)}")
    
    # Get image names and paths
    image_names, image_paths = get_image_info(test_loader)
    
    # Evaluate the model
    test_acc, test_f1, test_precision, test_recall, test_preds, test_labels, test_outputs = test(model, test_loader, device)
    
    # Calculate additional metrics
    sensitivity, specificity = calculate_sensitivity_specificity(test_labels, test_preds)
    per_class_sensitivity, per_class_specificity = calculate_per_class_metrics(test_labels, test_preds, len(config['data']['class_labels']))
    
    kappa = cohen_kappa_score(test_labels, test_preds)
    
    binary_labels = np.array(test_labels) >= 2
    binary_scores = np.array(test_outputs)[:, 2:].sum(axis=1)
    auc = roc_auc_score(binary_labels, binary_scores)
    
    n_classes = len(config['data']['class_labels'])
    y_onehot = np.zeros((len(test_labels), n_classes))
    for i, label in enumerate(test_labels):
        y_onehot[i, label] = 1
    brier = brier_score_loss(y_onehot.ravel(), np.array(test_outputs).ravel())
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Print metrics
    print(f"Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Kappa: {kappa:.4f}, AUC: {auc:.4f}")
    
    # Save confusion matrix and ROC curve with model name
    save_confusion_matrix(test_labels, test_preds, config['data']['class_names'], 
                         model_output_dir, epoch=0, acc=test_acc, filename_prefix=f"{model_name}_")
    
    test_outputs_np = np.array(test_outputs)
    positive_risk = test_outputs_np[:, 2:].sum(axis=1)
    save_roc_curve(test_labels, positive_risk, config['data']['class_names'], 
                  model_output_dir, filename_prefix=f"{model_name}_")
    
    # Generate and save GradCAM visualizations
    target_layer = get_target_layer(model, model_name)
    save_random_predictions(model, test_loader, device, model_output_dir, 
                           epoch=0, class_names=config['data']['class_names'],
                           use_gradcam_plus_plus=use_gradcam_plus_plus, 
                           target_layer=target_layer, model_name=model_name,
                           acc=test_acc)
    
    # Collect metrics in a dictionary
    metrics = {
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
        metrics[f'Sensitivity_{class_name}'] = per_class_sensitivity[i]
    
    for i, class_name in enumerate(config['data']['class_names']):
        metrics[f'Specificity_{class_name}'] = per_class_specificity[i]
    
    # Return metrics and predictions for each image
    return metrics, test_preds, test_labels, image_names, image_paths

def save_all_predictions_to_csv(all_predictions, output_dir):
    """Save all model predictions for each image to a CSV file."""
    if not all_predictions:
        print("No predictions to save")
        return
    
    # Create a DataFrame with image info
    first_model = next(iter(all_predictions.values()))
    image_names = first_model['image_names']
    image_paths = first_model['image_paths']
    ground_truth = first_model['labels']
    
    # Create DataFrame with image information
    df = pd.DataFrame({
        'image_name': image_names,
        'image_path': image_paths,
        'ground_truth': ground_truth
    })
    
    # Add predictions from each model as new columns
    for model_name, model_data in all_predictions.items():
        df[f'{model_name}'] = model_data['predictions']
    
    # Save the DataFrame to CSV
    csv_path = os.path.join(output_dir, "all_models_predictions.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"All models' predictions saved to {csv_path}")

def main(config='default.yaml', models_dir=None, use_gradcam_plus_plus=False):
    # Load configuration
    config_path = os.path.join('config', config)
    config = load_config(config_path)
    
    # Set output directory
    dataset_name =config['data']['dataset_name']
    output_dir = os.path.join(config['output_dir'], "final_logs_all_{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all model checkpoints in the specified directory
    if models_dir is None:
        models_dir = os.path.join(config['output_dir'], "checkpoints")
    
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    
    if not model_files:
        print(f"No model files found in {models_dir}")
        return
    
    print(f"Found {len(model_files)} model files.")
    
    # Dictionary to store all metrics for all models
    all_metrics = {}
    # Dictionary to store all predictions for all models
    all_predictions = {}
    
    # Test each model
    for model_file in model_files:
        model_name = extract_model_name(model_file)
        if model_name is None:
            print(f"Could not extract model name from {model_file}, skipping...")
            continue
        
        metrics, predictions, labels, image_names, image_paths = test_model(model_file, model_name, config_path, output_dir, use_gradcam_plus_plus)
        all_metrics[model_name] = metrics
        all_predictions[model_name] = {
            'predictions': predictions,
            'labels': labels,
            'image_names': image_names,
            'image_paths': image_paths
        }
    
    # Save consolidated metrics to CSV
    save_consolidated_metrics(all_metrics, output_dir)
    
    # Save all model predictions to CSV
    save_all_predictions_to_csv(all_predictions, output_dir)
    
    print(f"All models tested. Results saved in {output_dir}")

def save_consolidated_metrics(all_metrics, output_dir):
    """Save metrics from all models in a single CSV file with models as columns."""
    if not all_metrics:
        print("No metrics to save")
        return
    
    # Get a list of all metric names from the first model
    first_model = next(iter(all_metrics.values()))
    metric_names = list(first_model.keys())
    
    # Create a DataFrame with metric names as the index
    df = pd.DataFrame(index=metric_names)
    
    # Add a column for each model
    for model_name, metrics in all_metrics.items():
        # Round all metrics to 4 decimal places
        rounded_metrics = {k: round(v, 4) for k, v in metrics.items()}
        df[model_name] = pd.Series(rounded_metrics)
    
    # Ensure all numeric values are rounded to 4 decimal places in the DataFrame
    df = df.round(4)
    
    # Save the DataFrame to CSV with float_format to ensure consistent decimal places
    csv_path = os.path.join(output_dir, "all_models_metrics.csv")
    df.to_csv(csv_path, float_format='%.4f')
    
    print(f"Consolidated metrics saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test all models in a directory for knee osteoarthritis classification.')
    parser.add_argument('--config', type=str, default='default.yaml', help='Name of the configuration file.')
    parser.add_argument('--models_dir', type=str, help='Directory containing model checkpoints.')
    parser.add_argument('--use_gradcam_plus_plus', action='store_true', help='Use Grad-CAM++ instead of Grad-CAM.')
    args = parser.parse_args()
    
    main(config=args.config, models_dir=args.models_dir, use_gradcam_plus_plus=args.use_gradcam_plus_plus)
