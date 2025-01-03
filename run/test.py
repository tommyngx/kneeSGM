import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class.")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import classification_report
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

def main(config='default.yaml', model_name=None, model_path=None, use_gradcam_plus_plus=False):
    config_path = os.path.join('config', config)
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name is None:
        model_name = config['model']['name']
    
    model = get_model(model_name, config_path=config_path, pretrained=config['model']['pretrained'])
    checkpoint = torch.load(model_path, map_location=device)
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
    
    output_dir = os.path.join(config['output_dir'], "final_logs")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
    
    with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
    
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
    np.save(os.path.join(output_dir, "test_outputs.npy"), test_outputs)
    
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
