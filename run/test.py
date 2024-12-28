import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import argparse
from sklearn.metrics import classification_report
from models.model_architectures import get_model
from data.data_loader import get_dataloader
from utils.metrics import accuracy, f1, precision, recall
from utils.gradcam import generate_gradcam, show_cam_on_image, get_target_layer
from utils.plotting import save_confusion_matrix, save_roc_curve

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_random_predictions(model, dataloader, device, output_dir, class_names, target_layer, acc=None):
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    outputs = torch.softmax(outputs, dim=1)  # Apply softmax
    preds = torch.argmax(outputs, dim=1)
    for i in range(12):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        label = labels[i].item()
        pred = preds[i].item()
        heatmap = generate_gradcam(model, images[i].unsqueeze(0), target_layer)
        cam_image = show_cam_on_image(img, heatmap, use_rgb=True)
        plt.imsave(os.path.join(output_dir, f"prediction_img_{i}_pred_{class_names[pred]}_label_{class_names[label]}.png"), cam_image)

def test(model, dataloader, device):
    model.eval()
    running_acc = 0.0
    running_f1 = 0.0
    running_precision = 0.0
    running_recall = 0.0
    all_preds = []
    all_labels = []
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
    return running_acc / len(dataloader), running_f1 / len(dataloader), running_precision / len(dataloader), running_recall / len(dataloader), all_preds, all_labels

def main(config_path='config/default.yaml', model_name=None, model_path=None):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name is None:
        model_name = config['model']['name']
    
    model = get_model(model_name, config_path=config_path, pretrained=config['model']['pretrained'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    test_loader = get_dataloader('test', config['data']['batch_size'], config['data']['num_workers'], config_path=config_path)
    
    # Print details before testing
    print(f"Model: {model_name}")
    print(f"Number of classes: {len(config['data']['class_labels'])}")
    print(f"Class names: {config['data']['class_names']}")
    print(f"Number of test images: {len(test_loader.dataset)}")
    
    test_acc, test_f1, test_precision, test_recall, test_preds, test_labels = test(model, test_loader, device)
    
    output_dir = os.path.join(config['output_dir'], "final_logs")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
    
    with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
    
    target_layer = get_target_layer(model, model_name)
    
    save_confusion_matrix(test_labels, test_preds, config['data']['class_names'], output_dir)
    save_roc_curve(test_labels, test_preds, config['data']['class_names'], output_dir)
    save_random_predictions(model, test_loader, device, output_dir, config['data']['class_names'], target_layer, acc=test_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a model for knee osteoarthritis classification.')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file.')
    parser.add_argument('--model', type=str, help='Model name to use for testing.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint to load.')
    args = parser.parse_args()
    
    main(config_path=args.config, model_name=args.model, model_path=args.model_path)
