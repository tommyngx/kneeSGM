import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import argparse
from sklearn.metrics import classification_report
from .models.model_architectures import get_model
from .data.data_loader import get_dataloader
from .utils.metrics import accuracy, f1, precision, recall
from .utils.gradcam import GradCAM, show_cam_on_image
from .utils.plotting import save_confusion_matrix, save_roc_curve

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_random_predictions(model, dataloader, device, output_dir, class_names):
    model.eval()
    grad_cam = GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=torch.cuda.is_available())
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
        grayscale_cam = grad_cam(input_tensor=images[i].unsqueeze(0), target_category=pred)[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        plt.imsave(os.path.join(output_dir, f"prediction_img_{i}_pred_{class_names[pred]}_label_{class_names[label]}.png"), cam_image)

def test(model, dataloader, device):
    model.eval()
    running_acc = 0.0
    running_f1 = 0.0
    running_precision = 0.0
    running_recall = 0.0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)  # Apply softmax
            running_acc += accuracy(outputs, labels)
            running_f1 += f1(outputs, labels)
            running_precision += precision(outputs, labels)
            running_recall += recall(outputs, labels)
    return running_acc / len(dataloader), running_f1 / len(dataloader), running_precision / len(dataloader), running_recall / len(dataloader)

def main(config_path='config/default.yaml', model_name=None):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name is None:
        model_name = config['model']['name']
    
    model = get_model(model_name, config_path=config_path, pretrained=config['model']['pretrained'])
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], "models", "best_model.pth")))
    model = model.to(device)
    
    test_loader = get_dataloader('test', config['data']['batch_size'], config['data']['num_workers'], config_path=config_path)
    
    test_acc, test_f1, test_precision, test_recall = test(model, test_loader, device)
    
    output_dir = os.path.join(config['output_dir'], "final_logs")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
    
    with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
    
    save_confusion_matrix(test_loader.dataset.data.iloc[:, 1].values, test_loader.dataset.data.iloc[:, 0].values, config['data']['class_names'], output_dir)
    save_roc_curve(test_loader.dataset.data.iloc[:, 1].values, test_loader.dataset.data.iloc[:, 0].values, config['data']['class_names'], output_dir)
    save_random_predictions(model, test_loader, device, output_dir, config['data']['class_names'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a model for knee osteoarthritis classification.')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file.')
    parser.add_argument('--model', type=str, help='Model name to use for testing.')
    args = parser.parse_args()
    
    main(config_path=args.config, model_name=args.model)
