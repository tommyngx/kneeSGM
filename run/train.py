import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import datetime
import pytz
import argparse
from sklearn.metrics import classification_report
import numpy as np
from ..models.model_architectures import get_model
from ..data.data_loader import get_dataloader
from ..utils.metrics import accuracy, f1, precision, recall
from ..utils.gradcam import GradCAM, show_cam_on_image
from ..utils.plotting import save_confusion_matrix, save_roc_curve

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_output_dirs(base_dir, timezone):
    tz = pytz.timezone(timezone)
    timestamp = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "final_logs"), exist_ok=True)
    return output_dir

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)
    return running_loss / len(dataloader), running_acc / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)  # Apply softmax
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / len(dataloader), running_acc / len(dataloader), all_preds, all_labels

def save_random_predictions(model, dataloader, device, output_dir, epoch, class_names):
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
        plt.imsave(os.path.join(output_dir, f"prediction_epoch_{epoch}_img_{i}_pred_{class_names[pred]}_label_{class_names[label]}.png"), cam_image)

def main(config_path='config/default.yaml', model_name=None, epochs=None):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = create_output_dirs(config['output_dir'], config['timezone'])
    
    if model_name is None:
        model_name = config['model']['name']
    if epochs is None:
        epochs = config['training']['epochs']
    
    model = get_model(model_name, config_path=config_path, pretrained=config['model']['pretrained'])
    model = model.to(device)
    
    train_loader = get_dataloader('train', config['data']['batch_size'], config['data']['num_workers'], config_path=config_path)
    val_loader = get_dataloader('val', config['data']['batch_size'], config['data']['num_workers'], config_path=config_path)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if (epoch + 1) % 5 == 0:
            print("Classification Report:")
            print(classification_report(val_labels, val_preds, target_names=config['data']['class_names']))
            save_confusion_matrix(val_labels, val_preds, config['data']['class_names'], os.path.join(output_dir, "logs"), epoch)
            save_roc_curve(val_labels, val_preds, config['data']['class_names'], os.path.join(output_dir, "logs"), epoch)
            save_random_predictions(model, val_loader, device, os.path.join(output_dir, "logs"), epoch, config['data']['class_names'])
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "models", "best_model.pth"))
            print("Best model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for knee osteoarthritis classification.')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file.')
    parser.add_argument('--model', type=str, help='Model name to use for training.')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train.')
    args = parser.parse_args()
    
    main(config_path=args.config, model_name=args.model, epochs=args.epochs)
