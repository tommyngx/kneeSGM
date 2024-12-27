import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import os
import datetime
import pytz
import argparse
from sklearn.metrics import classification_report
import numpy as np
from models.model_architectures import get_model
from data.data_loader import get_dataloader
from utils.metrics import accuracy, f1, precision, recall
from utils.gradcam import generate_gradcam, show_cam_on_image
from utils.plotting import save_confusion_matrix, save_roc_curve, tr_plot

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
        print(f"Generating Grad-CAM for image {i}, label: {label}, pred: {pred}")
        heatmap = generate_gradcam(model, images[i].unsqueeze(0), model.layer4[-1])
        cam_image = show_cam_on_image(img, heatmap, use_rgb=True)
        plt.imsave(os.path.join(output_dir, f"prediction_epoch_{epoch}_img_{i}_pred_{class_names[pred]}_label_{class_names[label]}.png"), cam_image)

def main(config_path='config/default.yaml', model_name=None, epochs=None, resume_from=None):
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
    
    # Print details before training
    print(f"Model: {model_name}")
    print(f"Number of epochs: {epochs}")
    print(f"Number of classes: {len(config['data']['class_labels'])}")
    print(f"Class names: {config['data']['class_names']}")
    print(f"Number of training images: {len(train_loader.dataset)}")
    print(f"Number of validation images: {len(val_loader.dataset)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    start_epoch = 0
    best_val_acc = 0.0
    training_history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    best_models = []
    
    if resume_from:
        checkpoint = torch.load(resume_from, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        training_history = checkpoint['training_history']
        print(f"Resuming training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        training_history['accuracy'].append(train_acc)
        training_history['loss'].append(train_loss)
        training_history['val_accuracy'].append(val_acc)
        training_history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if (epoch + 1) % 2 == 0:
            print("Classification Report:")
            print(classification_report(val_labels, val_preds, target_names=config['data']['class_names'], zero_division=0))
            save_confusion_matrix(val_labels, val_preds, config['data']['class_names'], os.path.join(output_dir, "logs"), epoch)
            save_roc_curve(val_labels, val_preds, config['data']['class_names'], os.path.join(output_dir, "logs"), epoch)
            save_random_predictions(model, val_loader, device, os.path.join(output_dir, "logs"), epoch, config['data']['class_names'])
        
        if epoch >= 1:
            tr_plot(training_history, 0)
            plt.savefig(os.path.join(output_dir, "logs", f"training_plot_epoch_{epoch+1}.png"))
        
        with open(os.path.join(output_dir, "logs", "training_log.txt"), "a") as f:
            f.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_filename = f"{model_name}_epoch_{epoch+1}_acc_{val_acc:.4f}.pth"
            model_path = os.path.join(output_dir, "models", model_filename)
            torch.save(model.state_dict(), model_path)
            best_models.append((val_acc, model_path))
            best_models = sorted(best_models, key=lambda x: x[0], reverse=True)[:4]
            print("Best model saved!")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'training_history': training_history
        }
        torch.save(checkpoint, os.path.join(output_dir, "models", f"checkpoint_epoch_{epoch+1}.pth"))
        
        # Remove models beyond the top 4
        for _, model_path in best_models[4:]:
            if os.path.exists(model_path):
                os.remove(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for knee osteoarthritis classification.')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file.')
    parser.add_argument('--model', type=str, help='Model name to use for training.')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--resume_from', type=str, help='Path to the checkpoint to resume training from.')
    args = parser.parse_args()
    
    main(config_path=args.config, model_name=args.model, epochs=args.epochs, resume_from=args.resume_from)
