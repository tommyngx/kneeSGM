import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from models.model_architectures import get_model
from data.data_loader import get_dataloader
from utils.gradcam import save_random_predictions

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path='config/default.yaml', model_name=None, model_path=None, use_gradcam_plus_plus=False):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name is None:
        model_name = config['model']['name']
    
    model = get_model(model_name, config_path=config_path, pretrained=config['model']['pretrained'])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    test_loader = get_dataloader('test', config['data']['batch_size'], config['data']['num_workers'], config_path=config_path)
    
    # Print details before generating Grad-CAM
    print(f"Model: {model_name}")
    print(f"Number of classes: {len(config['data']['class_labels'])}")
    print(f"Class names: {config['data']['class_names']}")
    print(f"Number of test images: {len(test_loader.dataset)}")
    
    output_dir = os.path.join(config['output_dir'], "gradcam_logs")
    os.makedirs(output_dir, exist_ok=True)
    
    save_random_predictions(model, test_loader, device, output_dir, epoch=0, class_names=config['data']['class_names'], use_gradcam_plus_plus=use_gradcam_plus_plus)

    # Load and display the saved image
    saved_image_path = os.path.join(output_dir, "random_predictions_epoch_0.png")
    if os.path.exists(saved_image_path):
        img = Image.open(saved_image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    # Keep only the last 3 latest saved epochs
    saved_files = sorted([f for f in os.listdir(output_dir) if f.startswith("random_predictions_epoch_")], reverse=True)
    for file in saved_files[3:]:
        os.remove(os.path.join(output_dir, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Grad-CAM for knee osteoarthritis classification.')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file.')
    parser.add_argument('--model', type=str, help='Model name to use for generating Grad-CAM.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint to load.')
    parser.add_argument('--use_gradcam_plus_plus', action='store_true', help='Use Grad-CAM++ instead of Grad-CAM.')
    args = parser.parse_args()
    
    main(config_path=args.config, model_name=args.model, model_path=args.model_path, use_gradcam_plus_plus=args.use_gradcam_plus_plus)
