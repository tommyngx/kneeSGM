import os
import argparse
import random
import cv2
import torch
from torchvision import transforms
from models.model_architectures import get_model

def load_random_image(dataset_location):
    images = [img for img in os.listdir(dataset_location) if img.endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        raise FileNotFoundError("No images found in the dataset location.")
    
    random_image = random.choice(images)
    image_path = os.path.join(dataset_location, random_image)
    img = cv2.imread(image_path)
    return img, image_path

def preprocess_image(img, image_size):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def predict(model, img_tensor, device):
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
    return output

def main(dataset_location, model_name, config_path='config/default.yaml'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_model(model_name, config_path=config_path, pretrained=True)
    model = model.to(device)
    
    # Load and preprocess image
    img, image_path = load_random_image(dataset_location)
    img_tensor = preprocess_image(img, image_size=224)
    
    # Predict
    output = predict(model, img_tensor, device)
    
    # Print output
    print(f"Image path: {image_path}")
    print(f"Model output: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a model on a random image from the dataset")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset location')
    parser.add_argument('--model', type=str, required=True, help='Model name to use for prediction')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    main(args.dataset_location, args.model, args.config)
