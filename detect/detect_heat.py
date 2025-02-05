import os
import argparse
import cv2
from ultralytics import YOLO
from ultralytics.solutions import heatmap
import yaml
from tqdm import tqdm
from datetime import datetime
import random
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_random_image(dataset_location, dataX):
    if dataX == 'CGMH':
        subfolders = [os.path.join(dataset_location, str(i)) for i in range(5)]
        images = [os.path.join(subfolder, img) for subfolder in subfolders for img in os.listdir(subfolder) if img.endswith(('.jpg', '.jpeg', '.png'))]
    else:
        images = [os.path.join(dataset_location, img) for img in os.listdir(dataset_location) if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        raise FileNotFoundError("No images found in the dataset location.")
    
    random_image = random.choice(images)
    img = cv2.imread(random_image)
    return img, random_image

def load_image_paths(dataset_location, dataX):
    if dataX == 'CGMH':
        subfolders = [os.path.join(dataset_location, str(i)) for i in range(5)]
        images = [os.path.join(subfolder, img) for subfolder in subfolders for img in os.listdir(subfolder) if img.endswith(('.jpg', '.jpeg', '.png'))]
    else:
        images = [os.path.join(dataset_location, img) for img in os.listdir(dataset_location) if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        raise FileNotFoundError("No images found in the dataset location.")
    return images

def create_heatmap_image(model, img):
    # Initialize heatmap object with default settings
    heatmap_obj = heatmap.Heatmap()
    
    # Configure heatmap settings
    heatmap_obj.set_args(
        colormap=cv2.COLORMAP_JET,
        imw=img.shape[1],
        imh=img.shape[0],
        view_img=False,
        heatmap_alpha=0.6,
        show=False
    )
    
    # Perform object detection with tracking
    results = model.track(img, persist=True, verbose=False)
    
    # Generate and return heatmap
    return heatmap_obj.generate_heatmap(img, tracks=results)

def save_combined_image(input_img, detected_img, heatmap_img, output_path):
    # Ensure all images are the same size
    h, w = input_img.shape[:2]
    detected_img = cv2.resize(detected_img, (w, h))
    heatmap_img = cv2.resize(heatmap_img, (w, h))
    
    # Combine images horizontally
    combined_img = np.hstack((input_img, detected_img, heatmap_img))
    cv2.imwrite(output_path, combined_img)

def process_images(dataset_location, model, output_dir, source_type, dataX):
    if source_type == 'random':
        img, image_path = load_random_image(dataset_location, dataX)
        results = model(img, verbose=False)
        detected_img = results[0].plot()
        heatmap_img = create_heatmap_image(model, img.copy())  # Use copy to prevent modifications
        
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        save_combined_image(img, detected_img, heatmap_img, output_path)
        
        print(f"Image path: {image_path}")
        print("Saved combined image to:", output_path)
    else:
        image_paths = load_image_paths(dataset_location, dataX)
        for image_path in tqdm(image_paths, desc="Processing images"):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            
            results = model(img, verbose=False)
            detected_img = results[0].plot()
            heatmap_img = create_heatmap_image(model, img.copy())
            
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            save_combined_image(img, detected_img, heatmap_img, output_path)

def main(dataset_location, model_path, source_type, dataX='VOS', config_path='config/default.yaml'):
    config = load_config(config_path)
    folder_name = os.path.basename(dataset_location) + "_" + datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(config['output_dir'], 'yolo', 'runs', 'heatmaps', folder_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model
    model = YOLO(model_path)
    
    # Process images
    process_images(dataset_location, model, output_dir, source_type, dataX)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a YOLO model on images from the dataset and save combined images with input, detected, and heatmap")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset location')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--source_type', type=str, choices=['random', 'folder'], default='random', help='Source type: random image or whole folder')
    parser.add_argument('--dataX', type=str, default='VOS', help='Data source identifier')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    main(args.dataset_location, args.model, args.source_type, args.dataX, args.config)
