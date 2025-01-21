import os
import argparse
import cv2
from ultralytics import YOLO
import yaml
from tqdm import tqdm
from datetime import datetime
import random

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

def save_labels(img, box, name, image_path, output_dir, class_id):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.basename(image_path).split('.')[0]
    label_path = os.path.join(output_dir, f"{base_name}.txt")
    
    img_height, img_width = img.shape[:2]
    
    with open(label_path, 'w') as label_file:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def process_images(dataset_location, model, output_dir, source_type, dataX, class_id):
    if source_type == 'random':
        img, image_path = load_random_image(dataset_location, dataX)
        results = model(img, verbose=False)
        boxes = results[0].boxes
        names = results[0].names
        if boxes:
            save_labels(img, boxes[0], names[0], image_path, output_dir, class_id)
        
        # Print output
        print(f"Image path: {image_path}")
        print("Bounding box and name:")
        if boxes:
            print(f"Box: {boxes[0].xyxy}, Name: {names[0]}")
    else:
        image_paths = load_image_paths(dataset_location, dataX)
        for image_path in tqdm(image_paths, desc="Processing images", total=len(image_paths)):
            img = cv2.imread(image_path)
            results = model(img, verbose=False)
            boxes = results[0].boxes
            names = results[0].names
            if boxes:
                save_labels(img, boxes[0], names[0], image_path, output_dir, class_id)

def main(dataset_location, model_path, source_type, dataX='VOS', config_path='config/default.yaml', class_id=4):
    config = load_config(config_path)
    folder_name = os.path.basename(dataset_location) + "_" + datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(config['output_dir'], 'yolo', 'runs', 'labels', folder_name)
    
    # Load model
    model = YOLO(model_path)
    
    # Process images
    process_images(dataset_location, model, output_dir, source_type, dataX, class_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a YOLO model on images from the dataset and save YOLO format labels")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset location')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--source_type', type=str, choices=['random', 'folder'], default='random', help='Source type: random image or whole folder')
    parser.add_argument('--dataX', type=str, default='VOS', help='Data source identifier')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file')
    parser.add_argument('--class_id', type=int, default=4, help='Class ID for the bounding box')
    args = parser.parse_args()
    
    main(args.dataset_location, args.model, args.source_type, args.dataX, args.config, args.class_id)
