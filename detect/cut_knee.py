import os
import argparse
import random
import cv2
from ultralytics import YOLO
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_random_image(dataset_location):
    images = [img for img in os.listdir(dataset_location) if img.endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        raise FileNotFoundError("No images found in the dataset location.")
    
    random_image = random.choice(images)
    image_path = os.path.join(dataset_location, random_image)
    img = cv2.imread(image_path)
    return img, image_path

def save_cropped_images(img, boxes, names, image_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.basename(image_path).split('.')[0]
    for i, box in enumerate(boxes):
        class_id = int(box.cls.item())
        name = names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_img = img[y1:y2, x1:x2]
        output_path = os.path.join(output_dir, f"{base_name}_{name[0]}.png")
        cv2.imwrite(output_path, cropped_img)

def main(dataset_location, model_path, config_path='config/default.yaml'):
    config = load_config(config_path)
    output_dir = os.path.join(config['output_dir'], 'yolo', 'runs', 'processed')
    
    # Load model
    model = YOLO(model_path)
    
    # Load random image
    img, image_path = load_random_image(dataset_location)
    
    # Predict
    results = model(img, verbose=False)
    
    # Extract bounding boxes and names
    boxes = results[0].boxes
    names = results[0].names
    
    # Save cropped images
    save_cropped_images(img, boxes, names, image_path, output_dir)
    
    # Print output
    print(f"Image path: {image_path}")
    print("Bounding boxes and names:")
    for box in boxes:
        class_id = int(box.cls.item())
        print(f"Box: {box.xyxy}, Name: {names[class_id]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a YOLO model on a random image from the dataset and save cropped bounding boxes")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset location')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    main(args.dataset_location, args.model, args.config)
