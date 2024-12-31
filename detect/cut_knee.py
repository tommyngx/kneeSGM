import os
import argparse
import random
import cv2
from ultralytics import YOLO
import yaml
from tqdm import tqdm

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

def adjust_bounding_box(x1, y1, x2, y2, img_width, img_height):
    box_width = x2 - x1
    box_height = y2 - y1
    max_dim = max(box_width, box_height)
    
    center_x = x1 + box_width // 2
    center_y = y1 + box_height // 2
    
    new_x1 = max(center_x - max_dim // 2, 0)
    new_y1 = max(center_y - max_dim // 2, 0)
    new_x2 = min(center_x + max_dim // 2, img_width)
    new_y2 = min(center_y + max_dim // 2, img_height)
    
    return new_x1, new_y1, new_x2, new_y2

def save_cropped_images(img, boxes, names, image_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.basename(image_path).split('.')[0]
    img_height, img_width = img.shape[:2]
    
    # Keep only the bounding box with the highest confidence for each name
    unique_boxes = {}
    for i, box in enumerate(boxes):
        class_id = int(box.cls.item())
        name = names[class_id]
        conf = box.conf.item()
        if name not in unique_boxes or conf > unique_boxes[name][1]:
            unique_boxes[name] = (box, conf)
    
    for name, (box, _) in unique_boxes.items():
        if base_name == "58P2F2036KNEE01" and name == "Deformity":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1, x2, y2 = adjust_bounding_box(x1, y1, x2, y2, img_width, img_height)
        cropped_img = img[y1:y2, x1:x2]
        output_path = os.path.join(output_dir, f"{base_name}_{name[0]}.png")
        cv2.imwrite(output_path, cropped_img)

def process_images(dataset_location, model, output_dir, source_type, dataX):
    ignore_images = [
        "76P2F2152KNEE02.png", "76P2F2152KNEE01.png", "73P2F0268KNEE01.png", 
        "66P2M4404KNEE01.png", "59P2M4280KNEE01.png", "58P2F2036KNEE01.png", 
        "47P2F0352KNEE02.png"
    ]
    
    if source_type == 'random':
        img, image_path = load_random_image(dataset_location, dataX)
        results = model(img, verbose=False)
        boxes = results[0].boxes
        names = results[0].names
        save_cropped_images(img, boxes, names, image_path, output_dir)
        
        # Print output
        print(f"Image path: {image_path}")
        print("Bounding boxes and names:")
        for box in boxes:
            class_id = int(box.cls.item())
            print(f"Box: {box.xyxy}, Name: {names[class_id]}")
    else:
        image_paths = load_image_paths(dataset_location, dataX)
        for image_path in tqdm(image_paths, desc="Processing images", total=len(image_paths)):
            if os.path.basename(image_path) in ignore_images:
                continue
            img = cv2.imread(image_path)
            results = model(img, verbose=False)
            boxes = results[0].boxes
            names = results[0].names
            save_cropped_images(img, boxes, names, image_path, output_dir)

def main(dataset_location, model_path, source_type, dataX='VOS', config_path='config/default.yaml'):
    config = load_config(config_path)
    output_dir = os.path.join(config['output_dir'], 'yolo', 'runs', dataX.lower())
    
    # Load model
    model = YOLO(model_path)
    
    # Process images
    process_images(dataset_location, model, output_dir, source_type, dataX)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a YOLO model on images from the dataset and save cropped bounding boxes")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset location')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--source_type', type=str, choices=['random', 'folder'], default='random', help='Source type: random image or whole folder')
    parser.add_argument('--dataX', type=str, default='VOS', help='Data source identifier')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    main(args.dataset_location, args.model, args.source_type, args.dataX, args.config)
