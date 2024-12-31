import os
import argparse
import random
import cv2
from ultralytics import YOLO

def load_random_image(dataset_location):
    images = [img for img in os.listdir(dataset_location) if img.endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        raise FileNotFoundError("No images found in the dataset location.")
    
    random_image = random.choice(images)
    image_path = os.path.join(dataset_location, random_image)
    img = cv2.imread(image_path)
    return img, image_path

def main(dataset_location, model_path):
    # Load model
    model = YOLO(model_path)
    
    # Load random image
    img, image_path = load_random_image(dataset_location)
    
    # Predict
    results = model(img)
    
    # Extract bounding boxes and names
    boxes = results[0].boxes
    names = results[0].names
    
    # Print output
    print(f"Image path: {image_path}")
    print("Bounding boxes and names:")
    for box in boxes:
        class_id = int(box.cls.item())
        print(f"Box: {box.xyxy}, Name: {names[class_id]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a YOLO model on a random image from the dataset")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset location')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    args = parser.parse_args()
    
    main(args.dataset_location, args.model)
