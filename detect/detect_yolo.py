import os
import subprocess
import argparse
import random
import cv2

def detect_yolo(dataset_location, model, conf, source_type):
    test_images_dir = os.path.join(dataset_location, '')
    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Test images directory '{test_images_dir}' not found.")
    
    if source_type == 'random':
        images = [img for img in os.listdir(test_images_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            raise FileNotFoundError("No images found in the test images directory.")
        
        random_image = random.choice(images)
        source = os.path.join(test_images_dir, random_image)
    else:
        source = test_images_dir
    
    command = [
        "yolo", "task=detect", "mode=predict", f"model={model}",
        f"conf={conf}", f"source={source}", "save=True"
    ]
    subprocess.run(command, check=True)
    
    output_dir = os.path.join(os.getcwd(), 'runs/detect/predict')
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' not found.")
    
    output_images = [img for img in os.listdir(output_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
    if not output_images:
        raise FileNotFoundError("No images found in the output directory.")
    
    random_output_image = random.choice(output_images)
    output_image_path = os.path.join(output_dir, random_output_image)
    
    img = cv2.imread(output_image_path)
    cv2.imshow('Detected Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects using YOLO model")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset location')
    parser.add_argument('--model', type=str, required=True, help='Model file to be used for detection')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection')
    parser.add_argument('--source_type', type=str, choices=['random', 'folder'], default='random', help='Source type: random image or whole folder')
    args = parser.parse_args()
    
    detect_yolo(args.dataset_location, args.model, args.conf, args.source_type)
