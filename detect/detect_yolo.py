import os
import subprocess
import argparse
import random
import cv2

def detect_yolo(dataset_location, model, conf):
    test_images_dir = os.path.join(dataset_location, '')
    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Test images directory '{test_images_dir}' not found.")
    
    images = [img for img in os.listdir(test_images_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        raise FileNotFoundError("No images found in the test images directory.")
    
    random_image = random.choice(images)
    image_path = os.path.join(test_images_dir, random_image)
    
    command = [
        "yolo", "task=detect", "mode=predict", f"model={model}",
        f"conf={conf}", f"source={image_path}", "save=True"
    ]
    subprocess.run(command, check=True)
    
    output_image_path = os.path.join(os.path.dirname(image_path), 'predictions.jpg')
    if os.path.exists(output_image_path):
        img = cv2.imread(output_image_path)
        cv2.imshow('Detected Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        raise FileNotFoundError(f"Output image '{output_image_path}' not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects using YOLO model")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset location')
    parser.add_argument('--model', type=str, required=True, help='Model file to be used for detection')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection')
    args = parser.parse_args()
    
    detect_yolo(args.dataset_location, args.model, args.conf)
