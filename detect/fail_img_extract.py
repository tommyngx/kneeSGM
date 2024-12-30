import os
import cv2
import yaml
import argparse
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def resize_and_save_images(image_list, dataset_location, output_folder, size=(640, 640)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_name in tqdm(image_list, desc="Processing images"):
        image_path = os.path.join(dataset_location, image_name)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            resized_img = cv2.resize(img, size)
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, resized_img)
        else:
            print(f"Image {image_path} not found.")

def main(dataset_location, config_path='config/default.yaml'):
    config = load_config(config_path)
    output_folder = os.path.join(config['output_dir'], 'fail_img')
    
    image_list = [
        "38P2F4088KNEE01.png", "42P2F2889KNEE01.png", "42P2FNoIDKNEE01.png", "44P2F3901KNEE01.png",
        "46P2F0864KNEE01.png", "47P2F0352KNEE02.png", "49P2M1328KNEE01.png", "50P2F0084KNEE01.png",
        "50P2F0406KNEE01.png", "50P2M4434KNEE01.png", "53P2F3320KNEE01.png", "53P2M1001KNEE01.png",
        "54P2F0457KNEE01.png", "55P2F1482KNEE01.png", "55P2M4250KNEE01.png", "56P2F3820KNEE01.png",
        "57P2F3756KNEE01.png", "58P2F1046KNEE01.png", "58P2M3866KNEE01.png", "59P2F3721KNEE01.png",
        "59P2M4280KNEE01.png", "60P2F0141KNEE01.png", "60P2F0542KNEE01.png", "60P2F4350KNEE01.png",
        "60P2M2646KNEE01.png", "61P2F2516KNEE01.png", "62P2F2647KNEE01.png", "62P2F4082KNEE01.png",
        "63P2F0159KNEE01.png", "63P2F0259KNEE01.png", "63P2F2341KNEE01.png", "63P2F3963KNEE01.png",
        "65P2F3766KNEE01.png", "66P2M4404KNEE01.png", "67P2F1888KNEE01.png", "67P2M2855KNEE01.png",
        "70P2F1209KNEE01.png", "70P2F1578KNEE01.png", "70P2F3791KNEE01.png", "73P2F0289KNEE01.png",
        "77P2F2379KNEE01.png", "79P2F1778KNEE01.png", "86P2F4044KNEE01.png", "87P2F0841KNEE01.png",
        "88P2F0362KNEE01.png", "88P2F2657KNEE01.png"
    ]
    
    resize_and_save_images(image_list, dataset_location , output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and resize failed images")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the input folder containing images')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    main(args.dataset_location, args.config)
