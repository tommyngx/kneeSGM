import os
import yaml
import argparse

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_yolo(dataset_location):
    config = load_config('config/default.yaml')  # Update this path to your config file
    output_folder = config['output_dir']
    
    command = f"!yolo task=detect mode=train model=yolo11s.pt data={dataset_location}/data.yaml epochs=10 imgsz=640 plots=True project={output_folder}"
    os.system(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset location')
    args = parser.parse_args()
    
    train_yolo(args.dataset_location)
