import os
import subprocess
import yaml
import argparse

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_yolo(dataset_location, model, config='default.yaml'):
    config_path = os.path.join('config', config)
    config = load_config(config_path)
    output_folder = config['output_dir'] + "/yolo"
    print("output_folder:", output_folder)
    
    command = [
        "yolo", "task=detect", "mode=train", f"model={model}",
        f"data={dataset_location}/data.yaml", "epochs=4000", "imgsz=640", "plots=True",
        f"project={output_folder}", "patience=1000", "fliplr= 0.0"
    ]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset location')
    parser.add_argument('--model', type=str, required=True, help='Model file to be used for training')
    parser.add_argument('--config', type=str, default='default.yaml', help='Name of the configuration file')
    args = parser.parse_args()
    
    train_yolo(args.dataset_location, args.model, args.config)
