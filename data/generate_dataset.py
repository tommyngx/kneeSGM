import os
import argparse
import pandas as pd
import cv2
import yaml
from tqdm import tqdm
import random

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_metadata(metadata_csv):
    return pd.read_csv(metadata_csv)

def parse_filename(filename):
    parts = filename.split('P2')
    age = parts[0]
    rest = parts[1].split('KNEE')
    sex_id = rest[0]
    knee_side = rest[1].split('_')[1][0]
    return age, sex_id, knee_side

def get_kl_value(row, knee_side):
    if knee_side == 'L':
        return row['KL_Left']
    elif knee_side == 'R':
        return row['KL_Right']
    else:
        return None

def generate_dataset(input_folder, metadata_csv, output_dir, data_name):
    metadata = load_metadata(metadata_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metaraw_data = []
    
    for image_name in tqdm(os.listdir(input_folder), desc="Processing images"):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):
            age, sex_id, knee_side = parse_filename(image_name)
            id_value = sex_id[1:]
            sex = sex_id[0]
            
            if id_value == 'NoID':
                continue
            
            row = metadata[(metadata['ID'] == int(id_value)) & (metadata['Sex'] == sex)]
            if not row.empty:
                kl_value = get_kl_value(row.iloc[0], knee_side)
                if kl_value is not None:
                    img = cv2.imread(os.path.join(input_folder, image_name))
                    new_name = f"{age}P2{sex}{id_value}{knee_side}{kl_value}.png"
                    kl_output_dir = os.path.join(output_dir, str(kl_value))
                    if not os.path.exists(kl_output_dir):
                        os.makedirs(kl_output_dir)
                    cv2.imwrite(os.path.join(kl_output_dir, new_name), img)
                    
                    metaraw_data.append({
                        'data': data_name,
                        'ID': new_name,
                        'KL': kl_value,
                        'path': f"{data_name}/{kl_value}/{new_name}",
                        'split': 'TRAIN' if random.random() < 0.8 else 'TEST'
                    })
    
    metaraw_df = pd.DataFrame(metaraw_data)
    metaraw_df.to_csv(os.path.join(output_dir, 'metaraw.csv'), index=False)

def main(input_folder, metadata_csv, data_name, config_path='config/default.yaml'):
    config = load_config(config_path)
    output_dir = os.path.join(config['output_dir'], 'yolo', 'runs', data_name)
    
    generate_dataset(input_folder, metadata_csv, output_dir, data_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset based on metadata and image files")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing images')
    parser.add_argument('--metadata_csv', type=str, required=True, help='Path to the metadata CSV file')
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    main(args.input_folder, args.metadata_csv, args.data_name, args.config)
