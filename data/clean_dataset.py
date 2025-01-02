import os
import argparse
import pandas as pd
import cv2
import shutil
from tqdm import tqdm
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_metadata(metadata_csv):
    return pd.read_csv(metadata_csv)

def clean_dataset(input_folder, metadata_csv, output_folder, config_path='config/default.yaml'):
    config = load_config(config_path)
    metadata = load_metadata(metadata_csv)
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
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
        "88P2F0362KNEE01.png", "88P2F2657KNEE01.png", '69P2M2730KNEE09.png', "63P2F0971KNEE01.png",
        "58P2F2963KNEE01.png", "58P2F2294KNEE01.png", "40P2M2777KNEE01.png"
    ]
    
    all_images = os.listdir(input_folder)
    print(f"Total images in the input folder: {len(all_images)}")
    print(f"Total IDs in the metadata CSV: {len(metadata)}")
    
    # Create a dictionary to keep track of the latest image for each ID
    latest_images = {}
    id_age_match_count = 0
    id_age_sex_match_count = 0
    
    for image_name in all_images:
        if image_name in image_list:
            continue
        
        parts = image_name.split('KNEE')[0].split('P2')
        age = int(parts[0][:2])
        sex_id = parts[1]
        sex = sex_id[0]
        id_value = sex_id[1:]
        
        if id_value == 'NoID':
            continue
        
        id_value = int(id_value)
        
        if sex_id in latest_images:
            if image_name > latest_images[sex_id]:
                latest_images[sex_id] = image_name
        else:
            latest_images[sex_id] = image_name
        
        if id_value in metadata['ID'].values and age in metadata['Age'].values:
            id_age_match_count += 1
        
        if id_value in metadata['ID'].values and age in metadata['Age'].values and sex in metadata['Sex'].values:
            id_age_sex_match_count += 1
    
    # Remove entries with 'NoID' from latest_images
    latest_images = {k: v for k, v in latest_images.items() if 'NoID' not in k}
    
    print(f"Total IDs with matching ID and age: {id_age_match_count}")
    print(f"Total IDs with matching ID, age, and sex: {id_age_sex_match_count}")

    for image_name in tqdm(latest_images.values(), desc="Copying images"):
        parts = image_name.split('KNEE')[0].split('P2')
        age = int(parts[0][:2])
        sex_id = parts[1]
        sex = sex_id[0]
        id_value = sex_id[1:]
        
        if id_value == 'NoID':
            continue
        
        id_value = int(id_value)
        
        if id_value in metadata['ID'].values and age in metadata['Age'].values and sex in metadata['Sex'].values:
            src_path = os.path.join(input_folder, image_name)
            dst_path = os.path.join(output_folder, image_name)
            shutil.copy(src_path, dst_path)
    
    print(f"Total images copied to the clean folder: {len(latest_images)}")

def main(input_folder, metadata_csv, config_path='config/default.yaml'):
    config = load_config(config_path)
    output_folder = os.path.join(config['output_dir'], 'CLEAN_' + os.path.basename(input_folder))
    
    clean_dataset(input_folder, metadata_csv, output_folder, config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean dataset based on criteria")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing images')
    parser.add_argument('--metadata_csv', type=str, required=True, help='Path to the metadata CSV file')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    main(args.input_folder, args.metadata_csv, args.config)
