import os
import argparse
import pandas as pd
from tqdm import tqdm

def generate_metadata(folder_path):
    data = []
    
    for split in ['train', 'test', 'val']:
        split_folder = os.path.join(folder_path, split)
        if not os.path.exists(split_folder):
            continue
        
        for kl in range(5):
            kl_folder = os.path.join(split_folder, str(kl))
            if not os.path.exists(kl_folder):
                continue
            
            for filename in tqdm(os.listdir(kl_folder), desc=f"Processing {split}/{kl}"):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(split_folder, str(kl), filename)
                    path_relative = f"OAI299/{split}/{kl}/{filename}"
                    data.append({
                        'data': 'OAI299',
                        'filename': filename,
                        'KL': kl,
                        'path': path,
                        'path_relative': path_relative,
                        'split': split.upper()
                    })
    
    df = pd.DataFrame(data)
    output_csv = os.path.join(folder_path, 'OAI299', 'metadata.csv')
    df.to_csv(output_csv, index=False)
    print(f"Metadata saved to {output_csv}")

def main(folder_path):
    generate_metadata(folder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata.csv for OAI299 dataset")
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the OAI299 dataset folder')
    args = parser.parse_args()
    
    main(args.folder_path)
