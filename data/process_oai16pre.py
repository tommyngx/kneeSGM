import os
import pandas as pd
import argparse
from tqdm import tqdm
import shutil

def process_files(folder_path, process_oai_raw=False):
    if process_oai_raw:
        # Process OAIraw folder
        oai_raw_folder = os.path.join(folder_path, 'OAIraw')
        trainval_lab_path = os.path.join(oai_raw_folder, 'trainval_lab.csv')
        test_lab_path = os.path.join(oai_raw_folder, 'test_lab.csv')
        
        if not os.path.exists(trainval_lab_path) or not os.path.exists(test_lab_path):
            print("trainval_lab.csv or test_lab.csv not found in the OAIraw folder.")
            return
        
        trainval_lab_df = pd.read_csv(trainval_lab_path)
        test_lab_df = pd.read_csv(test_lab_path)
        
        trainval_lab_df['split'] = 'TRAIN'
        test_lab_df['split'] = 'TEST'
        
        combined_lab_df = pd.concat([trainval_lab_df, test_lab_df])
        
        output_lab_path = os.path.join(oai_raw_folder, 'OAIrawmetadata.csv')
        combined_lab_df.to_csv(output_lab_path, index=False)
        print(f"Combined lab file saved to {output_lab_path}")

    train_path = os.path.join(folder_path, 'train.csv')
    test_path = os.path.join(folder_path, 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("train.csv or test.csv not found in the specified folder.")
        return
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df['split'] = 'TRAIN'
    test_df['split'] = 'TEST'
    
    combined_df = pd.concat([train_df, test_df])
    
    # Remove specified columns
    combined_df = combined_df.drop(columns=['Unnamed: 0', 'raw_file'])
    
    # Fix path and path2 columns
    combined_df['filename'] = combined_df['filename'].str.replace(r'\.jpg$', '', regex=True)
    combined_df['path'] = combined_df['split'].str.lower() + "/" + combined_df['filename'] + ".jpg"
    combined_df['path2'] = "OAI16/" + combined_df['split'] + "/" + combined_df['KL'].astype(int).astype(str) + "/" + combined_df['filename'] + ".png"
    combined_df['KL'] = combined_df['KL'].astype(int)
    combined_df['data'] = 'OAI16'
    
    # Move 'data' column to the first position
    cols = combined_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('data')))
    combined_df = combined_df[cols]

    # Save combined file to OAI16 folder
    oai16_folder = os.path.join(folder_path, 'OAI16')
    if os.path.exists(oai16_folder):
        shutil.rmtree(oai16_folder)
    os.makedirs(oai16_folder, exist_ok=True)
    output_path = os.path.join(folder_path, 'OAI16metadata.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Combined file saved to {output_path}")
    
    # Filter out rows where KL is not equal to 5
    filtered_df = combined_df[combined_df['KL'] != 5]
    filtered_output_path = os.path.join(oai16_folder, 'OAI16metadata.csv')
    filtered_df.to_csv(filtered_output_path, index=False)
    print(f"Filtered combined file saved to {filtered_output_path}")
    
    # Remove folder 5 in TRAIN and TEST within OAI16 folder
    for split in ['train', 'test']:
        folder_to_remove = os.path.join(oai16_folder, split, '5')
        if os.path.exists(folder_to_remove):
            shutil.rmtree(folder_to_remove)
            print(f"Removed folder: {folder_to_remove}")
    
    # Load images from path and save to path2
    for _, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0]):
        src_path = os.path.join(folder_path, row['path'])
        dest_path = os.path.join(folder_path, row['path2'])
        dest_dir = os.path.dirname(dest_path)
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            print(f"Source file {src_path} not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process train.csv and test.csv files.')
    parser.add_argument('--folder_path', type=str, help='Path to the folder containing train.csv and test.csv')
    parser.add_argument('--process_oai_raw', action='store_true', help='Process OAIraw folder if specified')
    
    args = parser.parse_args()
    process_files(args.folder_path, args.process_oai_raw)
