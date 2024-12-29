import os
import pandas as pd
import argparse

def process_files(folder_path):
    train_path = os.path.join(folder_path, 'train.csv')
    test_path = os.path.join(folder_path, 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("train.csv or test.csv not found in the specified folder.")
        return
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    
    combined_df = pd.concat([train_df, test_df])
    
    output_path = os.path.join(folder_path, 'combined.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Combined file saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process train.csv and test.csv files.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing train.csv and test.csv')
    
    args = parser.parse_args()
    process_files(args.folder_path)
