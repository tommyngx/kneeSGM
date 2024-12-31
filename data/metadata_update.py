import os
import argparse
import pandas as pd

def concatenate_metadata(folder_base, output_name, save_link=None):
    all_metadata = []

    for subfolder in os.listdir(folder_base):
        subfolder_path = os.path.join(folder_base, subfolder)
        if os.path.isdir(subfolder_path):
            metadata_path = os.path.join(subfolder_path, 'metadata.csv')
            if os.path.exists(metadata_path):
                metadata = pd.read_csv(metadata_path)
                metadata['split'] = metadata['split'].replace('VAL', 'TEST')
                if subfolder == 'VOSP2':
                    metadata['path_relative'] = 'VOSP2/' + metadata['path_relative']
                all_metadata.append(metadata)
    
    if all_metadata:
        concatenated_metadata = pd.concat(all_metadata, ignore_index=True)
        concatenated_metadata['split'] = concatenated_metadata['split'].replace('VAL', 'TEST')
        output_path = os.path.join(folder_base, output_name)
        concatenated_metadata.to_csv(f"{output_path}.csv", index=False)
        print(f"Concatenated metadata saved to {output_path}")
        
        if save_link:
            concatenated_metadata.to_csv(f"{save_link}/{output_name}.csv", index=False)
            print(f"Link to metadata saved to {save_link}")
    else:
        print("No metadata files found.")

def main(folder_base, output_name, save_link=None):
    concatenate_metadata(folder_base, output_name, save_link)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate metadata from multiple subfolders")
    parser.add_argument('--folder_base', type=str, required=True, help='Path to the base folder containing subfolders with metadata.csv')
    parser.add_argument('--output_name', type=str, required=True, help='Name of the output concatenated metadata file')
    parser.add_argument('--save_link', type=str, help='Path to save the link to the concatenated metadata file')
    args = parser.parse_args()
    
    main(args.folder_base, args.output_name, args.save_link)
