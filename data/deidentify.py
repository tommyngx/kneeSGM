import os
import random
import pydicom
import argparse

def get_random_dcm_file(folder_path):
    print(f"Searching for DICOM files in folder: {folder_path}")
    dcm_files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.endswith('.dcm')]
    if not dcm_files:
        raise FileNotFoundError("No DICOM files found in the specified folder.")
    print(f"Total DICOM files found: {len(dcm_files)}")
    return random.choice(dcm_files)

def print_dcm_info(dcm_file):
    ds = pydicom.dcmread(dcm_file)
    print(f"Information for DICOM file: {dcm_file}")
    for elem in ds:
        if elem.tag != (0x7fe0, 0x0010):  # Exclude Pixel Data
            print(f"{elem.tag} {elem.name}: {elem.value}")

def main(folder):
    dcm_file = get_random_dcm_file(folder)
    print_dcm_info(dcm_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a random DICOM file and print its information.")
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing DICOM files')
    args = parser.parse_args()
    
    main(args.folder)
