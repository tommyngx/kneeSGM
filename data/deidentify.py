import os
import random
import pydicom
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Get a random DICOM file, deidentify it, and print its information.")
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing DICOM files.')
    return parser.parse_args()

def get_random_dcm_file(folder_path):
    print(f"Searching for DICOM files in folder: {folder_path}")
    dcm_files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.endswith('.dcm')]
    if not dcm_files:
        raise FileNotFoundError("No DICOM files found in the specified folder.")
    print(f"Total DICOM files found: {len(dcm_files)}")
    return random.choice(dcm_files)

def print_dcm_info(ds):
    for elem in ds:
        if elem.tag != (0x7fe0, 0x0010):  # Exclude Pixel Data
            print(f"{elem.tag} {elem.name}: {elem.value}")

def deidentify_dcm_file(dcm_file, output_folder):
    ds = pydicom.dcmread(dcm_file)
    ds.PatientName = "Deidentified"
    output_path = os.path.join(output_folder, os.path.basename(dcm_file))
    ds.save_as(output_path)
    return ds, output_path

def main(folder):
    dcm_file = get_random_dcm_file(folder)
    output_folder = os.path.join(os.path.dirname(folder), os.path.basename(folder) + "_deidentified")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    ds, output_path = deidentify_dcm_file(dcm_file, output_folder)
    print(f"Deidentified DICOM file saved to: {output_path}")
    print_dcm_info(ds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a random DICOM file, deidentify it, and print its information.")
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing DICOM files')
    args = parser.parse_args()
    
    main(args.folder)
