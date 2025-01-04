import os
import random
import pydicom
import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Get a random DICOM file, deidentify it, and print its information.")
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing DICOM files.')
    parser.add_argument('--test', type=bool, default=False, help='If true, only print information of a random DICOM file without saving.')
    parser.add_argument('--check_variables', type=bool, default=False, help='If true, check if any of the specified variables are not available in all DICOM files.')
    parser.add_argument('--all', type=bool, default=False, help='If true, deidentify all DICOM files in the folder and save them.')
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

def deidentify_dcm_file(dcm_file, output_folder, print_info=True):
    ds = pydicom.dcmread(dcm_file)
    if print_info:
        print(f"Original Patient's Name: {ds.PatientName}")
        print(f"Original Patient ID: {ds.PatientID}")
        print(f"Original Patient's Birth Date: {ds.PatientBirthDate}")
        print(f"Original Patient's Sex: {ds.PatientSex}")
        print(f"Original Patient's Age: {ds.PatientAge}")
    
    ds.PatientName = "Deidentified"
    ds.PatientID = "Deidentified"
    ds.PatientBirthDate = "Deidentified"
    ds.PatientSex = "Deidentified"
    ds.PatientAge = "Deidentified"
    
    output_path = os.path.join(output_folder, os.path.basename(dcm_file))
    ds.save_as(output_path)
    return ds, output_path

def check_variables(folder):
    missing_variables = {
        "PatientName": [],
        "PatientBirthDate": [],
        "PatientSex": [],
        "PatientAge": []
    }
    dcm_files = [os.path.join(root, file) for root, _, files in os.walk(folder) for file in files if file.endswith('.dcm')]
    for dcm_file in tqdm(dcm_files, desc="Checking variables"):
        ds = pydicom.dcmread(dcm_file)
        if not hasattr(ds, 'PatientName'):
            missing_variables["PatientName"].append(dcm_file)
        if not hasattr(ds, 'PatientBirthDate'):
            missing_variables["PatientBirthDate"].append(dcm_file)
        if not hasattr(ds, 'PatientSex'):
            missing_variables["PatientSex"].append(dcm_file)
        if not hasattr(ds, 'PatientAge'):
            missing_variables["PatientAge"].append(dcm_file)
    for var, files in missing_variables.items():
        if files:
            print(f"Missing {var} in the following files:")
            for file in files:
                print(f"  {file}")

def deidentify_all_dcm_files(folder):
    dcm_files = [os.path.join(root, file) for root, _, files in os.walk(folder) for file in files if file.endswith('.dcm')]
    output_folder = os.path.join(os.path.dirname(folder), os.path.basename(folder) + "_deidentified")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for dcm_file in tqdm(dcm_files, desc="Deidentifying DICOM files"):
        deidentify_dcm_file(dcm_file, output_folder, print_info=False)

def main(folder, test, check_variables_flag, all_flag):
    if check_variables_flag:
        check_variables(folder)
    elif all_flag:
        deidentify_all_dcm_files(folder)
    else:
        dcm_file = get_random_dcm_file(folder)
        if test:
            ds = pydicom.dcmread(dcm_file)
            print_dcm_info(ds)
        else:
            output_folder = os.path.join(os.path.dirname(folder), os.path.basename(folder) + "_deidentified")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            ds, output_path = deidentify_dcm_file(dcm_file, output_folder)
            print(f"Deidentified DICOM file saved to: {output_path}")
            print_dcm_info(ds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a random DICOM file, deidentify it, and print its information.")
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing DICOM files')
    parser.add_argument('--test', type=bool, default=False, help='If true, only print information of a random DICOM file without saving.')
    parser.add_argument('--check_variables', type=bool, default=False, help='If true, check if any of the specified variables are not available in all DICOM files.')
    parser.add_argument('--all', type=bool, default=False, help='If true, deidentify all DICOM files in the folder and save them.')
    args = parser.parse_args()
    
    main(args.folder, args.test, args.check_variables, args.all)
