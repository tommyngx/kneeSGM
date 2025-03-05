import os
import sys
import argparse
import pandas as pd
import yaml
import cv2
from ultralytics import YOLO
from tqdm import tqdm  # added tqdm import

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_yolo_on_image(image_path, model):
    """
    Run YOLO detection on a single image using the YOLO Python API
    and return a comma-separated string of detected object names.
    """
    img = cv2.imread(image_path)
    if img is None:
        return "Error: cannot read image"
    
    # Perform detection (verbose=False to suppress output)
    results = model(img, verbose=False)
    if not results:
        return "No detection"
    
    result = results[0]
    # Ensure that boxes exist
    if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
        return "No detection"
    
    boxes = result.boxes
    # Use the names provided in the detection result (a dict mapping class_id to name)
    names = result.names
    detected = []
    for box in boxes:
        class_id = int(box.cls.item())
        detected.append(names[class_id])
    
    if not detected:
        return "No detection"
    
    # Remove duplicates and join the names into a single string.
    detected = list(set(detected))
    return ", ".join(detected)

def main(config='default.yaml', csv_path=None, yolo_model=None, conf=0.25, save=True):
    # Load configuration
    config_path = os.path.join('config', config)
    config = load_config(config_path)
    
    # Determine output directory based on CSV location
    if csv_path is None:
        print("Please provide path to CSV file (all_models_predictions.csv)")
        sys.exit(1)
    csv_path = os.path.abspath(csv_path)
    project_dir = os.path.dirname(csv_path)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    if 'image_path' not in df.columns:
        print("CSV file must include an 'image_path' column.")
        sys.exit(1)
    
    # Use default YOLO model if not provided
    if yolo_model is None:
        yolo_model = "yolov8n.pt"
    
    # Load the YOLO model using the ultralytics API
    model = YOLO(yolo_model)
    # Set confidence (if supported by API)
    model.conf = conf
    
    # Iterate through each row in the CSV using tqdm for progress
    yolo_predictions = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_path = row['image_path']
        pred_summary = run_yolo_on_image(image_path, model)
        yolo_predictions.append(pred_summary)
    
    # Add the new column and save updated CSV
    df['YOLO_prediction'] = yolo_predictions
    output_csv = os.path.join(project_dir, "all_models_predictions_yolo.csv")
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV with YOLO predictions saved at: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a CSV of model predictions, run YOLO detection on each image, extract object names, and append the results.")
    parser.add_argument('--config', type=str, default='default.yaml', help='Name of the configuration file in config folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the all_models_predictions.csv file')
    parser.add_argument('--yolo_model', type=str, default=None, help='YOLO model file name (e.g., yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for YOLO detection')
    parser.add_argument('--save', type=bool, default=True, help='Whether to save YOLO output images (not used in this version)')
    args = parser.parse_args()
    
    main(config=args.config, csv_path=args.csv_path, yolo_model=args.yolo_model, conf=args.conf, save=args.save)
