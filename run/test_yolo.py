import os
import sys
import argparse
import pandas as pd
import yaml
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from collections import Counter
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_yolo_on_image(image_path, model):
    """
    Run YOLO detection on a single image using the YOLO Python API.
    Returns a tuple:
      (detection string, list of bounding box areas for detections labeled as osteophyte/osteophytemore)
    """
    img = cv2.imread(image_path)
    if img is None:
        return "Error: cannot read image", []
    
    # Resize image to 448x448 before YOLO processing
    img = cv2.resize(img, (448, 448))
    
    results = model(img, verbose=False)
    if not results:
        return "No detection", []
    
    result = results[0]
    if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
        return "No detection", []
    
    boxes = result.boxes
    names = result.names
    detected = []
    osteophyte_areas = []
    for box in boxes:
        class_id = int(box.cls.item())
        detection_name = names[class_id]
        detected.append(detection_name)
        if detection_name.lower() in ["osteophyte", "osteophytemore"]:
            # Extract bounding box coordinates: assume box.xyxy[0] gives [x1, y1, x2, y2]
            coords = box.xyxy[0].detach().cpu().numpy() if hasattr(box.xyxy[0], "detach") else box.xyxy[0]
            x1, y1, x2, y2 = coords
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            area = width * height
            osteophyte_areas.append(area)
    
    # Count frequency for adding extra label if needed
    detected_lower = [name.lower() for name in detected]
    counter = Counter(detected_lower)
    unique = {}
    for name in detected:
        key = name.lower()
        if key not in unique:
            unique[key] = name
    result_list = list(unique.values())
    if counter.get("osteophyte", 0) > 1:
        result_list.append("OsteophyteMore")
    # New rule: if any osteophyte area > 750, add "OsteophyteBig"
    if any(area > 500 for area in osteophyte_areas):
        if "OsteophyteBig" not in result_list:
            result_list.append("OsteophyteBig")
    
    detection_str = ", ".join(result_list)
    return detection_str, osteophyte_areas

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
    
    if yolo_model is None:
        yolo_model = "yolov8n.pt"
    
    model = YOLO(yolo_model)
    model.conf = conf
    
    # Global list to accumulate osteophyte areas from all images
    all_osteophyte_areas = []
    yolo_predictions = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_path = row['image_path']
        pred_str, areas = run_yolo_on_image(image_path, model)
        yolo_predictions.append(pred_str)
        all_osteophyte_areas.extend(areas)
    
    df['YOLO_prediction'] = yolo_predictions
    output_csv = os.path.join(project_dir, "all_models_predictions_yolo.csv")
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV with YOLO predictions saved at: {output_csv}")
    
    if all_osteophyte_areas:
        min_area = np.min(all_osteophyte_areas)
        max_area = np.max(all_osteophyte_areas)
        median_area = np.median(all_osteophyte_areas)
        avg_area = np.mean(all_osteophyte_areas)
        print("\nOsteophyte Detection Areas (in pixel^2):")
        print(f"Min: {min_area:.2f}, Max: {max_area:.2f}, Median: {median_area:.2f}, Average: {avg_area:.2f}")
    else:
        print("No osteophyte detections found to compute size statistics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a CSV of model predictions, run YOLO detection on each image, extract object names and osteophyte size statistics, and append the results.")
    parser.add_argument('--config', type=str, default='default.yaml', help='Name of the configuration file in config folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the all_models_predictions.csv file')
    parser.add_argument('--yolo_model', type=str, default=None, help='YOLO model file name (e.g., yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for YOLO detection')
    parser.add_argument('--save', type=bool, default=True, help='Whether to save YOLO output images (not used in this version)')
    args = parser.parse_args()
    
    main(config=args.config, csv_path=args.csv_path, yolo_model=args.yolo_model, conf=args.conf, save=args.save)
