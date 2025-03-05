import os
import sys
import argparse
import subprocess
import pandas as pd
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_yolo_on_image(image_path, yolo_model, conf, save, project):
    """
    Run YOLO detection on a single image and capture its prediction summary.
    The YOLO command is run as a subprocess.
    """
    name_folder = "predict"
    command = [
        "yolo", "task=detect", "mode=predict",
        f"model={yolo_model}",
        f"conf={conf}",
        f"source={image_path}",
        f"name= {name_folder}",        # Force outputs into a single folder named 'predict'
        f"save={str(save).lower()}",
        f"project={project}"
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, timeout=60)
        output_text = result.stdout.strip()
    except Exception as e:
        output_text = f"Error: {e}"
    # Use the first nonempty line as a summary
    summary = next((line for line in output_text.splitlines() if line.strip()), "No detection")
    return summary

def main(config='default.yaml', csv_path=None, yolo_model=None, conf=0.25, save=True):
    # Load configuration
    config_path = os.path.join('config', config)
    config = load_config(config_path)
    
    # Determine output directory (used as YOLO project folder)
    # Here we use the folder where the CSV is located
    if csv_path is None:
        print("Please provide path to CSV file (all_models_predictions.csv)")
        sys.exit(1)
    csv_path = os.path.abspath(csv_path)
    project_dir = os.path.dirname(csv_path)
    
    # Load CSV containing previous predictions
    df = pd.read_csv(csv_path)
    if 'image_path' not in df.columns:
        print("CSV file must include an 'image_path' column.")
        sys.exit(1)
    
    # Use default YOLO model if not provided (adjust as needed)
    if yolo_model is None:
        yolo_model = "yolov8n.pt"
    
    # Prepare a list to hold YOLO detection summaries
    yolo_predictions = []
    total = len(df)
    for idx, row in df.iterrows():
        image_path = row['image_path']
        print(f"Processing image ({idx+1}/{total}): {image_path}")
        pred_summary = run_yolo_on_image(image_path, yolo_model, conf, save, project_dir)
        yolo_predictions.append(pred_summary)
    
    # Add a new column for YOLO predictions
    df['YOLO_prediction'] = yolo_predictions
    # Save updated CSV
    output_csv = os.path.join(project_dir, "all_models_predictions_yolo.csv")
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV with YOLO predictions saved at: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a CSV of model predictions, run YOLO detection on each image, and append YOLO prediction output.")
    parser.add_argument('--config', type=str, default='default.yaml', help='Name of the configuration file in config folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the all_models_predictions.csv file')
    parser.add_argument('--yolo_model', type=str, default=None, help='YOLO model file name (e.g., yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for YOLO detection')
    parser.add_argument('--save', type=bool, default=True, help='Whether to save YOLO output images')
    args = parser.parse_args()
    
    main(config=args.config, csv_path=args.csv_path, yolo_model=args.yolo_model, conf=args.conf, save=args.save)
