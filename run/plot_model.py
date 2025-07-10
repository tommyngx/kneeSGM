import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.model_architectures import get_model
from data.data_loader import get_dataloader
from data.preprocess import get_transforms
from utils.gradcam import get_target_layer, plot_gradcam_on_image
from ultralytics import YOLO

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_random_images_by_class(dataset, class_indices, n_per_class=1):
    """
    Randomly select n_per_class images for each class in class_indices.
    Supports datasets with .samples, .imgs, or .data/.labels attributes.
    """
    selected = []
    # Try .samples or .imgs (standard torchvision datasets)
    if hasattr(dataset, "samples"):
        items = dataset.samples
    elif hasattr(dataset, "imgs"):
        items = dataset.imgs
    # Try custom attributes (e.g., .data and .labels)
    elif hasattr(dataset, "data") and hasattr(dataset, "labels"):
        items = list(zip(dataset.data, dataset.labels))
    # Try .image_paths and .labels
    elif hasattr(dataset, "image_paths") and hasattr(dataset, "labels"):
        items = list(zip(dataset.image_paths, dataset.labels))
    else:
        raise AttributeError("Dataset must have .samples, .imgs, (.data and .labels), or (.image_paths and .labels) attributes.")

    for cls in class_indices:
        idxs = [i for i, (_, label) in enumerate(items) if label == cls]
        if idxs:
            chosen = random.sample(idxs, min(n_per_class, len(idxs)))
            selected.extend(chosen)
    return selected

def run_yolo_on_image(image_path, yolo_model):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    results = yolo_model(img, verbose=False)
    if not results:
        return "No detection"
    result = results[0]
    if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
        return "No detection"
    boxes = result.boxes
    names = result.names
    detected = []
    for box in boxes:
        class_id = int(box.cls.item())
        detection_name = names[class_id]
        detected.append(detection_name)
    return ", ".join(set(detected)) if detected else "No detection"

def plot_model_gradcam_and_yolo(config_path, model_name, model_path, yolo_model_path, output_path):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, config_path=config_path, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    _, test_transform = get_transforms(config['data']['image_size'], config_path=config_path)
    test_loader = get_dataloader('test', config['data']['batch_size'], config['data']['num_workers'], 
                                 transform=test_transform, config_path=config_path)
    dataset = test_loader.dataset

    # Randomly select 1 image for each class 1,2,3,4 (not 0)
    class_indices = [1, 2, 3, 4]
    selected_idxs = get_random_images_by_class(dataset, class_indices, n_per_class=1)
    if not selected_idxs:
        print("No images found for the specified classes.")
        return

    # Load YOLO model
    yolo_model = YOLO(yolo_model_path)

    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.suptitle(f"GradCAM and YOLO results for {model_name}", fontsize=18)

    for row, idx in enumerate(selected_idxs):
        # Get image and label
        img_path, label = dataset.samples[idx]
        orig_img = Image.open(img_path).convert("RGB")
        img_tensor = test_transform(orig_img).unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        # GradCAM (use util function)
        target_layer = get_target_layer(model, model_name)
        gradcam_img = plot_gradcam_on_image(model, img_tensor, orig_img, target_layer, pred, device)

        # YOLO prediction
        yolo_pred = run_yolo_on_image(img_path, yolo_model)

        # Plot original
        axes[row, 0].imshow(orig_img)
        axes[row, 0].set_title(f"Original\nLabel: {label}")
        axes[row, 0].axis('off')

        # Plot GradCAM
        axes[row, 1].imshow(gradcam_img)
        axes[row, 1].set_title(f"GradCAM\nPred: {pred} (prob: {probs[pred]:.2f})")
        axes[row, 1].axis('off')

        # Plot YOLO
        axes[row, 2].imshow(orig_img)
        axes[row, 2].set_title(f"YOLO: {yolo_pred}")
        axes[row, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path)
    plt.close()
    print(f"Saved combined GradCAM and YOLO plot to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot GradCAM and YOLO results for random images of each class.")
    parser.add_argument('--config', type=str, default='default.yaml', help='Config file path')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--yolo_model_path', type=str, required=True, help='Path to YOLO model')
    parser.add_argument('--output_path', type=str, default='gradcam_yolo_plot.png', help='Output image path')
    args = parser.parse_args()
    plot_model_gradcam_and_yolo(
        config_path=args.config,
        model_name=args.model_name,
        model_path=args.model_path,
        yolo_model_path=args.yolo_model_path,
        output_path=args.output_path
    )

"""
Ví dụ cách chạy:
python run/plot_model.py \
    --config config/default.yaml \
    --model_name resnet50 \
    --model_path checkpoints/resnet50_epoch_10.pth \
    --yolo_model_path yolov8n.pt \
    --output_path gradcam_yolo_plot.png
"""
