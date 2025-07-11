import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from models.model_architectures import get_model
from data.data_loader import get_dataloader
from data.preprocess import get_transforms
from utils.gradcam import get_target_layer, plot_gradcam_on_image
from ultralytics import YOLO

# Import get_image_info and load_config from test_all.py for consistency
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_all import get_image_info, load_config

def get_random_images_by_class(image_paths, labels, class_indices, n_per_class=1):
    """
    Randomly select n_per_class images for each class in class_indices.
    Returns a list of (img_path, label) tuples, guaranteed to exist on disk.
    """
    items = list(zip(image_paths, labels))
    selected = []
    for cls in class_indices:
        cls_items = [(img_path, label) for img_path, label in items if label == cls and img_path and os.path.exists(img_path)]
        if cls_items:
            chosen = random.sample(cls_items, min(n_per_class, len(cls_items)))
            selected.extend(chosen)
    # Only keep up to 3 images total
    return selected[:3]

def run_yolo_on_image(image_path, yolo_model, return_boxes=False):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    results = yolo_model(img_np, verbose=False)
    if not results:
        return "No detection" if not return_boxes else (img, [])
    result = results[0]
    if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
        return "No detection" if not return_boxes else (img, [])
    boxes = result.boxes
    names = result.names
    detected = []
    box_list = []
    for box in boxes:
        class_id = int(box.cls.item())
        detection_name = names[class_id]
        detected.append(detection_name)
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf.item()) if hasattr(box, "conf") else 1.0
        box_list.append((xyxy, detection_name, conf, class_id))
    if return_boxes:
        return img, box_list
    return ", ".join(set(detected)) if detected else "No detection"

def plot_model_gradcam_and_yolo(config_path, model_name, model_path, yolo_model_path, output_path):
    print("[DEBUG] Loading config...")
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[DEBUG] Loading model...")
    model = get_model(model_name, config_path=config_path, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("[DEBUG] Preparing transforms and dataloader...")
    _, test_transform = get_transforms(config['data']['image_size'], config_path=config_path)
    train_loader = get_dataloader('train', config['data']['batch_size'], config['data']['num_workers'],
                                  transform=test_transform, config_path=config_path)
    dataset = train_loader.dataset

    print("[DEBUG] Getting image info...")
    image_names, image_paths = get_image_info(type('FakeLoader', (), {'dataset': dataset})())
    if hasattr(dataset, 'labels'):
        labels = list(dataset.labels)
    elif hasattr(dataset, 'data') and 'label' in dataset.data:
        labels = list(dataset.data['label'])
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]

    # --- Lọc chỉ lấy lớp 2, 3, 4 ---
    filtered = [(p, l) for p, l in zip(image_paths, labels) if l in [2, 3, 4]]
    if not filtered:
        print("Không có ảnh thuộc lớp 2, 3, 4.")
        return
    image_paths, labels = zip(*filtered)
    image_paths = list(image_paths)
    labels = list(labels)
    # --- hết đoạn lọc ---

    print("[DEBUG] Selecting random images by class...")
    class_indices = [1, 2, 3, 4]
    # Select 15 images (5 for each class, or as many as available)
    selected_items = []
    for _ in range(5):
        selected_items.extend(get_random_images_by_class(image_paths, labels, class_indices, n_per_class=1))
    selected_items = selected_items[:15]
    if not selected_items:
        print("No images found for the specified classes.")
        return

    print("[DEBUG] Loading YOLO model...")
    yolo_model = YOLO(yolo_model_path)

    # Save 5 images, each with 3x3 grid (15 images total)
    num_imgs_per_fig = 3
    num_figs = 5
    for fig_idx in range(num_figs):
        print(f"[DEBUG] Creating plot {fig_idx+1}/{num_figs}...")
        fig, axes = plt.subplots(num_imgs_per_fig, 3, figsize=(15, 5 * num_imgs_per_fig))
        fig.suptitle(f"GradCAM and YOLO results for {model_name} (Batch {fig_idx+1})", fontsize=22)

        # Font for title
        import matplotlib
        font_size = 22
        font_path = "Poppins.ttf"
        try:
            matplotlib.font_manager.fontManager.addfont(font_path)
            prop = matplotlib.font_manager.FontProperties(fname=font_path, size=font_size)
        except Exception:
            prop = None

        for row in range(num_imgs_per_fig):
            idx = fig_idx * num_imgs_per_fig + row
            if idx >= len(selected_items):
                for col in range(3):
                    axes[row, col].axis('off')
                continue
            img_path, label = selected_items[idx]
            print(f"[DEBUG] Processing image {idx+1}/{len(selected_items)}: {img_path}")
            orig_img = Image.open(img_path).convert("RGB")
            # Handle albumentations transform (expects dict input)
            if hasattr(test_transform, "__call__") and (
                hasattr(test_transform, "is_albumentations") and test_transform.is_albumentations
                or "albumentations" in str(type(test_transform)).lower()
            ):
                transformed = test_transform(image=np.array(orig_img))
                img = transformed["image"]
                if isinstance(img, torch.Tensor):
                    if img.ndim == 3:
                        img_tensor = img.unsqueeze(0).float().to(device)
                    else:
                        img_tensor = img.float().to(device)
                elif isinstance(img, np.ndarray):
                    if img.ndim == 3 and img.shape[2] in [1, 3]:
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    else:
                        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
                else:
                    raise TypeError(f"Unknown image type after albumentations: {type(img)}")
            else:
                img_tensor = test_transform(orig_img).unsqueeze(0).to(device)

            # Model prediction
            print("[DEBUG] Running model prediction...")
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).item()
                pred = label
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]

            # GradCAM (use util function)
            print("[DEBUG] Generating GradCAM image...")
            target_layer = get_target_layer(model, model_name)
            gradcam_img = plot_gradcam_on_image(
                model, img_tensor, orig_img, target_layer, pred, device, model_name=model_name
            )

            # YOLO prediction with bounding boxes and labels
            print("[DEBUG] Running YOLO detection...")
            yolo_img, yolo_boxes = run_yolo_on_image(img_path, yolo_model, return_boxes=True)
            yolo_img_draw = np.array(orig_img).copy()
            import cv2
            symptom_names = []
            color_palette = [
                (0, 255, 255),  # Yellow
                (0, 128, 255),  # Light Blue
                (255, 128, 0),  # Orange
                (128, 0, 128),  # Purple
                (128, 128, 0),  # Olive
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow

            ]
            for xyxy, name, conf, class_id in yolo_boxes:
                x1, y1, x2, y2 = xyxy
                color = color_palette[class_id % len(color_palette)]
                cv2.rectangle(yolo_img_draw, (x1, y1), (x2, y2), color, 2)
                font_scale = 1.2
                font_thickness = 3
                label_text = f"{name}"
                symptom_names.append(f"{name}")
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(yolo_img_draw, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
                cv2.putText(yolo_img_draw, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness)

            # Plot original
            axes[row, 0].imshow(orig_img)
            axes[row, 0].set_title(f"Original\nLabel: {label}", fontproperties=prop, fontsize=font_size)
            axes[row, 0].axis('off')

            # Plot GradCAM
            title_color = 'green' if label == pred else 'red'
            axes[row, 1].imshow(gradcam_img)
            axes[row, 1].set_title(
                f"GradCAM\nPred: {pred} (prob: {probs[pred]:.2f})",
                fontproperties=prop, fontsize=font_size, color=title_color
            )
            axes[row, 1].axis('off')

            # Plot YOLO with bounding boxes and symptom summary
            symptom_summary = ", ".join(sorted(set(symptom_names))) if symptom_names else "Không phát hiện"
            axes[row, 2].imshow(yolo_img_draw)
            axes[row, 2].set_title(
                f"Features:\n {symptom_summary}",
                fontproperties=prop, fontsize=font_size
            )
            axes[row, 2].axis('off')

        print(f"[DEBUG] Saving plot {fig_idx+1}/{num_figs}...")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        if not output_path or output_path.strip() == "":
            save_path = os.path.join(os.getcwd(), f"gradcam_yolo_plot_{fig_idx+1}.png")
        elif os.path.isdir(output_path):
            save_path = os.path.join(output_path, f"gradcam_yolo_plot_{fig_idx+1}.png")
        else:
            save_path = output_path.replace(".png", f"_{model_name}_{fig_idx+1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved combined GradCAM and YOLO plot to {save_path}")

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