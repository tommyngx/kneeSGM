import torch, os
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from .utils import blue_to_gray_np, red_to_gray_np, red_to_0_np
from matplotlib import font_manager
import requests

def register_hooks(model, image, target_layer):
    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image.requires_grad = True

    features = []
    def hook_fn(module, input, output):
        features.append(output)

    handle = target_layer.register_forward_hook(hook_fn)
    output = model(image)
    handle.remove()

    score = output[:, output.max(1)[-1]]
    score.backward()

    gradients = image.grad.data
    activations = features[0].detach()
    return activations, gradients

def generate_gradcam(model, image, target_layer, model_name):
    activations, gradients = register_hooks(model, image, target_layer)
    
    if any(cnn_model in model_name for cnn_model in ['resnet', 'resnext', 'efficientnet', 'densenet', 'convnext', 'resnext50_32x4d', 'xception']):
        return generate_gradcam_cnn(activations, gradients, image)
    elif 'fastvit' in model_name:
        return generate_gradcam_fastvit(activations, gradients, image)
    elif 'vit' in model_name and 'fastvit' not in model_name:
        return generate_gradcam_vit(activations, gradients, image)
    elif 'caformer' in model_name:
        return generate_gradcam_caformer(activations, gradients, image)
    else:
        raise ValueError(f"Model {model_name} not supported for Grad-CAM.")

def generate_gradcam_cnn(activations, gradients, image):
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(min(activations.shape[1], pooled_gradients.shape[0])):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    return post_process_heatmap(heatmap, image)

def generate_gradcam_vit(activations, gradients, image):
    if gradients.dim() == 4:
        gradients = torch.mean(gradients, dim=[2, 3])
    if gradients.dim() == 2:
        if gradients.size(1) != activations.size(2):
            gradients = gradients.unsqueeze(1)
            gradients = torch.mean(gradients, dim=-1, keepdim=True)
            gradients = gradients.expand(-1, activations.size(1), activations.size(2))
        else:
            gradients = gradients.unsqueeze(1).expand(-1, activations.size(1), -1)
    elif gradients.dim() == 3 and gradients.size(1) == 1:
        gradients = gradients.expand(-1, activations.size(1), activations.size(2))
    elif gradients.dim() == 3 and gradients.size(2) == 3:
        gradients = torch.mean(gradients, dim=2, keepdim=True)
        gradients = gradients.expand(activations.size(0), activations.size(1), activations.size(2))
    else:
        raise ValueError(f"Unexpected gradients dimensions: {gradients.shape}")

    pooled_gradients = torch.mean(gradients, dim=1, keepdim=True)
    pooled_gradients = pooled_gradients.expand_as(activations)
    weighted_activations = activations * pooled_gradients
    heatmap = torch.sum(weighted_activations, dim=-1).squeeze()
    grid_size = int(np.sqrt(heatmap.size(0) - 1))
    heatmap = heatmap[1:].view(grid_size, grid_size)
    return post_process_heatmap(heatmap, image)

def generate_gradcam_caformer(activations, gradients, image):
    if gradients.dim() == 4:
        gradients = torch.mean(gradients, dim=[2, 3])
    if gradients.dim() == 2:
        if gradients.size(1) != activations.size(2):
            gradients = gradients.unsqueeze(2)
            gradients = gradients.expand(-1, -1, activations.size(2))
            gradients = gradients.mean(dim=1, keepdim=False)
        else:
            gradients = gradients.unsqueeze(1).expand(-1, activations.size(1), -1)
    elif gradients.dim() == 3:
        if gradients.size(1) != activations.size(1) or gradients.size(2) != activations.size(2):
            gradients = torch.mean(gradients, dim=1, keepdim=True)
            gradients = gradients.expand(-1, activations.size(1), activations.size(2))
    elif gradients.dim() == 3 and gradients.size(2) == 3:
        gradients = torch.mean(gradients, dim=2, keepdim=True)
        gradients = gradients.expand(-1, activations.size(1), activations.size(2))
    else:
        raise ValueError(f"Unexpected gradients dimensions: {gradients.shape}")

    pooled_gradients = torch.mean(gradients, dim=1, keepdim=True)
    pooled_gradients = pooled_gradients.expand_as(activations)
    weighted_activations = activations * pooled_gradients
    heatmap = torch.sum(weighted_activations, dim=-1).squeeze()
    grid_size = int(np.sqrt(heatmap.size(0)))
    heatmap = heatmap.view(grid_size, grid_size)
    return post_process_heatmap(heatmap, image)

def generate_gradcam_fastvit(activations, gradients, image):
    if activations.dim() == 4:
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(min(activations.shape[1], pooled_gradients.shape[0])):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()
    elif activations.dim() == 3:
        if gradients.dim() == 4:
            gradients = torch.mean(gradients, dim=[2, 3])
        if gradients.dim() == 2:
            gradients = gradients.unsqueeze(1).expand(activations.size(0), activations.size(1), gradients.size(1))
        elif gradients.dim() == 3 and gradients.size(1) == 1:
            gradients = gradients.expand(activations.size(0), activations.size(1), gradients.size(2))
        elif gradients.dim() == 3 and gradients.size(2) == 224:
            gradients = torch.mean(gradients, dim=1, keepdim=True)
            gradients = gradients.expand(activations.size(0), activations.size(1), activations.size(2))
        elif gradients.dim() == 3 and gradients.size(2) == 3:
            gradients = torch.mean(gradients, dim=2, keepdim=True)
            gradients = gradients.expand(activations.size(0), activations.size(1), activations.size(2))
        else:
            raise ValueError(f"Unexpected gradients dimensions: {gradients.dim()}")

        pooled_gradients = torch.mean(gradients, dim=1, keepdim=True)
        pooled_gradients = pooled_gradients.expand(activations.size(0), activations.size(1), activations.size(2))
        weighted_activations = activations * pooled_gradients
        heatmap = torch.sum(weighted_activations, dim=-1).squeeze()
        grid_size = int(np.sqrt(heatmap.size(0)))
        heatmap = heatmap.view(grid_size, grid_size)
    else:
        raise ValueError(f"Unexpected activations dimensions: {activations.dim()}")
    return post_process_heatmap(heatmap, image)

def post_process_heatmap(heatmap, image):
    # Ensure heatmap is float32 and normalized
    heatmap = F.relu(heatmap)
    if torch.max(heatmap) > 0:
        heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()
    # image shape: (B, C, H, W) or (C, H, W)
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            h, w = image.shape[2], image.shape[3]
        elif image.dim() == 3:
            h, w = image.shape[1], image.shape[2]
        else:
            raise ValueError("Unexpected image tensor shape for resizing GradCAM heatmap.")
    elif isinstance(image, np.ndarray):
        h, w = image.shape[-2], image.shape[-1]
    else:
        raise ValueError("Unknown image type for resizing GradCAM heatmap.")

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (w, h))
    # Normalize again after resize (sometimes resize can change range)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    # Convert to uint8 for color mapping
    heatmap_uint8 = np.uint8(255 * heatmap)
    # Apply color map for color GradCAM
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return heatmap_color

def generate_gradcam_plus_plus(model, image, target_layer, model_name):
    activations, gradients = register_hooks(model, image, target_layer)

    if any(cnn_model in model_name for cnn_model in ['resnet', 'resnext', 'efficientnet', 'densenet', 'convnext', 'resnext50_32x4d', 'xception']):
        return generate_gradcam_plus_plus_cnn(activations, gradients, image)
    elif 'fastvit' in model_name:
        return generate_gradcam_plus_plus_fastvit(activations, gradients, image)
    elif 'vit' in model_name and 'fastvit' not in model_name:
        return generate_gradcam_plus_plus_vit(activations, gradients, image)
    elif 'caformer' in model_name:
        return generate_gradcam_plus_plus_caformer(activations, gradients, image)
    else:
        raise ValueError(f"Model {model_name} not supported for Grad-CAM++.")

def generate_gradcam_plus_plus_cnn(activations, gradients, image):
    b, k, u, v = gradients.size()
    alpha = gradients.pow(2)
    alpha /= (2 * gradients.pow(2) + (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True) + 1e-8)
    weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

    heatmap = (weights * activations).sum(dim=1).squeeze(0)
    heatmap = F.relu(heatmap)
    heatmap = heatmap / heatmap.max()

    heatmap = heatmap.cpu().numpy()
    heatmap = cv2.resize(heatmap, (image.shape[3], image.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = 255 - heatmap
    return heatmap

def generate_gradcam_plus_plus_vit(activations, gradients, image):
    return generate_gradcam_vit(activations, gradients, image)

def generate_gradcam_plus_plus_caformer(activations, gradients, image):
    return generate_gradcam_caformer(activations, gradients, image)

def generate_gradcam_plus_plus_fastvit(activations, gradients, image):
    return generate_gradcam_fastvit(activations, gradients, image)

def show_cam_on_image(img, mask, use_rgb=False):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = red_to_gray_np(heatmap)
    heatmap = np.float32(heatmap) / 255
    
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam

def save_random_predictions(model, dataloader, device, output_dir, epoch, class_names, use_gradcam_plus_plus=False, target_layer=None, acc=None, model_name=None):
    plt.style.use('default')
    if model_name is None:
        raise ValueError("model_name must be provided")
    
    # Download the font file if it does not exist
    font_url = 'https://github.com/tommyngx/style/blob/main/Poppins.ttf?raw=true'
    font_path = 'Poppins.ttf'
    if not os.path.exists(font_path):
        response = requests.get(font_url)
        with open(font_path, 'wb') as f:
            f.write(response.content)

    # Load the font
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    outputs = torch.softmax(outputs, dim=1)  # Apply softmax
    preds = torch.argmax(outputs, dim=1)
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    for i in range(4):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        label = labels[i].item()
        pred = preds[i].item()
        if use_gradcam_plus_plus:
            heatmap = generate_gradcam_plus_plus(model, images[i].unsqueeze(0), target_layer, model_name)
        else:
            heatmap = generate_gradcam(model, images[i].unsqueeze(0), target_layer, model_name)
        cam_image = show_cam_on_image(img, heatmap, use_rgb=True)
        
        if label == pred:
            title_color = 'green'
        else:
            title_color = 'red'
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i+1}\nLabel: {class_names[label]}\nPred: {class_names[pred]}", fontproperties=prop, fontsize=18, color=title_color)
        axes[i, 0].axis('off')
        
        if label == pred:
            title_color = 'green'
        else:
            title_color = 'red'
        axes[i, 1].imshow(cam_image)
        axes[i, 1].set_title(f"Grad-CAM {i+1}\nLabel: {class_names[label]}\nPred: {class_names[pred]}", fontproperties=prop, fontsize=18, color=title_color)
        axes[i, 1].axis('off')
        
        if i + 4 < len(images):
            img = images[i + 4].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            label = labels[i + 4].item()
            pred = preds[i + 4].item()
            if use_gradcam_plus_plus:
                heatmap = generate_gradcam_plus_plus(model, images[i + 4].unsqueeze(0), target_layer, model_name)
            else:
                heatmap = generate_gradcam(model, images[i + 4].unsqueeze(0), target_layer, model_name)
            cam_image = show_cam_on_image(img, heatmap, use_rgb=True)
            
            if label == pred:
                title_color = 'green'
            else:
                title_color = 'red'
            axes[i, 2].imshow(img)
            axes[i, 2].set_title(f"Image {i+5}\nLabel: {class_names[label]}\nPred: {class_names[pred]}", fontproperties=prop, fontsize=18)
            axes[i, 2].title.set_color(title_color)
            axes[i, 2].axis('off')
            
            if label == pred:
                title_color = 'green'
            else:
                title_color = 'red'
            axes[i, 3].imshow(cam_image)
            axes[i, 3].set_title(f"Grad-CAM {i+5}\nLabel: {class_names[label]}\nPred: {class_names[pred]}", fontproperties=prop, fontsize=18)
            axes[i, 3].title.set_color(title_color)
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    if acc is not None:
        filename = f"random_predictions_{model_name}_epoch_{epoch}_acc_{acc:.4f}.png"
    else:
        filename = f"random_predictions_{model_name}_epoch_{epoch}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    # Keep only the best top 3 random predictions based on accuracy for the same model_name
    saved_files = sorted([f for f in os.listdir(output_dir) if f.startswith(f"random_predictions_{model_name}_")], key=lambda x: float(x.split('_acc_')[-1].split('.png')[0]) if '_acc_' in x else 0, reverse=True)
    for file in saved_files[3:]:
        os.remove(os.path.join(output_dir, file))

def get_target_layer(model, model_name):
    if 'vit_base_patch16_224' in model_name:
        return model.blocks[-1].norm1
    elif 'convnext_base' in model_name:
        return model.stages[-1].downsample
    elif 'resnet' in model_name or 'resnext' in model_name:
        return model.layer4[-1]
    elif 'densenet' in model_name:
        return model.features[-1]
    elif 'caformer_s18' in model_name:
        return model.stages[-1].blocks[-1].norm2
    elif 'fastvit' in model_name:
        return model.stages[-1].blocks[-1].norm
    elif 'efficientnet_b0' in model_name or 'efficientnet_b7' in model_name or 'efficientnet' in model_name:
        return model.conv_head
    elif 'xception' in model_name:
        # Using exit_flow or the last block (usually block12 in standard Xception)
        if hasattr(model, 'exit_flow'):
            return model.exit_flow
        elif hasattr(model, 'block12'):
            return model.block12
        elif hasattr(model, 'bn4'):
            return model.bn4
        else:
            # Fallback to the final convolutional layer
            for i in range(12, 0, -1):
                if hasattr(model, f'block{i}'):
                    return getattr(model, f'block{i}')
            # Last resort fallback
            return model.act4 if hasattr(model, 'act4') else model.block1
    else:
        raise ValueError(f"Model {model_name} not supported for Grad-CAM.")

def plot_gradcam_on_image(model, input_tensor, orig_img, target_layer, target_class, device, model_name=""):
    """
    Generate GradCAM++ heatmap and overlay it on the original image.
    Returns a PIL Image with the overlay.
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    orig_img = orig_img.convert("RGB")
    orig_np = np.array(orig_img).astype(np.float32) / 255.0

    # Use GradCAM++ for CNN models, fallback to GradCAM if shape mismatch
    try:
        if any(cnn_model in model_name for cnn_model in ['resnet', 'resnext', 'efficientnet', 'densenet', 'convnext', 'resnext50_32x4d', 'xception']):
            activations, gradients = register_hooks(model, input_tensor, target_layer)
            # Check shape compatibility for GradCAM++
            if (
                gradients.dim() == 4 and activations.dim() == 4 and
                gradients.shape[2:] == activations.shape[2:]
            ):
                heatmap = generate_gradcam_plus_plus_cnn(activations, gradients, input_tensor)
            else:
                # Fallback to GradCAM if shape mismatch
                heatmap = generate_gradcam_cnn(activations, gradients, input_tensor)
        else:
            heatmap = generate_gradcam(model, input_tensor, target_layer, model_name=model_name)
    except Exception as e:
        print(f"[WARN] GradCAM++ failed ({e}), fallback to GradCAM.")
        heatmap = generate_gradcam(model, input_tensor, target_layer, model_name=model_name)

    # --- Use show_cam_on_image for overlay (like save_random_predictions) ---
    # Normalize orig_np to [0,1] if not already
    if orig_np.max() > 1.0:
        orig_np = orig_np / 255.0

    # Resize heatmap/mask to match orig_np size before blending
    h, w = orig_np.shape[:2]
    if isinstance(heatmap, np.ndarray) and heatmap.ndim == 3 and heatmap.shape[2] == 3:
        # Convert to grayscale mask for show_cam_on_image
        mask = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
        mask = mask.astype(np.float32) / 255.0
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h))
    elif isinstance(heatmap, np.ndarray) and heatmap.ndim == 2:
        mask = heatmap.astype(np.float32) / np.max(heatmap)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h))
    else:
        raise ValueError("Unexpected heatmap shape for GradCAM overlay.")

    # Use show_cam_on_image for blending
    cam_img = show_cam_on_image(orig_np, mask, use_rgb=True)
    overlay_img = Image.fromarray(cam_img)
    return overlay_img
