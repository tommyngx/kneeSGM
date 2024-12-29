import torch, os
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from .utils import blue_to_gray_np, red_to_gray_np, red_to_0_np

def generate_gradcam(model, image, target_layer, model_name):
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

    with open('tensor_shapes.txt', "w") as f:
        f.write(f"Activations shape: {activations[0].shape}\n")
        #f.write(f"Pooled gradients shape: {pooled_gradients.shape}\n")
    
    if 'cnn' in model_name:
        return generate_gradcam_cnn(activations, gradients, image)
    elif 'vit' in model_name:
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
    return post_process_heatmap(heatmap, image)

def generate_gradcam_caformer(activations, gradients, image):
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
    return post_process_heatmap(heatmap, image)

def post_process_heatmap(heatmap, image):
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = 255 - heatmap
    heatmap_colored = np.stack([heatmap] * 3, axis=-1)
    return heatmap_colored

def generate_gradcam_plus_plus(model, image, target_layer):
    """
    Generate Grad-CAM++ heatmap for a given model, input image, and target layer.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        image (torch.Tensor): Input image tensor of shape (C, H, W) or (1, C, H, W).
        target_layer (torch.nn.Module): The target convolutional layer (e.g., last Conv2D layer in ResNet's Bottleneck).

    Returns:
        np.array: Grad-CAM++ heatmap (H, W, 3) ready to overlay on the input image.
    """
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    image.requires_grad = True

    # Hook to capture activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks for the target layer
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image)
    target_class = output.argmax(dim=1).item()
    score = output[:, target_class]

    # Backward pass
    model.zero_grad()
    score.backward(retain_graph=True)

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Extract activations and gradients
    activations = activations[0].detach()
    gradients = gradients[0].detach()

    # Grad-CAM++ weight computation
    b, k, u, v = gradients.size()
    alpha = gradients.pow(2)
    alpha /= (2 * gradients.pow(2) + (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True) + 1e-8)
    weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

    # Weighted sum of activations
    heatmap = (weights * activations).sum(dim=1).squeeze(0)
    heatmap = F.relu(heatmap)
    heatmap = heatmap / heatmap.max()

    # Convert to NumPy array
    heatmap = heatmap.cpu().numpy()
    heatmap = cv2.resize(heatmap, (image.shape[3], image.shape[2]))

    # Normalize and apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = 255 - heatmap
    #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def show_cam_on_image(img, mask, use_rgb=False):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #cv2.imwrite("abc0.png", heatmap)
    heatmap = red_to_gray_np(heatmap)
    #heatmap = red_to_0_np(heatmap)
    heatmap = np.float32(heatmap) / 255
    
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    
    #cam = 255 - cam
    #cv2.imwrite("abc1.png", cam)
    #cam = red_to_gray_np(cam)
    #cv2.imwrite("abc2.png", cam)
    return cam

def save_random_predictions(model, dataloader, device, output_dir, epoch, class_names, use_gradcam_plus_plus=False, target_layer=None, acc=None):
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
            heatmap = generate_gradcam_plus_plus(model, images[i].unsqueeze(0), target_layer)
        else:
            heatmap = generate_gradcam(model, images[i].unsqueeze(0), target_layer, model_name)
        cam_image = show_cam_on_image(img, heatmap, use_rgb=True)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i+1}\nLabel: {class_names[label]}\nPred: {class_names[pred]}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cam_image)
        axes[i, 1].set_title(f"Grad-CAM {i+1}\nLabel: {class_names[label]}\nPred: {class_names[pred]}")
        axes[i, 1].axis('off')
        
        if i + 4 < len(images):
            img = images[i + 4].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            label = labels[i + 4].item()
            pred = preds[i + 4].item()
            if use_gradcam_plus_plus:
                heatmap = generate_gradcam_plus_plus(model, images[i + 4].unsqueeze(0), target_layer)
            else:
                heatmap = generate_gradcam(model, images[i + 4].unsqueeze(0), target_layer, model_name)
            cam_image = show_cam_on_image(img, heatmap, use_rgb=True)
            
            axes[i, 2].imshow(img)
            axes[i, 2].set_title(f"Image {i+5}\nLabel: {class_names[label]}\nPred: {class_names[pred]}")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(cam_image)
            axes[i, 3].set_title(f"Grad-CAM {i+5}\nLabel: {class_names[label]}\nPred: {class_names[pred]}")
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    if acc is not None:
        filename = f"random_predictions_epoch_{epoch}_acc_{acc:.4f}.png"
    else:
        filename = f"random_predictions_epoch_{epoch}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    # Keep only the best top 3 random predictions based on accuracy
    saved_files = sorted([f for f in os.listdir(output_dir) if f.startswith("random_predictions_epoch_")], key=lambda x: float(x.split('_acc_')[-1].split('.png')[0]) if '_acc_' in x else 0, reverse=True)
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
    elif 'efficientnet_b0' in model_name:
        return model.conv_head
    else:
        raise ValueError(f"Model {model_name} not supported for Grad-CAM.")
