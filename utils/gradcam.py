import torch, os
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def generate_gradcam(model, image, target_layer):
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
    
    if activations.dim() == 4:  # CNN-based models
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(min(activations.shape[1], pooled_gradients.shape[0])):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()

    elif activations.dim() == 3:  # For activations shaped as [channels, height, width]
        # Pooled gradients: Average over height and width dimensions
        pooled_gradients = torch.mean(gradients, dim=[1, 2], keepdim=True)  # Shape: [channels, 1, 1]
        activations *= pooled_gradients  # Element-wise multiplication
        heatmap = torch.mean(activations, dim=0)  # Average across channels (collapse channel dimension)

    else:
        raise ValueError(f"Unexpected activations dimensions: {activations.dim()}") 

    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.cpu().numpy()  # Move to CPU before converting to NumPy
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = 255 - heatmap

    heatmap_colored = np.stack([heatmap] * 3, axis=-1)
    return heatmap_colored #heatmap


def generate_gradcam2(model, image, target_layer):
    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image.requires_grad = True

    activations = []
    gradients = []

    # Hook to capture activations
    def forward_hook(module, input, output):
        activations.append(output)

    # Hook to capture gradients
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])  # Gradients w.r.t. activations

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image)
    forward_handle.remove()

    # Backward pass
    score = output[:, output.max(1)[-1]]
    model.zero_grad()
    score.backward(retain_graph=True)
    backward_handle.remove()

    # Retrieve activations and gradients
    activations = activations[0].detach()  # Shape: [batch_size, num_patches, embedding_dim]
    gradients = gradients[0].detach()  # Shape: [batch_size, num_patches, embedding_dim]

    # Debugging: Log shapes
    #with open('tensor_shapes.txt', "w") as f:
    #    f.write(f"Activations shape: {activations.shape}\n")
    #    f.write(f"Gradients shape: {gradients.shape}\n")

    if activations.dim() == 4:  # CNN-based models
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(min(activations.shape[1], pooled_gradients.shape[0])):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()

    elif activations.dim() == 3:  # ViT models
        # Exclude class token (first patch)
        activations = activations[:, 1:, :]  # Remove class token
        gradients = gradients[:, 1:, :]  # Remove class token

        # Calculate pooled_gradients
        pooled_gradients = torch.mean(gradients, dim=1, keepdim=True)  # Average across patches
        pooled_gradients = pooled_gradients.expand_as(activations)  # Match activations shape

        # Debugging: Log pooled_gradients shape
        #with open('tensor_shapes.txt', "a") as f:
        #    f.write(f"Pooled gradients shape after adjustment: {pooled_gradients.shape}\n")

        # Calculate heatmap for ViT models
        heatmap = torch.sum(activations * pooled_gradients, dim=-1)  # [batch_size, num_patches]

        # Reshape heatmap to spatial dimensions
        grid_size = int(np.sqrt(heatmap.size(1)))  # Compute grid size (e.g., 14x14)
        heatmap = heatmap.view(activations.size(0), grid_size, grid_size)  # Reshape to [batch_size, height, width]

    else:
        raise ValueError("Unexpected activations dimensions.")

    # Post-process heatmap
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.cpu().numpy()  # Move to CPU before converting to NumPy
    heatmap = cv2.resize(heatmap[0], (image.shape[2], image.shape[3]))  # Resize first batch
    heatmap = np.uint8(255 * heatmap)
    heatmap = 255 - heatmap

    heatmap_colored = np.stack([heatmap] * 3, axis=-1)
    return heatmap_colored

def blue_to_gray_np(image: np.ndarray) -> np.ndarray:
    """
    Convert blue areas in an image (NumPy array) to gray.
    
    Args:
        image (np.ndarray): Input image in BGR format as a NumPy array.
    
    Returns:
        np.ndarray: Processed image with blue areas converted to gray.
    """
    if image is None:
        raise ValueError("Input image is None.")
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel color image (BGR).")
    
    # Convert to HSV for easier color range detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the blue range in HSV (adjust values if needed)
    lower_blue = np.array([100, 50, 50])  # Hue: 100-140, Saturation/Value: 50-255
    upper_blue = np.array([140, 255, 255])
    
    # Create a mask for blue areas
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Create a grayscale version of the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale to 3-channel image to match the input format
    gray_3_channel = cv2.merge([gray_image, gray_image, gray_image])
    
    # Replace blue areas with the corresponding grayscale pixels
    result = np.where(blue_mask[:, :, None] == 255, gray_3_channel, image)
    return result

def red_to_gray_np(image: np.ndarray) -> np.ndarray:
    """
    Convert red areas in an image (NumPy array) to gray.
    
    Args:
        image (np.ndarray): Input image in BGR format as a NumPy array.
    
    Returns:
        np.ndarray: Processed image with red areas converted to gray.
    """
    if image is None:
        raise ValueError("Input image is None.")
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel color image (BGR).")
    
    # Convert to HSV for easier color range detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the red range in HSV
    # Red in HSV often spans two ranges due to hue wrap-around (0-10 and 170-180)
    lower_red1 = np.array([0, 50, 50])    # Lower red range 1
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])  # Lower red range 2
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges and combine them
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Create a grayscale version of the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale to 3-channel image to match the input format
    gray_3_channel = cv2.merge([gray_image, gray_image, gray_image])
    
    # Replace red areas with the corresponding grayscale pixels
    result = np.where(red_mask[:, :, None] == 255, gray_3_channel, image)
    return result

def red_to_0_np(image: np.ndarray) -> np.ndarray:
    """
    Convert red areas in an image (NumPy array) to black.
    
    Args:
        image (np.ndarray): Input image in BGR format as a NumPy array.
    
    Returns:
        np.ndarray: Processed image with red areas turned to black.
    """
    if image is None:
        raise ValueError("Input image is None.")
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel color image (BGR).")
    
    # Convert to HSV for easier color range detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the red range in HSV
    # Red in HSV often spans two ranges due to hue wrap-around (0-10 and 170-180)
    lower_red1 = np.array([0, 50, 50])    # Lower red range 1
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])  # Lower red range 2
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges and combine them
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Create a black image to replace red areas
    black_image = np.zeros_like(image)
    
    # Replace red areas with black
    result = np.where(red_mask[:, :, None] == 255, black_image, image)
    return result

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
            heatmap = generate_gradcam(model, images[i].unsqueeze(0), target_layer)
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
                heatmap = generate_gradcam(model, images[i + 4].unsqueeze(0), target_layer)
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
        return model.stages[-1][-1].norm
    elif 'fastvit_t8' in model_name:
        return model.blocks[-1].norm
    else:
        raise ValueError(f"Model {model_name} not supported for Grad-CAM.")
