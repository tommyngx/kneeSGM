import torch, os
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
#from pytorch_grad_cam import GradCAM
#from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

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
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = features[0].detach()

    for i in range(min(activations.shape[1], pooled_gradients.shape[0])):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.cpu().numpy()  # Move to CPU before converting to NumPy
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = 255 - heatmap

    #heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    #print(' step 1' ,heatmap.shape)
    heatmap_colored = np.stack([heatmap] * 3, axis=-1)
    #heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    #heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_HSV2BGR)
    #heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    #heatmap_colored = heatmap_colored.astype(np.uint8)
    #cv2.imwrite("abc.png", heatmap_colored)
    #heatmap_colored = cv2.imread("abc.png")
    #heatmap_colored = blue_to_gray_np(heatmap_colored)

    #print("RGB/BGR pixel value:", heatmap_colored[0, 0])
    #print(' step 2' ,np.unique(heatmap_colored))
    #print(' step 4' ,heatmap_colored.shape)
    #blue_mask = (heatmap_colored[:, :, 0] > 128) & (heatmap_colored[:, :, 1] < 50) & (heatmap_colored[:, :, 2] < 50)
    #gray_value = (0.3 * heatmap_colored[blue_mask, 2] + 
    #          0.59 * heatmap_colored[blue_mask, 1] + 
    #          0.11 * heatmap_colored[blue_mask, 0]).astype(np.uint8)
    #heatmap_colored[blue_mask] = np.stack([gray_value, gray_value, gray_value], axis=-1)

    #print("afterww",heatmap.shape)
    #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #print("after",heatmap.shape)
    return heatmap_colored #heatmap

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


def generate_gradcam_ori(model, image, target_layer):
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
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = features[0].detach()

    for i in range(min(activations.shape[1], pooled_gradients.shape[0])):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.cpu().numpy()  # Move to CPU before converting to NumPy
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3]))
    heatmap = heatmap / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap


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
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    
    #cam = 255 - cam
    cv2.imwrite("abc1.png", cam)
    cam = blue_to_gray_np(cam)
    cv2.imwrite("abc2.png", cam)
    return cam

def save_random_predictions(model, dataloader, device, output_dir, epoch, class_names, use_gradcam_plus_plus=False):
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
            heatmap = generate_gradcam_plus_plus(model, images[i].unsqueeze(0), model.layer4[-1])
        else:
            heatmap = generate_gradcam(model, images[i].unsqueeze(0), model.layer4[-1])
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
                heatmap = generate_gradcam_plus_plus(model, images[i + 4].unsqueeze(0), model.layer4[-1])
            else:
                heatmap = generate_gradcam(model, images[i + 4].unsqueeze(0), model.layer4[-1])
            cam_image = show_cam_on_image(img, heatmap, use_rgb=True)
            
            axes[i, 2].imshow(img)
            axes[i, 2].set_title(f"Image {i+5}\nLabel: {class_names[label]}\nPred: {class_names[pred]}")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(cam_image)
            axes[i, 3].set_title(f"Grad-CAM {i+5}\nLabel: {class_names[label]}\nPred: {class_names[pred]}")
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"random_predictions_epoch_{epoch}.png"))
    plt.close()
    
    # Keep only the last 3 latest saved epochs
    saved_files = sorted([f for f in os.listdir(output_dir) if f.startswith("random_predictions_epoch_")], reverse=True)
    for file in saved_files[3:]:
        os.remove(os.path.join(output_dir, file))
