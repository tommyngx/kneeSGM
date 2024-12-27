import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2

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

    # Debugging information
    print(f"Activations shape: {activations.shape}")
    print(f"Pooled gradients shape: {pooled_gradients.shape}")

    for i in range(min(activations.shape[1], pooled_gradients.shape[0])):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.cpu().numpy()  # Move to CPU before converting to NumPy
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

def show_cam_on_image(img, mask, use_rgb=False):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
