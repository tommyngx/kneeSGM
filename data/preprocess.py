from torchvision import transforms
import yaml
from .augmentations import get_augmentations

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_transforms(image_size, config_path='config/default.yaml'):
    config = load_config(config_path)
    mean = config['transforms']['normalize']['mean']
    std = config['transforms']['normalize']['std']
    
    train_transform = get_augmentations(config_path, split='train')
    val_transform = get_augmentations(config_path, split='val')

    return train_transform, val_transform
