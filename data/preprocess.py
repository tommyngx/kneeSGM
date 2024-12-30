from torchvision import transforms
import yaml
from .augmentations import get_augmentations

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_transforms(image_size, config_path='config/default.yaml'):
    config = load_config(config_path)
    
    train_transform = get_augmentations(config_path, split='train')
    val_transform = get_augmentations(config_path, split='val')

    mean = config['transforms']['normalize']['mean']
    std = config['transforms']['normalize']['std']

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform


