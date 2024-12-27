from albumentations import Compose, HorizontalFlip, VerticalFlip, Rotate, ColorJitter, RandomCrop, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_augmentations(config_path='config/default.yaml'):
    config = load_config(config_path)
    augmentations = []
    
    if config['data']['augmentations']['horizontal_flip']['enabled']:
        augmentations.append(HorizontalFlip(p=config['data']['augmentations']['horizontal_flip']['p']))
    if config['data']['augmentations']['vertical_flip']['enabled']:
        augmentations.append(VerticalFlip(p=config['data']['augmentations']['vertical_flip']['p']))
    if config['data']['augmentations']['rotate']['enabled']:
        augmentations.append(Rotate(limit=config['data']['augmentations']['rotate']['limit'], p=config['data']['augmentations']['rotate']['p']))
    if config['data']['augmentations']['color_jitter']['enabled']:
        augmentations.append(ColorJitter(
            brightness=config['data']['augmentations']['color_jitter']['brightness'],
            contrast=config['data']['augmentations']['color_jitter']['contrast'],
            saturation=config['data']['augmentations']['color_jitter']['saturation'],
            hue=config['data']['augmentations']['color_jitter']['hue'],
            p=config['data']['augmentations']['color_jitter']['p']
        ))
    if config['data']['augmentations']['random_crop']['enabled']:
        augmentations.append(RandomCrop(
            height=config['data']['augmentations']['random_crop']['height'],
            width=config['data']['augmentations']['random_crop']['width'],
            p=config['data']['augmentations']['random_crop']['p']
        ))
    if config['data']['augmentations']['normalize']['enabled']:
        augmentations.append(Normalize(
            mean=config['data']['augmentations']['normalize']['mean'],
            std=config['data']['augmentations']['normalize']['std'],
            p=config['data']['augmentations']['normalize']['p']
        ))
    if config['data']['augmentations']['resize']['enabled']:
        augmentations.append(Resize(
            height=config['data']['augmentations']['resize']['height'],
            width=config['data']['augmentations']['resize']['width']
        ))
    
    augmentations.append(ToTensorV2())
    return Compose(augmentations)
