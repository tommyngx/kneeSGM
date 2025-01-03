from albumentations import Compose, HorizontalFlip, VerticalFlip, Rotate, ColorJitter, RandomCrop, Normalize, Resize, RandomScale, PadIfNeeded, CLAHE
from albumentations.pytorch import ToTensorV2
import yaml
import cv2
import albumentations as A

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_augmentations(config_path='config/default.yaml', split='train'):
    config = load_config(config_path)
    augmentations = []
    
    # Always apply resize and normalize
    augmentations.append(Resize(
        height=config['data']['augmentations']['resize']['height'],
        width=config['data']['augmentations']['resize']['width']
    ))

    augmentations.append(Normalize(
        mean=config['data']['augmentations']['normalize']['mean'],
        std=config['data']['augmentations']['normalize']['std'],
        p=config['data']['augmentations']['normalize']['p']
    ))
    
    if config['data']['augmentations']['clahe']['enabled']:
        augmentations.append(CLAHE(
            clip_limit=config['data']['augmentations']['clahe']['clip_limit'],
            tile_grid_size=(config['data']['augmentations']['clahe']['tile_grid_size'], config['data']['augmentations']['clahe']['tile_grid_size']),
            p=config['data']['augmentations']['clahe']['p']
        ))

    if split == 'train':
        if config['data']['augmentations']['horizontal_flip']['enabled']:
            augmentations.append(HorizontalFlip(p=config['data']['augmentations']['horizontal_flip']['p']))
        if config['data']['augmentations']['vertical_flip']['enabled']:
            augmentations.append(VerticalFlip(p=config['data']['augmentations']['vertical_flip']['p']))

        if config['data']['augmentations']['zoom_out']['enabled']:
            scale_limit = config['data']['augmentations']['zoom_out']['scale_limit']
            probability = config['data']['augmentations']['zoom_out']['p']
            original_height = config['data']['augmentations']['resize']['height']
            original_width = config['data']['augmentations']['resize']['width']

            augmentations.append(RandomScale(scale_limit=(-scale_limit, 0), p=probability))
            augmentations.append(PadIfNeeded(min_height=original_height, min_width=original_width, border_mode=cv2.BORDER_CONSTANT, value=0))

        if config['data']['augmentations']['rotate']['enabled']:
            augmentations.append(Rotate(
                limit=config['data']['augmentations']['rotate']['limit'], 
                p=config['data']['augmentations']['rotate']['p'],
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ))
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
    
    augmentations.append(ToTensorV2())
    return Compose(augmentations)
