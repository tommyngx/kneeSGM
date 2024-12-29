import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
from PIL import Image
from .augmentations import get_augmentations
import yaml
import os

class KneeOsteoarthritisDataset(Dataset):
    def __init__(self, csv_file, split, image_path_column, label_column, dataset_based_link, dataset_name, external_datasets, transform=None, config_path='config/default.yaml'):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['data'] == dataset_name]
        self.data = self.data[self.data['split'] == split]
        
        if split == 'TRAIN' and external_datasets:
            for external_dataset in external_datasets.split(','):
                external_data = pd.read_csv(csv_file)
                external_data = external_data[external_data['data'] == external_dataset.strip()]
                external_data['split'] = 'TRAIN'
                self.data = pd.concat([self.data, external_data])
        
        self.image_path_column = image_path_column
        self.label_column = label_column
        self.dataset_based_link = dataset_based_link
        self.transform = transform or get_augmentations(config_path, split=split.lower())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_based_link, self.data.iloc[idx][self.image_path_column])
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx][self.label_column]
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image, label

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_dataloader(split, batch_size, num_workers, transform=None, config_path='config/default.yaml'):
    config = load_config(config_path)
    csv_file = config['data']['metadata_csv']
    dataset_name = config['data']['dataset_name']
    external_datasets = config['data'].get('external_datasets', None)
    split = config['data'][split.lower() + '_split']
    image_path_column = config['data']['image_path_column']
    label_column = config['data']['label_column']
    dataset_based_link = config['data']['dataset_based_link']
    
    dataset = KneeOsteoarthritisDataset(csv_file, split, image_path_column, label_column, dataset_based_link, dataset_name, external_datasets, transform=transform, config_path=config_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=num_workers)
    return dataloader
