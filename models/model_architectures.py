import timm
import torch.nn as nn
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_model(model_name, config_path='config/default.yaml', pretrained=True):
    if 'fastvit' in model_name:
        model_name = "fastvit_sa12.apple_in1k"
    config = load_config(config_path)
    num_classes = len(config['data']['class_labels'])
    model = timm.create_model(model_name, pretrained=pretrained)
    
    if 'convnext_base' in model_name:
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    elif 'resnet50' in model_name or 'resnet101' in model_name: # test ok
        model.fc = nn.Linear(model.fc.in_features, num_classes) 
    elif 'resnext50_32x4d' in model_name or 'resnext101_32x8d' in model_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'vit_base_patch16_224' in model_name:
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif 'densenet' in model_name:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'caformer_s18' in model_name:
        #print(model.head) 
        model.head.fc.fc2 = nn.Linear(model.head.fc.fc2.in_features, num_classes)
    elif 'fastvit' in model_name:
        #print(model.head) 
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    elif 'efficientnet_b0' in model_name:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    return model
