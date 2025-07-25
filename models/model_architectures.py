import timm
import torch
import torch.nn as nn
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_model(model_name, config_path='config/default.yaml', pretrained=True):
    if 'fastvit' in model_name:
        model_name = "fastvit_sa12.apple_in1k"
    
    # Set pretrained=False specifically for efficientnet_b7 which doesn't have pretrained weights
    if 'efficientnet_b7' in model_name:
        use_pretrained = False
    else:
        use_pretrained = pretrained
        
    config = load_config(config_path)
    num_classes = len(config['data']['class_labels'])

    # Special handling for dinov2
    if 'dinov2' in model_name:
        # Set image size to 518 for dinov2 large patch14
        model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, img_size=518)
        model = model.eval()
        # If model.head is Identity, replace with a new Linear layer using the last feature dim
        if isinstance(model.head, nn.Identity):
            if hasattr(model, 'norm') and hasattr(model.norm, 'normalized_shape'):
                in_features = model.norm.normalized_shape[0]
            elif hasattr(model, 'pre_logits') and hasattr(model.pre_logits, 'out_features'):
                in_features = model.pre_logits.out_features
            else:
                raise AttributeError("Cannot determine in_features for dinov2 head replacement.")
            model.head = nn.Linear(in_features, num_classes)
        else:
            model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        model = timm.create_model(model_name, pretrained=use_pretrained)
        if 'convnext_base' in model_name:
            model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
        elif 'resnet50' in model_name or 'resnet101' in model_name:
            model.fc = nn.Linear(model.fc.in_features, num_classes) 
        elif 'resnext50_32x4d' in model_name or 'resnext101_32x8d' in model_name:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'vit_base_patch16_224' in model_name:
            model.head = nn.Linear(model.head.in_features, num_classes)
        elif 'densenet121' in model_name or 'densenet169' in model_name or 'densenet201' in model_name or 'densenet161' in model_name:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif 'xception' in model_name:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'caformer_s18' in model_name:
            model.head.fc.fc2 = nn.Linear(model.head.fc.fc2.in_features, num_classes)
        elif 'fastvit' in model_name:
            model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
        elif 'efficientnet_b0' in model_name or 'efficientnet_b7' in model_name or 'efficientnet' in model_name:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported.")
    
    return model
