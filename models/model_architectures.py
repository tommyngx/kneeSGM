import timm
import torch
import torch.nn as nn
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class MOEModel(nn.Module):
    def __init__(self, experts, gating_model, num_classes):
        super(MOEModel, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.gating_model = gating_model
        self.num_classes = num_classes
        self.dense_layer = nn.Linear(experts[0].num_features, 64)
        self.output_layer = nn.Linear(64, num_classes)
    
    def forward(self, x):
        expert_outputs = [self.dense_layer(expert(x).view(x.size(0), -1)) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gating_weights = self.gating_model(x)
        gating_weights = gating_weights.unsqueeze(-1)
        mixture_output = torch.sum(expert_outputs * gating_weights, dim=1)
        output = self.output_layer(mixture_output)
        return output

def get_model(model_name, config_path='config/default.yaml', pretrained=True):
    config = load_config(config_path)
    num_classes = len(config['data']['class_labels'])
    
    if config['model']['architecture'].get('MOE', False):
        expert_names = config['model']['architecture']['expert_models']
        experts = [timm.create_model(name, pretrained=pretrained, num_classes=num_classes) for name in expert_names]
        gating_model = nn.Sequential(
            nn.Linear(experts[0].num_features, len(experts)),
            nn.Softmax(dim=-1)
        )
        return MOEModel(experts, gating_model, num_classes)
    
    if 'fastvit' in model_name:
        model_name = "fastvit_sa12.apple_in1k"
    model = timm.create_model(model_name, pretrained=pretrained)
    
    if 'convnext_base' in model_name:
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    elif 'resnet50' in model_name or 'resnet101' in model_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes) 
    elif 'resnext50_32x4d' in model_name or 'resnext101_32x8d' in model_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'vit_base_patch16_224' in model_name:
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif 'densenet' in model_name:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'caformer_s18' in model_name:
        model.head.fc.fc2 = nn.Linear(model.head.fc.fc2.in_features, num_classes)
    elif 'fastvit' in model_name:
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    elif 'efficientnet_b0' in model_name:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    return model
