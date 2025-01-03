import timm
import torch
import torch.nn as nn
import yaml
import torch.nn.functional as F

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        gate_outputs = self.fc(x)
        return F.softmax(gate_outputs, dim=1)

class MOEModel(nn.Module):
    def __init__(self, experts, gating_network):
        super(MOEModel, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.gating_network = gating_network
    
    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        flattened_x = torch.flatten(expert_outputs, start_dim=1)
        gate_weights = self.gating_network(flattened_x)
        output = torch.sum(gate_weights.unsqueeze(2) * expert_outputs, dim=1)
        return output

def get_model(model_name, config_path='config/default.yaml', pretrained=True):
    config = load_config(config_path)
    num_classes = len(config['data']['class_labels'])
    
    if config['model']['architecture'].get('MOE', False):
        num_experts = config['model']['architecture']['num_experts']
        experts = [timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes) for _ in range(num_experts)]
        input_dim = experts[0].num_features
        gating_network = GatingNetwork(input_dim, num_experts)
        return MOEModel(experts, gating_network)
    
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
