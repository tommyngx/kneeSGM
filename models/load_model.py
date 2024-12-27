import torch

def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    return model
