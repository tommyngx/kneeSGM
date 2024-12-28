import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum().item()
    return correct / target.size(0)

def f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        return f1_score(target.cpu(), pred.cpu(), average='weighted')

def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        return precision_score(target.cpu(), pred.cpu(), average='weighted', zero_division=0)

def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        return recall_score(target.cpu(), pred.cpu(), average='weighted', zero_division=0)
