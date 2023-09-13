import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from base import BaseModel

class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        self.resnet50 = resnet50(pretrained=pretrained)
        self.resnet50 = nn.Sequential(*(list(self.resnet50.children())[:-2]))
    def forward(self, x):
        x = self.resnet50(x)
        return x

class FeatureSynthesisModel(nn.Module):
    def __init__(self, embed_dim=300):
        super(FeatureSynthesisModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(1024, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)
    
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x