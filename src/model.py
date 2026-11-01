import torch.nn as nn
from torchvision import models

def get_model():
    # Load ResNet18 with pretrained weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze all layers in the network
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final fully connected layer
    # ResNet18's final layer is called 'fc'
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # 2 classes: 0=Cracked, 1=Uncracked
    
    return model