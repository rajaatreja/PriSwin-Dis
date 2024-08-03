import torch
import torch.nn as nn
import torchvision.models as models

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Load a pre-existing ResNet model
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify the first convolutional layer to accept 1 channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Replace the final fully connected layer to output a single value
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        
    def forward(self, x):
        x = self.resnet(x)
        return torch.sigmoid(x)