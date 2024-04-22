# File        : imageModel.py
# Date        : 17.4. 2024
# Description : CNN model for visual detection

import torch
import torch.nn as nn
from imageConfig import device


class CNN(nn.Module):
    def __init__(self, num_classes=1, cnn_dropout=0.3, fc_dropout=0.5):
        super(CNN, self).__init__()
        self.features = nn.Sequential(                   # 80x80 input image
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 input channels, 32 output feature maps (because of 32 filters) 
            nn.BatchNorm2d(32),                          # Batch normalization 
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 40x40 feature map
            nn.Dropout(cnn_dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 20x20 feature map
            nn.Dropout(cnn_dropout)
        )
        
        # Calculate the size of the feature vector
        # NOTE: Hand-calculated based on the input size and the operations above
        # 64 filters, 20x20 feature map after 2 max pools
        # Needs to be updated if the architecture changes!
        feature_size = 64 * 20 * 20 

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 128),   # 128 hidden units
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(128, num_classes),
        )
        
        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.sigmoid(x)