import torch
import torch.nn as nn
import torch.optim as optim
import random

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(769, 256)  # First layer: 769 inputs (board + turn), 256 outputs
        self.fc2 = nn.Linear(256, 64)   # Second layer: 256 inputs, 64 outputs
        self.fc3 = nn.Linear(64, 1)     # Third layer: 64 inputs, 1 output

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]  # Ensure x is a tensor, not a tuple
        x = x.view(-1, 769)  # Ensure the input tensor is correctly reshaped to match the first layer's input size
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid to output a probability (useful for binary classification or similar tasks)
        return x

