import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class ToyNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 2
        self.bn1 = nn.BatchNorm2d(3, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(3, 5, bias=False)
        self.bn2 = nn.BatchNorm2d(5, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(5, self.rep_dim, bias=False)
        self.bn3 = nn.BatchNorm2d(self.rep_dim, eps=1e-04, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.bn3(x)
        # x = torch.nn.functional.sigmoid(x)        
        return x


class ToyNetDecoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 2
        self.bn1 = nn.BatchNorm2d(self.rep_dim, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(self.rep_dim, 5, bias=False)
        self.bn2 = nn.BatchNorm2d(5, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(5,3, bias=False)
        self.bn3 = nn.BatchNorm2d(3, eps=1e-04, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.bn3(x)     
        return x

class ToyAutoEncoder(BaseNet):
    def __init__(self):
        super().__init__()
        self.encoder = ToyNet()
        self.decoder = ToyNetDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
