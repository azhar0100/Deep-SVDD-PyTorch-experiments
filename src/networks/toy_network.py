import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
import logging
logger = logging.getLogger("toy_network.py")

class ToyNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 2
        self.bn1 = nn.BatchNorm1d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4, 100, bias=False)
        self.bn2 = nn.BatchNorm1d(100, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(100, self.rep_dim, bias=False)
        self.bn3 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

    def forward(self, x):
        # x = x.double()
        # logger.info(x.shape)
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
        self.bn1 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(self.rep_dim, 100, bias=False)
        self.bn2 = nn.BatchNorm1d(100, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(100,4, bias=False)
        self.bn3 = nn.BatchNorm1d(4, eps=1e-04, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.bn3(x)
        return x

class ToyNetAutoEncoder(BaseNet):
    def __init__(self):
        super().__init__()
        self.encoder = ToyNet()
        self.decoder = ToyNetDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
