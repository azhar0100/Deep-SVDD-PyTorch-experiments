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
        self.fc1 = nn.Linear(4, 400, bias=False)
        self.bn2 = nn.BatchNorm1d(400, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(400, self.rep_dim, bias=False)
        self.bn3 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

    def forward(self, x):
        x = x.float()
        # logger.info(x.dtype)
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
        self.dbn1 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)
        self.dfc1 = nn.Linear(self.rep_dim, 400, bias=False)
        self.dbn2 = nn.BatchNorm1d(400, eps=1e-04, affine=False)
        self.dfc2 = nn.Linear(400,4, bias=False)
        self.dbn3 = nn.BatchNorm1d(4, eps=1e-04, affine=False)

    def forward(self, x):
        x = self.dbn1(x)
        x = self.dfc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dbn2(x)
        x = self.dfc2(x)
        x = torch.nn.functional.relu(x)
        x = self.dbn3(x)
        return x

class ToyNetAutoEncoder(BaseNet):
    def __init__(self):
        super().__init__()
        self.rep_dim = 2
        self.bn1 = nn.BatchNorm1d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4, 400, bias=False)
        self.bn2 = nn.BatchNorm1d(400, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(400, self.rep_dim, bias=False)
        self.bn3 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        self.rep_dim = 2
        self.dbn1 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)
        self.dfc1 = nn.Linear(self.rep_dim, 400, bias=False)
        self.dbn2 = nn.BatchNorm1d(400, eps=1e-04, affine=False)
        self.dfc2 = nn.Linear(400,4, bias=False)
        self.dbn3 = nn.BatchNorm1d(4, eps=1e-04, affine=False)


    def forward(self, x):
        # x = self.encoder(x)
        # x = self.decoder(x)

        x = x.float()
        x = self.bn1(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.bn3(x)

        x = self.dbn1(x)
        x = self.dfc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dbn2(x)
        x = self.dfc2(x)
        x = torch.nn.functional.relu(x)
        x = self.dbn3(x)


        return x
