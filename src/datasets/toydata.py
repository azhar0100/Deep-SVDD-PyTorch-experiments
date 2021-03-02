from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from base.base_dataset import BaseADDataset as BaseDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
from sklearn.datasets import load_iris
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch


def normalize_to_zero_one(dat):
    return (dat - dat.min(axis=0))/(dat.max(axis=0) - dat.min(axis=0))

def normalize_labels(labels,normal_class):
    if normal_class != -1:
        labels[labels == normal_class] = 0
        labels[labels != normal_class] = 1
    return labels

class Toy_Dataset_Base(Dataset):

    def __init__(self, root: str, normal_class=-1):
        super().__init__()
        self.root = root
        self.iris = load_iris(True)
        self.iris = normalize_to_zero_one(self.iris[0].astype(np.double)),normalize_labels(self.iris[1],normal_class)
        self.normal_class = normal_class

    def __getitem__(self,index):
        datap = torch.Tensor(self.iris[0][index,:].tolist())
        label = np.array([self.iris[1][index]])[0]
        idx = np.array([index])[0]
        return datap,label,idx

    def __len__(self):
        return len(self.iris[1])

class Toy_Dataset_Subset(Dataset):
    def __init__(self,basedata : Toy_Dataset_Base,indices):
        self.base = basedata
        self.indices = indices

    def __getitem__(self,index):
        return self.base[self.indices[index]]

    def __len__(self):
        return len(self.indices)

class Toy_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class = -1):
        super().__init__(root)
        self.base = Toy_Dataset_Base(root,normal_class)
        np.random.seed(0)
        if normal_class == -1:
            indices = np.random.permutation(len(self.base.iris[1]))
            self.train_indices = indices[:100]
            self.test_indices  = indices[100:]
        else:
            indices = np.random.permutation(len(self.base.iris[1]))
            normal_indices = indices[self.base.iris[1][indices] == normal_class]
            self.train_indices = normal_indices[:int(0.8*len(normal_indices))]
            self.test_indices  = indices[~np.isin(indices,self.train_indices)]
            # self.train_indices[0],self.test_indices[0] = self.test_indices[0], self.train_indices[0]
            # self.train_indices[0],self.test_indices[0] = self.test_indices[0], self.train_indices[0]
            # self.train_indices[0],self.test_indices[0] = self.test_indices[0], self.train_indices[0]
        self.train_set = Toy_Dataset_Subset(self.base,self.train_indices)
        self.test_set  = Toy_Dataset_Subset(self.base,self.test_indices )

    def __getitem__(self,index):
        return self.base[index]

    def __len__(self):
        return len(self.base)

