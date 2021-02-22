from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms


class Toy_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)
        a = np.arange(5000)
        b = 2*a + 5
        c = a.copy()
        c[:len(c)//2] = np.exp(c[:len(c)//2]/5000)
        c[len(c)//2:] = np.sin(c[len(c)//2:])
        d = 3*a
        self.a = a
        self.npdat = np.vstack([b,c,d]).T

    def __getitem__(self,index):
        label = self.a[index] < 3500 and self.a[index] > 2000
        idx = self.a[index]
        datap = self.npdat[index,:]
        return idx,datap,label
