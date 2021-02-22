from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
from sklearn.datasets import load_iris
import torchvision.transforms as transforms


class Toy_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)
        self.iris = load_iris(True)

    def __getitem__(self,index):
        datap = self.iris[0][index,:]
        label = self.iris[1][index]
        idx = index
        return idx,datap,label
