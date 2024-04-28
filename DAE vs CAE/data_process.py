
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import random_split


path = Path.home().joinpath("Documents","Data")

def get_mnist():
    mnist_train = datasets.MNIST(root=path, train=True, download=True, 
                                 transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root=path, train=False, download=True, 
                                transform=transforms.ToTensor())
    return mnist_train, mnist_test

def split_data(data, ratio):
    n_train = int(len(data)*ratio); n_valid = len(data)-n_train
    train, valid = random_split(data, [n_train, n_valid])
    return train, valid


