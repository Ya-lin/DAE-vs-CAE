from torchvision import datasets, transforms
from torch.utils.data import random_split

path = "C:/Users/yalin/Documents/Data"
T = transforms.Compose([transforms.ToTensor()])

def MNIST_loader():
    mnist_train = datasets.MNIST(root=path, train=True, download=True, transform=T)
    mnist_test = datasets.MNIST(root=path, train=False, download=True, transform=T)
    return mnist_train, mnist_test


def Split_data(data,ratio):
    n_train = int(len(data)*ratio); n_valid = len(data)-n_train
    train, valid = random_split(data, [n_train, n_valid])
    return train, valid

