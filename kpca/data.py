from sklearn import datasets
import torch
import torchvision
import numpy as np

MNIST_DIR ='./'

class Loader:
    def __init__(self, x, batch_size=64):
        self.x = x
        self.batch_size = batch_size
        
    def __iter__(self):
        while True:
            if self.batch_size is None:
                yield x
            else:
                ixs = torch.randperm(len(self.x))[:self.batch_size]
                yield self.x[ixs]

def half_moons(device='cpu',noise=0.1):
    dtype = torch.float32
    ds = datasets.make_moons(n_samples=1600, noise=noise)
    x = torch.tensor(ds[0], dtype=dtype, device=device)
    x = x-x.mean(0)
    labels = torch.tensor(ds[1], dtype=dtype, device=device)
    return x, labels

def mnist(device='cpu'):
    # train/val
    ds = torchvision.datasets.MNIST(MNIST_DIR, download=True, train=True)
    U = ds.data.float().reshape(-1, 1, 28, 28).to(device)
    U /= 78.5675 # normalize by U.std()
    U_train = U[:50000]
    U_val = U[50000:]

    Y = ds.targets.to(device)
    Y_train = Y[:50000]
    Y_val = Y[50000:]

    # test
    ds = torchvision.datasets.MNIST(MNIST_DIR, download=True, train=False)
    U_test = ds.data.float().reshape(-1, 1, 28, 28).to(device)
    U_test /= 78.5675 # normalize by U.std()
    Y_test = ds.targets.to(device)

    return U_train, Y_train, U_val, Y_val, U_test, Y_test