import torch
import torchvision
import numpy as np

MNIST_DIR = '/usr/people/kluther/seungmount/research/kluther/data/mnist'
EMNIST_DIR = '/usr/people/kluther/seungmount/research/kluther/data/emnist'
FASHION_MNIST_DIR = '/usr/people/kluther/seungmount/research/kluther/data/fashion_mnist'
CIFAR_DIR = '/usr/people/kluther/seungmount/research/kluther/data/cifar10'
STL_DIR = '/usr/people/kluther/seungmount/research/kluther/data/stl10'


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

def emnist(device='cpu'):
    # train/val
    ds = torchvision.datasets.EMNIST(EMNIST_DIR, download=True, train=True, split='letters')
    U = ds.data.float().reshape(-1, 1, 28, 28).permute(0,1,3,2).to(device)
    U /= 84.3914 # normalize by U.std()
    Y = ds.targets.to(device) - 1
    
    U_train = U[:100000]
    U_val = U[100000:]

    Y_train = Y[:100000]
    Y_val = Y[100000:]

    # test
    ds = torchvision.datasets.EMNIST(EMNIST_DIR, download=True, train=False, split='letters')
    U_test = ds.data.float().reshape(-1, 1, 28, 28).permute(0,1,3,2).to(device)
    U_test /= 84.3914 # normalize by U.std()
    Y_test = ds.targets.to(device) - 1

    return U_train, Y_train, U_val, Y_val, U_test, Y_test, ds.classes


def fashion_mnist(device='cpu'):
    # train/val
    ds = torchvision.datasets.FashionMNIST(FASHION_MNIST_DIR, download=True, train=True)
    U = ds.data.float().reshape(-1, 1, 28, 28).to(device)
    std = U.float().std()
    
    U /= std # normalize by U.std()
    U_train = U[:50000]
    U_val = U[50000:]

    Y = ds.targets.to(device)
    Y_train = Y[:50000]
    Y_val = Y[50000:]

    # test
    ds = torchvision.datasets.FashionMNIST(FASHION_MNIST_DIR, download=True, train=False)
    U_test = ds.data.float().reshape(-1, 1, 28, 28).to(device)
    U_test /= std # normalize by U.std()
    Y_test = ds.targets.to(device)

    return U_train, Y_train, U_val, Y_val, U_test, Y_test


def cifar(device='cpu',color=True):
    # load 
    ds = torchvision.datasets.CIFAR10(CIFAR_DIR, train=True)
    U = torch.from_numpy(ds.data)
    U = U.float() / 256.0
    if not color:
        U = 0.2989*U[:,:,:,0:1] + 0.5870*U[:,:,:,1:2] + 0.1140*U[:,:,:,2:3]
        
    U = U.permute(0,3,1,2)
    Y = torch.from_numpy(np.array(ds.targets))
    
    # move to gpu
    U = U.to(device)
    Y = Y.to(device)
    
    # split
    return U, Y

def ring(n=64, sig=1.0):
    """Return gaussian bumps on a ring"""
    x = torch.arange(-n,n+1)
    u = torch.zeros(len(x),len(x))
    for i in x:
        u[i] = (-1/2*(x.roll(i.item()+n+1)/sig)**2).exp()
        
    return u

def stl(split='train'):
    # load 
    ds = torchvision.datasets.STL10(STL_DIR, split=split, download=False)
    print('ds loaded')
    U = torch.from_numpy(ds.data)
    U = U.float() / 256.0
    U = 0.2989*U[:,0,:,:] + 0.5870*U[:,1,:,:] + 0.1140*U[:,2,:,:]
    U = U.unsqueeze(1)
    # Y = torch.from_numpy(np.array(ds.labels))
    
    return U