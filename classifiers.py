import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from .utils import cos_mat, euc_mat, tovec, track_grad_


class NearestNeighborClassifier:
    def __init__(self, distance_fn, k):
        if distance_fn == 'cosine':
            self.distance_fn = lambda x,y: -cos_mat(x,y)
        elif distance_fn == 'euclidean':
            self.distance_fn = euc_mat
        else:
            raise ValueError('Unrecognized distance')
        
        self.k = k
        
    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def score(self, x_test, y_test):
        ds = self.distance_fn(self.x_train, x_test)
        ps = self.y_train[ds.min(dim=0)[1]]
        acc = (y_test == ps).float().mean().item()
        return acc
    

class LogisticRegressionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, lam, center=False, scale=False, n_iter=1000):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        with torch.no_grad():
            self.linear.weight.zero_()
        self.lam = lam
        
        self.n_iter = n_iter
        self.eta = 0.1
        
        self.center = center
        self.scale = scale 
        
        self.mean = None
        self.std = None
        
    def train(self, x, y, print_iter=None):
        assert(x.shape[0] == y.shape[0])
        assert(len(x.shape) == 2)
        
        # rescale x
        self.mean = x.mean(0) if self.center else 0.0 
        self.std = x.std(0).clamp(0.01) if self.scale else 1.0 
        
        x = x - self.mean
        x = x / self.std
        
        # logging 
        t0 = time.time()
        info = {'es': [], 'etas': []}
        
        # init 
        eta = self.eta
        w0 = self.linear.weight
        e0 = F.cross_entropy(x @ w0.t(),y) + self.lam/2 * (w0**2).sum()
        e0.backward()
        
        # train
        for i in range(self.n_iter):
            # try update
            with torch.no_grad():
                w1 = w0 - eta*w0.grad
                track_grad_(w1)
                
            # if lower energy, accept. otherwise try lower learning rate
            e1 = F.cross_entropy(x @ w1.t(),y) + self.lam/2 * (w1**2).sum()
            if e1 < e0:
                e0 = e1
                e0.backward()
                w0.data = w1.data
                w0.grad = w1.grad
                eta *= 1.1
            else:
                eta /= 2.0
                
            # convergence check
            if eta < 1e-10:
                break
            
            # print
            if print_iter is not None and i % print_iter == 0:
                t1=time.time()
                print('{}/{}: e={:.4f}, t={:.4f}'.format(i, self.n_iter, e0.item(), t1-t0))
                t0=t1

            # log 
            info['es'].append(e0.item())
            info['etas'].append(eta)
            
        with torch.no_grad():
            self.linear.weight = w0
        return info

    def score(self, x, y):
        x = x - self.mean
        x = x / self.std
        
        p = self.linear(x)
        return (p.max(dim=1)[1] == y).float().mean().item()

def accuracy_curve(x_train, y_train, x_test, y_test, ks, CLF, n_trials=5, compute_trn=True):
    assert(x_train.shape[0] == y_train.shape[0])
    assert(x_test.shape[0] == y_test.shape[0])
    x_train, x_test = tovec(x_train), tovec(x_test)
    
    accs_trn, accs_test = [], []
    for k in ks:
        accs_trn_, accs_test_ = [], []
        for i in range(min(n_trials,x_train.shape[0]//k)):
            ixs = torch.randperm(x_train.shape[0])[:k]
            clf = CLF()
            clf.train(x_train[ixs], y_train[ixs])
            if compute_trn:
                accs_trn_.append(clf.score(x_train[ixs],y_train[ixs]))
            accs_test_.append(clf.score(x_test,y_test))
        accs_trn.append(np.mean(accs_trn_))
        accs_test.append(np.mean(accs_test_))

    return accs_trn, accs_test