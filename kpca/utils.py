import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

def toim(u,ch=1):
    """ Reshapes vector into square image
    Args:
        u: tensor (n,m), input vector
        ch: int, number of output channels
    Returns:
        u: tensor (n,ch,s,s) where s = sqrt(m//ch)
    """
    s = int(np.sqrt(u.shape[1]//ch))
    return u.view(u.shape[0],ch,s,s)

def tovec(u):
    """ Reshapes image into vector
    Args:
        u: tensor (n,ch,h,w), input vector
    Returns:
        u: tensor (n,m) where m=ch*h*w
    """
    return u.reshape(u.shape[0], -1)

def ondiag(M):
    """ Returns diagonal elements of tensor M """
    return M[torch.eye(M.shape[0]).bool()]

def offdiag(M):
    """ Return off diagonal elements of tensor M """
    return M[~torch.eye(M.shape[0]).bool()]

def normdiag(M, eps=0.001):
    """ Normalizes M to have unit diagonal"""
    return M / (M.diag().view(1,-1)*M.diag().view(-1,1)).sqrt().clamp(eps)

def cor_mat(x,y):
    """ Returns correlation matrix """
    return x.t() @ y / x.shape[0]

def cos_mat(x,y,eps=0.001):
    """ Returns cosine similarity matrix """
    x = x / x.norm(dim=1,keepdim=True).clamp(eps)
    y = y / y.norm(dim=1,keepdim=True).clamp(eps)
    return x @ y.t()

def euc_mat(x,y):
    """ Returns euclidean distance matrix """
    dot = x @ y.t() # TxM
    xn = (x**2).sum(1, keepdim=True)
    yn = (y**2).sum(1, keepdim=True)
    return xn + yn.t() - 2*dot

def wiener_filter(u,x,tau=0.1):
    """ Wiener filter (linear filter to predict x from u) """
    cuu = cor_mat(u,u)
    cxu = cor_mat(x,u)

    v, l = torch.symeig(cuu, eigenvectors=True)
    cuu_inv = l @ (v.clamp(tau)**-1).diag() @ l.t()
    return cxu @ cuu_inv

def sta(u,x,tau=0.1):
    """ spike triggered average """
    cxu = cor_mat(x,u)
    
    if tau is not None:
        cuu = cor_mat(u,u)
        v, l = torch.symeig(cuu, eigenvectors=True)
        cuu_inv = l @ (v.clamp(tau)**-1).diag() @ l.t()
        cxu = cxu @ cuu_inv
        
    return cxu

def stc(u,x,tau=0.1):
    """ spike triggered covariance """
    cxuu = torch.einsum('ti,ta,tb->iab',x,u,u)/x.shape[0]

    if tau is not None:
        cuu = cor_mat(u,u)
        v, l = torch.symeig(cuu, eigenvectors=True)
        cuu_inv = l @ (v.clamp(tau)**-1).diag() @ l.t()
        cxuu = torch.einsum('iab,bc->iac',cxuu,cuu_inv)
        
    return cxuu

######################
# Optimization Utils #
######################
def track_grad_(M):
    M.requires_grad_()
    M.retain_grad()
    M.grad = torch.zeros_like(M)
    
class AutoMinimizer:
    def __init__(self, eta=0.01, n_iter=1000, eta_plus=1.1, eta_minus=0.5, check_energy=True, non_neg=True, print_iter=100):
        self.eta = eta
        self.n_iter = n_iter
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.check_energy = check_energy
        self.non_neg = non_neg
        self.print_iter = print_iter
        
    def minimize(self, energy, x0):
        """ Return local min of energy(x), initialization is x0
        Auto-tunes learning rate
        Args:
            u: tensor (n, *), input
            x0: tensor (n, *), initial guess for encoding. If None, initialize with zeros
        Returns:
            x: tensor (n, *), encoding
            info: dict, contains optimization info
        """
        # initialize x
        track_grad_(x0)

        # logging
        info = {'es': [], 'etas': []}

        # don't modify eta
        eta = self.eta

        x0.grad.zero_()
        e0 = energy(x0)
        e0.backward()
        t0 = time.time()
        for i in range(self.n_iter):
            # compute trial guess
            with torch.no_grad():
                x1 = x0 - eta*x0.grad
                if self.non_neg:
                    x1.clamp_(0.0)
                track_grad_(x1)

            # if lower energy, accept. otherwise try lower learning rate
            e1 = energy(x1)
            if self.check_energy:
                if e1 < e0:
                    e0 = e1
                    e0.backward()
                    x0.data = x1.data
                    x0.grad = x1.grad
                    eta *= self.eta_plus
                else:
                    eta *= self.eta_minus
                    
                if e1 == 0:
                    print('equl')
            
            else:
                e0 = e1
                e0.backward()
                x0.data = x1.data
                x0.grad = x1.grad
                
            # convergence
            if eta < 1e-10:
                break

            # log
            info['es'].append(e0.item())
            info['etas'].append(eta)
            
            if self.print_iter is not None and i % self.print_iter == 0:
                print('iter {}/{}: e={}, eta={}, t={}'.format(i,self.n_iter, e0.item(), eta, time.time()-t0))
                
        x0 = x0.detach()
        return x0, info

#######################
# Visualization Utils #
#######################
def show_grid(M,figsize=(8,8),dpi=150,**kwargs):
    if 'norm_every' in kwargs:
        if kwargs['norm_every']:
            M = M.clone()
            for i in range(M.shape[0]):
                M[i] -= M[i].min()
                M[i] /= M[i].max().clamp(0.001)
        del kwargs['norm_every']
#     if 'shuffle' in kwargs:
#         if kwargs['shuffle']:
#             M=M[torch.randperm(M.shape[0])]
#         del kwargs['shuffle']
    if 'normalize' not in kwargs:
        kwargs['normalize'] = True
    if 'pad_value' not in kwargs:
        kwargs['pad_value'] = 0.25
    grid = make_grid(M,**kwargs).permute(1,2,0).detach().cpu()
    plt.figure(figsize=figsize,dpi=dpi)
    plt.imshow(grid)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def imsave(img,fname,normalize=True):
    if normalize:
        img = img - img.min()
        img = img / img.max()

    # format 
    img = img * 255
    img = img.byte()
    img = img.detach().cpu().numpy()
    
    img = Image.fromarray(img)
    img.save(fname)
    
# def imsave(img,fname,normalize=True):
#     plt.
#     if normalize:
#         img = img - img.min()
#         img = img / img.max()

#     # format 
#     img = img * 255
#     img = img.byte()
#     img = img.detach().cpu().numpy()
    
#     img = Image.fromarray(img)
#     img.save(fname)

####################
# image transforms #
####################
def affine(batch, rotate=0.0, scale_x=1.0, scale_y=1.0, shift_x=0.0, shift_y=0.0):
    # setup
    N = batch.shape[0]
    device = batch.device
    theta = torch.zeros(N,2,3,device=device)
    theta[:,0,0] = 1.0
    theta[:,1,1] = 1.0
    
    # rotate
    rotate = torch.tensor(rotate) if type(rotate) is not torch.Tensor else rotate
    phi = rotate * np.pi / 180.0
    
    trans = torch.zeros(N,2,2,device=device)
    trans[:,0,0] = phi.cos()
    trans[:,0,1] = -phi.sin()
    trans[:,1,0] = phi.sin()
    trans[:,1,1] = phi.cos()
    theta = torch.einsum('bij,bki->bkj', theta, trans)
    
    # scale
    trans = torch.zeros(N,2,2,device=device)
    trans[:,0,0] = scale_x
    trans[:,1,1] = scale_y
    theta = torch.einsum('bij,bki->bkj', theta, trans)
    
    # translate
    shift_x = shift_x / (batch.shape[2] // 2)
    shift_y = shift_y / (batch.shape[3] // 2)

    trans = torch.zeros(N,2,3,device=device)
    trans[:,0,2] = shift_x
    trans[:,1,2] = shift_y
    theta = theta + trans
    
    ##############
    # apply grid #
    ##############
    grid = F.affine_grid(theta, batch.shape, align_corners=False)
    aug = F.grid_sample(batch, grid, align_corners=False)
    
    return aug

def random_affine(batch, max_rotate=0.0, min_scale=1.0, max_scale=1.0, max_shift=0.0):
    N = batch.shape[0]
    device = batch.device
    rotate = (torch.rand(N,device=device) - 0.5) * 2 * max_rotate
    scale_x = min_scale+torch.rand(N,device=device)*(max_scale-min_scale)
    scale_y = min_scale+torch.rand(N,device=device)*(max_scale-min_scale)
    shift_x = (torch.rand(N,device=device) - 0.5) * 2 * max_shift
    shift_y = (torch.rand(N,device=device) - 0.5) * 2 * max_shift
    
    return affine(batch, rotate, scale_x, scale_y, shift_x, shift_y)