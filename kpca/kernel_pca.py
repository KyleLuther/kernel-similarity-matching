import torch
import torch.nn as nn

class KernelPCA(nn.Module):
    def __init__(self, m=784, n=64, kernel='poly', c=0, d=1, sig=1, lam=1e-3, device='cpu', dtype=torch.float):
        super().__init__()
        self.m = m # input dimensionality
        self.n = n # output dimensionality
        self.lam = lam # regularization parameter
        
        # kernel
        if kernel == 'poly':
            self.f = lambda x,y: self.poly_kernel_(x,y,c,d)
        elif kernel == 'rbf':
            self.f = lambda x,y: self.rbf_kernel_(x,y,sig)
        elif kernel == 'cos':
            self.f = lambda x,y: self.cos_kernel_(x,y,d)
        else:
            raise ValueError('Unrecognized kernel type: {}'.format(kernel))

        # init synapses
        self.q = torch.ones(n, device=device, dtype=dtype, requires_grad=True)
        self.w = torch.randn(n, m, device=device, dtype=dtype, requires_grad=True)
        self.l = torch.eye(n, device=device, dtype=dtype, requires_grad=True)

    def rbf_kernel_(self,x,y,sig=1):
        d2 = (x**2).sum(-1).unsqueeze(-1) + (y**2).sum(-1).unsqueeze(-2) - 2 * x @ y.transpose(-1,-2)
        return (-1/2*d2/sig**2).exp()
    
    def poly_kernel_(self,x,y,c=0,d=1):
        return (x@y.t()+c)**d
    
    def cos_kernel_(self,x,y,d):
        return x.norm(dim=1).view(-1,1) * y.norm(dim=1).view(1,-1) * (cos_mat(x,y)**d)
    
    def energy(self, x, y):
        cyy = (y.t() @ y) / y.shape[0]
        cyb = (y*self.f(x,self.w)).mean(0)
        
        e1 = (self.q*cyb).sum() - 1/2*(self.q**2 * self.f(self.w,self.w).diag()).sum()
        e2 = (self.l*cyy - 1/2*self.l**2).sum()
        e3 = (self.lam*cyy.diag()).sum()
        
        return e1 - 1/2*e2 - 1/2*e3
    
    def forward(self, x):
        b = self.f(x,self.w)*self.q
        eye = torch.eye(self.l.shape[0], device=x.device, dtype=x.dtype)
        l_inv = (self.l+self.lam*eye).inverse()

        return (b @ l_inv).detach()

    def train(self,loader,etaw=0.1,etaq=0.1,etal=0.1,n_iter=1000,print_iter=10,info=None):
        # logging utils
        if info is None:
            info = {'es': [], 'etaqs': [], 'etaws': [], 'etals': [], 'gqs': [], 'gws': [], 'gls': [], 'ts': []}
        t0 = time.time()
        
        # train loop
        for i in range(n_iter):
            # inference
            x = next(loader)
            y = self.forward(x)
            
            # gradients
            e = self.energy(x,y)
            gq, gw, gl = torch.autograd.grad(e, [self.q, self.w, self.l])
            
            # updates
            with torch.no_grad():
                self.q += etaq * gq
                self.w += etaw * gw / (self.q**2).unsqueeze(1)
                self.l -= etal * gl

            # log
            info['es'].append(e.item())
            info['gqs'].append(gq.abs().mean().item())
            info['gws'].append(gw.abs().mean().item())
            info['gls'].append(gl.abs().mean().item())
            info['ts'].append(time.time()-t0)
            
            # print
            if print_iter is not None and i % (n_iter // print_iter) == 0:
                print('{}/{}: e={:.4f}, t={:.4f}'.format(i, n_iter, info['es'][-1], info['ts'][-1]))
                
        return info