import torch
import numpy as np
from models.models_v2 import deit_base_patch16_LS
import cv2
from matplotlib import pyplot as plt
import time
import math



#'''
model = deit_base_patch16_LS(img_size=112)

#checkpoint = torch.load('/mnt/HDD1/shih/OSV/deit_osv/pretrain/deit_3_base_224_21k.pth')
#model.load_state_dict(checkpoint["model"])

print("success")

X = torch.zeros([2, 3, 112, 112])

out = model.forward_test(X)

print(out.shape)
#'''
'''
class SinkhornDistance(torch.nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))
        self.actual_nits = actual_nits
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

n = 100
x = torch.linspace(0, 100, n)
mu1 = torch.distributions.Normal(20., 10.).log_prob(x).exp()
mu2 = torch.distributions.Normal(60., 30.).log_prob(x).exp()
cost = (x[None, :]-x[:, None])**2
cost /= cost.max()
mu1 /= mu1.sum()
mu2 /= mu2.sum()
mu1, mu2, cost = mu1.cuda(), mu2.cuda(), cost.cuda()
sinkhorn = SinkhornDistance(eps=1e-3, max_iter=200)

print(mu1.shape, mu2.shape, cost.shape)

mu1_, mu2_, cost_ = mu1.detach().cpu().numpy(), mu2.detach().cpu().numpy(), cost.detach().cpu().numpy()
t = time.time()
dist, _, flow = cv2.EMD(mu1_, mu2_, cv2.DIST_USER, cost_)
#print(flow)
print("--- %s seconds ---" % (time.time() - t))

mu1_, mu2_, cost_ = mu1.cuda(), mu2.cuda(), cost.cuda()
t = time.time()
dist, P, C = sinkhorn(mu1_, mu2_, cost_)
#print(P.detach().cpu().numpy())
print("--- %s seconds ---" % (time.time() - t))

l = SinkhornOT.apply(mu1_.unsqueeze(0), mu2.unsqueeze(0), cost_, 1e-3, 200)
print(l)

#dist = sinkhorn_logsumexp(mu1.cuda(), mu2.cuda(), cost.cuda(), reg=1e-1, maxiter=200, momentum=0.)

fig, ((ax1, ax2)) = plt.subplots(nrows= 1, ncols = 2, figsize=(16, 8))

ax1.set_title('sinkhorn')
ax2.set_title('cv2')
z1_plot = ax1.imshow(P.detach().cpu().numpy())
z2_plot = ax2.imshow(flow)
plt.colorbar(z1_plot, ax=ax1)
plt.colorbar(z2_plot, ax=ax2)
plt.savefig('tmp.png')
plt.close()
'''