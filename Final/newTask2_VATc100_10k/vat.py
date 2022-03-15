
import torch
import torch.nn as nn
import torch.nn.functional as F

def l2_norm(c):
    c_reshape = c.view(c.shape[0], -1, *(1 for _ in range(c.dim() - 2)))
    c /= torch.norm(c_reshape, dim=1, keepdim=True) + 1e-8
    return c

class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter

    def forward(self, model, x):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        r = torch.randn(x.shape).to(device)
        r = l2_norm(r)
        pred = F.softmax(model(x), dim=1)
        
        for num in range(self.vat_iter):
            r.requires_grad_(True)
            advEx = x + self.xi*r
            advPred = F.softmax(model(advEx), dim=1)
            adv_dist = F.kl_div(pred, advPred)
            adv_dist.backward(retain_graph=True)
            d = r.grad
            model.zero_grad()
        
        r_adv = l2_norm(d) * self.eps
        adv_pred = F.softmax(model(x + r_adv), dim=1)
        loss = F.kl_div(pred, adv_pred)
        return loss    
            
        raise NotImplementedError