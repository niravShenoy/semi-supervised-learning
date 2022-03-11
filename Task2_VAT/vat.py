
import torch
import torch.nn as nn
import torch.nn.functional as F

class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter

    def forward(self, model, x):
        r = torch.randn(x.shape).device('cuda' if torch.cuda.is_available() else 'cpu')
        r = F.normalize(r, p=2, dim=1)
        r.requires_grad_(True)

        pred_ul = model(x)
        smax = nn.Softmax(dim=1)
        ul_y = smax(pred_ul)

        for vatIter in range(self.vat_iter):
          adv_x = x + self.xi*r
          adv_pred = model(adv_x)
          adv_y = smax(adv_pred)
          adv_dist = F.kl_div(ul_y, adv_y)
          adv_dist.backward(retain_graph=True)
          adv_grad = r.grad
        
        r_adv = F.normalize(adv_grad, p=2, dim=1)*self.eps
        adv_x = x+r_adv
        adv_pred = model(adv_x)
        adv_y = smax(adv_pred)

        loss = F.kl_div(ul_y, adv_y)
        return loss
        
        raise NotImplementedError