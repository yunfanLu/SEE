import torch
from absl.logging import info
from torch import nn

from see.functions.match_mean import search_gamma_to_match_mean


class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        self.eps = torch.tensor(1e-10)

        info(f"Init PSNR:")
        info(f"  Note: the psnr max value is {-10 * torch.log10(self.eps)}")

    def forward(self, x, y):
        d = x - y
        mse = torch.mean(d * d) + self.eps
        psnr = -10 * torch.log10(mse)
        return psnr


class NormalizedPSNR(nn.Module):
    def __init__(self, normalize_type):
        super(NormalizedPSNR, self).__init__()
        assert normalize_type in ["linear", "gamma"]
        self.eps = torch.tensor(1e-10)
        self.normalize_type = normalize_type

    def forward(self, x, y):
        # x: prediction
        # y: ground truth
        if self.normalize_type == "linear":
            x_ = x * y.mean() / x.mean()
        else:  # self.normalize_type == "gamma":
            gamma = search_gamma_to_match_mean(x, y)
            x_ = torch.pow(x, gamma)
        d = x_ - y
        mse = torch.mean(d * d) + self.eps
        psnr = -10 * torch.log10(mse)
        return psnr
