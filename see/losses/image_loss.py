import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class L1CharbonnierLoss(_Loss):
    def __init__(self):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        diff_sq = diff * diff
        error = torch.sqrt(diff_sq + self.eps)
        loss = torch.mean(error)
        return loss


class GradientLoss(_Loss):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, i, j):
        b, c, h, w = i.shape
        idx = torch.abs(i[:, :, :, 1:] - i[:, :, :, : w - 1])
        idy = torch.abs(i[:, :, 1:, :] - i[:, :, : h - 1, :])
        jdx = torch.abs(j[:, :, :, 1:] - j[:, :, :, : w - 1])
        jdy = torch.abs(j[:, :, 1:, :] - j[:, :, : h - 1, :])
        loss = torch.mean(torch.abs(idx - jdx)) + torch.mean(torch.abs(idy - jdy))
        return loss


# Selfconstraints


class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()

        kernel_lf = torch.tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]], dtype=torch.float32)
        kernel_rt = torch.tensor([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]]], dtype=torch.float32)
        kernel_up = torch.tensor([[[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32)
        kernel_dn = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]], dtype=torch.float32)

        self.weight_lf = nn.Parameter(kernel_lf, requires_grad=False)
        self.weight_rt = nn.Parameter(kernel_rt, requires_grad=False)
        self.weight_up = nn.Parameter(kernel_up, requires_grad=False)
        self.weight_dn = nn.Parameter(kernel_dn, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, original, enhanced):
        b, c, h, w = original.shape
        original_mean = torch.mean(original, dim=1, keepdim=True)
        enhanced_mean = torch.mean(enhanced, dim=1, keepdim=True)
        original_pool = self.pool(original_mean)
        enhanced_pool = self.pool(enhanced_mean)

        D_org_lf = F.conv2d(original_pool, self.weight_lf, padding=1)
        D_org_rt = F.conv2d(original_pool, self.weight_rt, padding=1)
        D_org_up = F.conv2d(original_pool, self.weight_up, padding=1)
        D_org_dn = F.conv2d(original_pool, self.weight_dn, padding=1)

        D_enh_lf = F.conv2d(enhanced_pool, self.weight_lf, padding=1)
        D_enh_rt = F.conv2d(enhanced_pool, self.weight_rt, padding=1)
        D_enh_up = F.conv2d(enhanced_pool, self.weight_up, padding=1)
        D_enh_dn = F.conv2d(enhanced_pool, self.weight_dn, padding=1)

        D_lf = torch.pow(D_org_lf - D_enh_lf, 2)
        D_rt = torch.pow(D_org_rt - D_enh_rt, 2)
        D_up = torch.pow(D_org_up - D_enh_up, 2)
        D_dn = torch.pow(D_org_dn - D_enh_dn, 2)

        E = D_lf + D_rt + D_up + D_dn
        E = E.mean()
        return E


# More Sample Constraints


class SEEMoreSampleConstraint(nn.Module):
    def __init__(self, config):
        super(SEEMoreSampleConstraint, self).__init__()
        self.config = config
        self.slw = config.spatial_loss_weight
        self.ecw = config.exposure_constancy_weight
        self.ccw = config.color_constancy_weight
        self.isw = config.ill_smooth_weight

        self.spatial_loss = SpatialConsistencyLoss()
        self.color_constancy = ColorConstancyRegularization()
        self.ill_smooth = IlluminationSmoothnessRegularization()

    def forward(self, nl, sc, ll, nlr, nlr_e):
        """
        nl: normal light. B 3 H W
        sc: self reconstructed. B 3 H W
        ll: low light. B 3 H W
        nlr: normal light reconstructed. B N 3 H W
        nlr_e: enhanced normal light reconstructed. B N
        """
        B, N, C, H, W = nlr.shape
        loss = 0
        # spatial loss
        loss = loss + self.spatial_loss(nl, sc) * self.slw
        for i in range(N):
            loss = loss + self.spatial_loss(nl, nlr[:, i, :, :, :]) * self.slw / N
        # exposure loss
        loss = loss + F.mse_loss(sc.mean([1, 2, 3], keepdim=True), ll.mean([1, 2, 3], keepdim=True)).mean() * self.ecw
        loss = loss + F.mse_loss(nlr.mean([2, 3, 4], keepdim=True), nlr_e).mean() * self.ecw
        # color constancy loss
        loss = loss + self.color_constancy(sc) * self.ccw
        for i in range(N):
            loss = loss + self.color_constancy(nlr[:, i, :, :, :]) * self.ccw / N
        # illumination smoothness loss
        loss = loss + self.ill_smooth(sc) * self.isw
        for i in range(N):
            loss = loss + self.ill_smooth(nlr[:, i, :, :, :]) * self.isw / N
        return loss


# Regularization Items


class ColorConstancyRegularization(_Loss):
    def __init__(self):
        super(ColorConstancyRegularization, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        k = torch.mean(k)
        return k


class ExposureControlRegularization(_Loss):
    def __init__(self, smoothing_kernal_size, expected_exposure_mean):
        super(ExposureControlRegularization, self).__init__()
        assert smoothing_kernal_size % 2 == 1
        assert expected_exposure_mean > 0.4 and expected_exposure_mean < 0.7
        self.pool = nn.AvgPool2d(smoothing_kernal_size)
        self.mean_val = torch.FloatTensor([expected_exposure_mean]).cuda()

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - self.mean_val, 2))
        return d


class IlluminationSmoothnessRegularization(_Loss):
    def __init__(self):
        super(IlluminationSmoothnessRegularization, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h - 1, :]), 2).mean()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w - 1]), 2).mean()
        loss = h_tv + w_tv
        return torch.mean(loss)
