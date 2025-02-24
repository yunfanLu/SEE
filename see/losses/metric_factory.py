import torch
from absl.logging import info
from torch import nn
from torch.nn.modules.loss import _Loss

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB
from see.losses.psnr import PSNR, NormalizedPSNR
from see.losses.ssim import SSIM


def get_single_metric(config):
    if config.NAME == "PSNR":
        return PSNR()
    elif config.NAME == "PSNR-Linear_N":
        return NormalizedPSNR("linear")
    elif config.NAME == "PSNR-Gamma_N":
        return NormalizedPSNR("gamma")
    elif config.NAME == "SSIM":
        return SSIM()
    else:
        raise ValueError(f"Unknown loss: {config.NAME}")


class EventLowLightBatchMetric(_Loss):
    def __init__(self, configs):
        super(EventLowLightBatchMetric, self).__init__()
        self.metric = get_single_metric(configs)

    def forward(self, batch):
        return self.metric(batch[ELB.PRD], batch[ELB.NL])


class MixedMetric(nn.Module):
    def __init__(self, configs):
        super(MixedMetric, self).__init__()
        self.metric = []
        self.eval = []
        for config in configs:
            self.metric.append(config.NAME)
            self.eval.append(EventLowLightBatchMetric(config))
        info(f"Init Mixed Metric: {configs}")

    def forward(self, batch):
        r = []
        for m, e in zip(self.metric, self.eval):
            r.append((m, e(batch)))
        return r
