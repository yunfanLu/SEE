from absl.logging import info
from torch import nn
from torch.nn.modules.loss import _Loss

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB
from see.losses.image_loss import (
    ColorConstancyRegularization,
    ExposureControlRegularization,
    GradientLoss,
    IlluminationSmoothnessRegularization,
    L1CharbonnierLoss,
    SEEMoreSampleConstraint,
    SpatialConsistencyLoss,
)


def get_single_loss(config):
    if config.NAME == "l1_charbonnier_loss":
        return L1CharbonnierLoss()
    elif config.NAME == "gradient_loss":
        return GradientLoss()
    elif config.NAME == "spatial_consistency_selfconstraints":
        return SpatialConsistencyLoss()
    elif config.NAME == "color_constancy_regularization":
        return ColorConstancyRegularization()
    elif config.NAME == "exposure_control_regularization":
        return ExposureControlRegularization(config.smoothing_kernal_size, config.expected_exposure_mean)
    elif config.NAME == "illumination_smoothness_regularization":
        return IlluminationSmoothnessRegularization()
    elif config.NAME == "see_net_more-sample-constraints":
        return SEEMoreSampleConstraint(config)
    else:
        raise ValueError(f"Unknown loss: {config.NAME}")


class EventLowLightBatchLoss(_Loss):
    def __init__(self, configs):
        super(EventLowLightBatchLoss, self).__init__()
        self.loss_or_regularization = configs.NAME.lower().split("_")[-1]
        self.loss = get_single_loss(configs)

    def forward(self, batch):
        if self.loss_or_regularization == "loss":
            return self.loss(batch[ELB.NL], batch[ELB.PRD])
        elif self.loss_or_regularization == "selfconstraints":
            return self.loss(batch[ELB.LL], batch[ELB.PRD])
        elif self.loss_or_regularization == "regularization":
            return self.loss(batch[ELB.PRD])
        elif self.loss_or_regularization == "more-sample-constraints":
            return self.loss(batch[ELB.NL], batch[ELB.SSR], batch[ELB.LL], batch[ELB.NLR], batch[ELB.NLR_EP])


class MixedLoss(_Loss):
    def __init__(self, configs):
        super(MixedLoss, self).__init__()
        self.loss = []
        self.weight = []
        self.criterion = nn.ModuleList()
        for item in configs:
            self.loss.append(item.NAME)
            self.weight.append(item.WEIGHT)
            self.criterion.append(EventLowLightBatchLoss(item))
        info(f"Init Mixed Loss: {configs}")

    def forward(self, batch):
        name_to_loss = []
        total = 0
        for n, w, fun in zip(self.loss, self.weight, self.criterion):
            tmp = fun(batch)
            name_to_loss.append((n, tmp))
            total = total + tmp * w
        return total, name_to_loss
