from see.models.dce import DeepCurveEstimationNet
from see.models.eift import UNetEventLigntEnhance
from see.models.esl import ESLBackBone
from see.models.ev_light import EventGuidedLowLightImageEnhacement
from see.models.evlowlight import EvLowLightNet


def get_model(config):
    if config.NAME == "eSL":
        return ESLBackBone(
            input_frames=config.input_frames,
            event_moments=config.event_moments,
            hidden_channels=config.hidden_channels,
            high_dim_channels=config.high_dim_channels,
            loop=config.loop,
        )
    elif config.NAME == "EvLightNet":
        return EventGuidedLowLightImageEnhacement()
    elif config.NAME == "DeepCurveEstimationNet":
        return DeepCurveEstimationNet()
    elif config.NAME == "EventImageFusionTransformer":
        return UNetEventLigntEnhance(
            in_channels=3, out_channels=3, event_dim=config.event_moments, feature_dim=128, H=config.H, W=config.W
        )
    elif config.NAME == "EvLowLight":
        return EvLowLightNet()
    elif config.NAME == "SEENet":
        from see.models.see_net import SEENet

        return SEENet(
            frames=config.input_frames,
            moments=config.event_moments,
            C1=config.C1,
            C2=config.C2,
            loop=config.loop,
            exposure_sample_num=config.exposure_sample_num,
            SEE_config=config.SEE_config,
        )
    elif config.NAME == "SEENet_R1":
        from see.models.see_net_r1 import SEENetR1

        return SEENetR1(
            frames=config.input_frames,
            moments=config.event_moments,
            C1=config.C1,
            C2=config.C2,
            loop=config.loop,
            exposure_sample_num=config.exposure_sample_num,
            SEE_config=config.SEE_config,
        )
    else:
        raise ValueError(f"Unknown model: {config.NAME}")
