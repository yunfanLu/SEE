import torch
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import nn
from torch.nn import functional as F

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB

"""
@inproceedings{10.1609/aaai.v37i2.25257,
    author = {Liu, Lin and An, Junfeng and Liu, Jianzhuang and Yuan, Shanxin and Chen, Xiangyu and Zhou, Wengang and Li, Houqiang and Wang, Yan Feng and Tian, Qi},
    title = {Low-light video enhancement with synthetic event guidance},
    year = {2023},
    publisher = {AAAI Press},
    booktitle = {Proceedings of the Thirty-Seventh AAAI Conference},
}
"""


class EIRTBlock(nn.Module):
    def __init__(self, in_dim, transformer_dim, H, W):
        super(EIRTBlock, self).__init__()

        C = transformer_dim
        self.f_1 = nn.Sequential(
            nn.Conv2d(in_dim, C, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([C, H, W]),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            Rearrange("b c h w -> b (h w) c"),
        )

        self.f_2 = nn.Sequential(
            nn.Conv2d(in_dim, C, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([C, H, W]),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

        self.f_3 = nn.Sequential(
            Rearrange("b c h w -> b c (h w)"),
            nn.Linear(H * W, H * W),  # 1x1, todo 3x3 ?
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

        self.f_4 = nn.Sequential(
            Rearrange("b c h w -> b h w c"),
            Rearrange("b h w c -> b (h w) c"),
            nn.Linear(C, C),  # 1x1, todo 3x3 ?
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

        self.f_5 = nn.Sequential(
            nn.Linear(C, C),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            Rearrange("b (h w) c -> b c h w", h=H, w=W),
        )

        self.f_6 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([C, H, W]),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([C, H, W]),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

        self.final_1x1 = nn.Conv2d(C, in_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, f_e, f_i):
        """
        f_e: feature map of event \in R^{H W C}
        f_i: feature map of image \in R^{H W C}
        """
        # b c h w -> b (h w) c
        f_e_1 = self.f_1(f_e)
        # b c h w -> b h w c
        f_i_2 = self.f_2(f_i)
        # b h w c -> b c h w -> b c (h w)
        f_i_3 = self.f_3(f_i_2)  # k
        # b h w c -> b (h w) c
        f_e_4 = self.f_4(f_i_2)  # q
        channel_transformer_map = torch.matmul(f_e_4, f_i_3)  # (h w) (h w)
        channel_transformer_map = F.softmax(channel_transformer_map, dim=-1)
        f_e_5_in = torch.matmul(channel_transformer_map, f_e_1)
        f_e_5 = self.f_5(f_e_5_in)
        f_i_6 = self.f_6(f_i_2)
        f_e_7 = f_e_5 * f_i_6
        f_e_8 = self.final_1x1(f_e_7)
        return f_e_8, f_i_2


class EventImageFusionTransformer(nn.Module):
    def __init__(self, in_dim, transformer_dim, H, W):
        super(EventImageFusionTransformer, self).__init__()
        self.eift_block_1 = EIRTBlock(in_dim, transformer_dim, H, W)
        self.eift_block_2 = EIRTBlock(in_dim, transformer_dim, H, W)

    def forward(self, f_0_e, f_0_i):
        """
        f_i: feature map of image \in R^{H W C}
        f_e: feature map of event \in R^{H W C}
        """
        f_1_e_, f_1_i_ = self.eift_block_1(f_0_i, f_0_e)
        f_1_e, f_1_i = self.eift_block_2(f_1_e_, f_1_i_)
        return f_1_e, f_1_i


class TwoResBlock(nn.Module):
    def __init__(self, C) -> None:
        super().__init__()
        self.res_1 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_2 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_1 = self.res_1(x) + x
        x_2 = self.res_2(x_1) + x_1
        return x_2


class EventGuidedDualBranch(nn.Module):
    def __init__(self, event_dim, feature_dim, H, W):
        super(EventGuidedDualBranch, self).__init__()
        self.mask_generation = nn.Sequential(
            nn.Conv2d(event_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([feature_dim, H, W]),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.aprf_1 = nn.AdaptiveAvgPool2d((H, W))
        self.aprf_2 = Rearrange("b c h w -> b (h w) c", h=H, w=W)
        self.aprf_3 = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )
        self.aprf_4 = Rearrange("b (h w) c -> b c h w", h=H, w=W)
        self.aprf_5 = nn.AdaptiveAvgPool2d((H, W))

        self.bottom_res = TwoResBlock(feature_dim)

    def forward(self, f_e, f_i, e_plus):
        """
        f_e: feature map of event \in R^{C H W}
        f_i: feature map of image \in R^{C H W}
        e_plus: event image \in R^{M H W}
        """
        mask_0 = self.mask_generation(e_plus)
        mask_1 = 1 - mask_0
        # top branch
        f_i_m = f_i * mask_1
        # [1, 256, 3, 3]
        f_e_i_c = torch.cat([f_e, f_i_m], dim=1)
        # [1, 256, 3, 3]
        f_g = self.aprf_1(f_e_i_c)
        # [1, 9, 256]
        f_g = self.aprf_2(f_g)
        f_g = self.aprf_3(f_g)
        f_g = self.aprf_4(f_g)
        f_g = self.aprf_5(f_g)
        # bottom branch
        f_i = f_i * mask_0
        f_i = self.bottom_res(f_i)
        return torch.cat([f_g, f_i], dim=1)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, bias=True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 7, 1, 3, bias=bias)
        self.conv2 = nn.Conv2d(32, 32, 7, 1, 3, bias=bias)
        self.relu = nn.LeakyReLU(0.1, True)
        # Down 1
        self.avgpool1 = nn.AvgPool2d(kernel_size=7, stride=2, padding=3)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2, bias=bias)
        self.conv4 = nn.Conv2d(64, 64, 5, 1, 2, bias=bias)
        # Down 2
        self.avgpool2 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1, bias=bias)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x_1 = x

        x = self.avgpool1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x_2 = x

        x = self.avgpool2(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x_3 = x

        return x_1, x_2, x_3


class UNetDecoder(nn.Module):
    def __init__(self, middle_feature_dim, out_channels, bias) -> None:
        super().__init__()
        # Decoder
        self.upsample_2d = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv16 = nn.Conv2d(middle_feature_dim * 3, 256, 3, 1, 1, bias=bias)
        self.conv17 = nn.Conv2d(256, 64, 3, 1, 1, bias=bias)
        self.conv18 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.conv19 = nn.Conv2d(64, 32, 3, 1, 1, bias=bias)
        self.conv20 = nn.Conv2d(32, 32, 3, 1, 1, bias=bias)
        self.conv21 = nn.Conv2d(32, 32, 3, 1, 1, bias=bias)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 1, bias=bias)
        self.conv23 = nn.Conv2d(32, out_channels, 3, 1, 1, bias=bias)

    def forward(self, x_1, x_2, x_3):
        # x_1: 1, 32, 12, 12 (H, W)
        # x_2: 1, 64, 6, 6 (H // 2, W // 2)
        # x_3: 1, 128 * 3, 3, 3 (H // 4, W // 4)
        x = F.relu(self.conv16(x_3))
        x = self.upsample_2d(x)
        x = F.relu(self.conv17(x))
        x = x + x_2
        x = F.relu(self.conv18(x))
        x = self.upsample_2d(x)
        x = F.relu(self.conv19(x))
        x = x + x_1
        x = F.relu(self.conv20(x))
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv23(x))
        return x


class UNetEventLigntEnhance(nn.Module):
    def __init__(self, in_channels, out_channels, event_dim, feature_dim, H, W, bias=True):
        super(UNetEventLigntEnhance, self).__init__()
        assert H % 4 == 0 and W % 4 == 0, f"H: {H}, W: {W}"
        middle_feature_dim = 128
        ## Encoder
        self.low_light_image_encoder = UNetEncoder(in_channels, bias=bias)
        self.event_encoder = UNetEncoder(event_dim, bias=bias)
        self.eift = EventImageFusionTransformer(in_dim=128, transformer_dim=128, H=H // 4, W=W // 4)
        self.event_guided_dual_branch = EventGuidedDualBranch(middle_feature_dim, feature_dim, H // 4, W // 4)

        self.decoder = UNetDecoder(middle_feature_dim, out_channels, bias=bias)

    def forward(self, batch):
        # Load data from batch
        f_e, f_i = batch[ELB.E], batch[ELB.LL]
        # f_i_1: 1, 32, 12, 12 (H, W)
        # f_i_2: 1, 64, 6, 6 (H // 2, W // 2)
        # f_i_3: 1, 128, 3, 3 (H // 4, W // 4)
        f_i_1, f_i_2, f_i_3 = self.low_light_image_encoder(f_i)
        # f_e_3: 1, 128, 3, 3 (H // 4, W // 4)
        _, _, f_e_3 = self.event_encoder(f_e)
        # f_e_f: 1, 128, 3, 3 (H // 4, W // 4)
        # f_i_f: 1, 128, 3, 3 (H // 4, W // 4)
        f_e_f, f_i_f = self.eift(f_e_3, f_i_3)
        # f_f: [1, 384, 3, 3]
        f_f = self.event_guided_dual_branch(f_e_f, f_i_f, f_e_3)
        out = self.decoder(f_i_1, f_i_2, f_f)
        # Store prediction in batch
        batch[ELB.PRD] = out
        return batch
