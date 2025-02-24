import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELBC
from see.datasets.basic_batch import get_ev_low_light_batch
from see.utils import print_batch

"""
@inproceedings{liang2024towards,
  title={Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel
    Approach},
  author={Liang, Guoqiang and Chen, Kanghao and Li, Hangyu and Lu, Yunfan and Wang, Lin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23--33},
  year={2024}
}
"""


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class IG_MSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (
                q_inp,
                k_inp,
                v_inp,
            ),
        )
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = k @ q.transpose(-2, -1)  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(
            y.transpose(-1, -3),
            kernel_size=(1, self.k_size),
            padding=(0, (self.k_size - 1) // 2),
        )
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x


class ECAResidualBlock(nn.Module):
    def __init__(self, nf):
        super(ECAResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.eca = ECALayer(nf)
        self.norm = nn.InstanceNorm2d(nf // 2, affine=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.eca(out)
        out = out + residual
        out = self.relu(out)
        return out


class CA_layer(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_ch // 2, in_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        res = self.conv1(x)
        cross = self.conv2(x)
        res = res * cross
        x = x + res
        return x


class SNR_enhance(nn.Module):
    def __init__(self, channel, snr_threshold, depth):
        super().__init__()
        self.channel = channel
        self.depth = depth
        self.img_extractor = nn.ModuleList()
        self.ev_extractor = nn.ModuleList()
        for i in range(self.depth):
            self.img_extractor.append(ECAResidualBlock(self.channel))
            self.ev_extractor.append(ECAResidualBlock(self.channel))
        self.fea_align = nn.Sequential(
            CA_layer(self.channel * 3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(self.channel * 3, self.channel * 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.threshold = snr_threshold

    def forward(self, cnn_fea, snr_map, att_fea, events):
        snr_map[snr_map <= self.threshold] = 0.3
        snr_map[snr_map > self.threshold] = 0.7
        snr_reverse_map = 1 - snr_map
        snr_map_enlarge = snr_map.repeat(1, self.channel, 1, 1)
        snr_reverse_map_enlarge = snr_reverse_map.repeat(1, self.channel, 1, 1)
        for i in range(self.depth):
            cnn_fea = self.img_extractor[i](cnn_fea)
            events = self.ev_extractor[i](events)
        out_img = torch.mul(cnn_fea, snr_map_enlarge)
        out_ev = torch.mul(events, snr_reverse_map_enlarge)
        out = self.fea_align(torch.concat((out_img, out_ev, att_fea), dim=1))
        visual_pack = {
            "original_snr": snr_map,
            "snr_map": snr_map_enlarge,
            "snr_reverse_map": snr_reverse_map_enlarge,
            "snr_img": out_img,
            "snr_ev": out_ev,
            "input_ev": events,
            "cnn_fea": cnn_fea,
        }
        if self.depth == 0:
            return att_fea, visual_pack
        return out, visual_pack


class Unet_ReFormer(nn.Module):
    def __init__(
        self,
        in_dim=3,
        out_dim=3,
        dim=31,
        level=2,
        num_blocks=[2, 4, 4],
        snr_depth_list=[2, 4, 6],
        snr_threshold_list=[0.5, 0.5, 0.5],
    ):
        super(Unet_ReFormer, self).__init__()
        self.dim = dim
        self.level = level
        self.snr_threshold_list = snr_threshold_list
        self.snr_depth_list = snr_depth_list
        # Input projection
        self.img_head = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        IGAB(
                            dim=dim_level,
                            num_blocks=num_blocks[i],
                            dim_head=dim,
                            heads=dim_level // dim,
                        ),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                        SNR_enhance(
                            dim_level,
                            snr_threshold_list[i],
                            snr_depth_list[i],
                        ),
                    ]
                )
            )
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=num_blocks[-1],
        )
        self.bottleneck_SNR = SNR_enhance(
            dim_level,
            snr_threshold_list[-1],
            snr_depth_list[-1],
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            dim_level,
                            dim_level // 2,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                        ),
                        nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                        IGAB(
                            dim=dim_level // 2,
                            num_blocks=num_blocks[level - 1 - i],
                            dim_head=dim,
                            heads=(dim_level // 2) // dim,
                        ),
                        SNR_enhance(
                            dim_level // 2,
                            snr_threshold_list[level - 1 - i],
                            snr_depth_list[level - 1 - i],
                        ),
                    ]
                )
            )
            dim_level //= 2
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, event_img, enhance_low_img, fea_img, SNR, events):
        fea_event_img = event_img
        fea_event_img_encoder = []
        SNRDownsample_list = []
        fea_img_list = []
        event_free_list = []
        for (
            IGAB,
            FeaDownSample,
            ImgDownSample,
            EvDownSample,
            SNR_enhance,
        ) in self.encoder_layers:
            fea_event_img = IGAB(fea_event_img)  # bchw
            fea_event_img_encoder.append(fea_event_img)
            event_free_list.append(events)
            SNRDownsample_list.append(SNR)
            fea_img_list.append(fea_img)
            SNR = self.avg_pool(SNR)
            fea_event_img = FeaDownSample(fea_event_img)
            fea_img = ImgDownSample(fea_img)
            events = EvDownSample(events)
        fea_event_img, _ = self.bottleneck_SNR(fea_img, SNR, fea_event_img, events)
        fea_event_img = self.bottleneck(fea_event_img)
        for i, (FeaUpSample, Fusion, REIGAB, RESNR_enhance) in enumerate(self.decoder_layers):
            fea_event_img = FeaUpSample(fea_event_img)
            fea_event_img = Fusion(torch.cat([fea_event_img, fea_event_img_encoder[self.level - 1 - i]], dim=1))
            SNR = SNRDownsample_list[self.level - 1 - i]
            fea_img = fea_img_list[self.level - 1 - i]
            events = event_free_list[self.level - 1 - i]
            fea_event_img, visual_pack = RESNR_enhance(fea_img, SNR, fea_event_img, events)
            fea_event_img = REIGAB(fea_event_img)
        # Mapping
        out = self.mapping(fea_event_img) + enhance_low_img
        return out, visual_pack


class IllumiinationNet(nn.Module):
    def __init__(self):
        super().__init__()
        illumiantion_level = 1
        base_chs = 48
        self.ill_extractor = nn.Sequential(
            nn.Conv2d(
                illumiantion_level + 3,
                illumiantion_level * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(
                illumiantion_level * 2,
                base_chs,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.illumiantion_set = [0]
        self.reduce = nn.Sequential(
            nn.Conv2d(base_chs, 1, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, low_light_img, inital_ill):
        pred_illu_feature = self.ill_extractor(torch.concat((inital_ill, low_light_img), dim=1))
        pred_illumaintion = self.reduce(pred_illu_feature)
        return pred_illumaintion, pred_illu_feature


class ImageEnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        voxel_grid_channel = 32
        base_chs = 48
        snr_threshold_list = [0.6, 0.5, 0.4]
        self.snr_factor = 3.0
        self.ev_img_align = nn.Conv2d(
            base_chs * 2,
            base_chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.ev_extractor = nn.Sequential(
            nn.Conv2d(
                voxel_grid_channel,
                base_chs,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.img_extractor = nn.Sequential(
            nn.Conv2d(
                3,
                base_chs,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.Unet_ReFormer = Unet_ReFormer(
            dim=base_chs,
            snr_threshold_list=snr_threshold_list,
        )

    def _snr_generate(self, low_img, low_img_blur):
        dark = low_img
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = low_img_blur
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)
        mask = torch.div(light, noise + 0.0001).contiguous()
        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * self.snr_factor / (mask_max + 0.0001)
        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        return mask

    def forward(self, low_light_img, pred_illumaintion, low_light_img_blur, events):
        enhance_low_img_mid = low_light_img * pred_illumaintion + low_light_img
        enhance_low_img_blur = low_light_img_blur * pred_illumaintion + low_light_img_blur
        snr_lightup = self._snr_generate(enhance_low_img_mid, enhance_low_img_blur)
        snr_enhance = snr_lightup.detach()
        events = self.ev_extractor(events)
        enhance_low_img = self.img_extractor(enhance_low_img_mid)
        img_event = self.ev_img_align(torch.concat((events, enhance_low_img), dim=1))
        pred_normal_img, visual_pack = self.Unet_ReFormer(
            img_event, enhance_low_img_mid, enhance_low_img, snr_enhance, events
        )
        return pred_normal_img, enhance_low_img_mid, visual_pack


class EventGuidedLowLightImageEnhacement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.IllumiinationNet = IllumiinationNet()
        self.ImageEnhanceNet = ImageEnhanceNet()

    def forward(self, batch):
        # load data from batch
        low_light_img = batch[ELBC.LL]
        inital_ill = batch[ELBC.ILL]
        low_light_img_blur = batch[ELBC.LLB]
        events = batch[ELBC.E]
        # inference
        pred_illumaintion, _ = self.IllumiinationNet(low_light_img, inital_ill)
        output, enhance_low_img_mid, visual_pack = self.ImageEnhanceNet(
            low_light_img, pred_illumaintion, low_light_img_blur, events
        )
        # store data into batch
        batch[ELBC.PRD] = output
        return batch

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)  # conv init
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)  # linear init
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight.data)  # deconv init
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 1, 0.02)
                torch.nn.init.zeros_(m.bias)
