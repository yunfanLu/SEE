import random

import torch
import torch.nn as nn
from absl.logging import info
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB
from see.models import swin_transformer_layers
from see.utils.model_size import model_size


def hswish(x):
    out = x * F.relu6(x + 3, inplace=True) / 6
    return out


def hsigmoid(x):
    out = F.relu6(x + 3, inplace=True) / 6
    return out


class HSwish(nn.Module):
    def forward(self, x):
        return hswish(x)


class HSigmoid(nn.Module):
    def forward(self, x):
        return hsigmoid(x)


def get_bayer_pattern_coordinate(h: int, w: int, w_xy_coords):
    h_coords = torch.linspace(0, h - 1, h).to(torch.int32)
    w_coords = torch.linspace(0, w - 1, w).to(torch.int32)
    mesh_h, mesh_w = torch.meshgrid([h_coords, w_coords])
    mesh_h_2 = torch.remainder(mesh_h, 2).float()
    mesh_w_2 = torch.remainder(mesh_w, 2).float()
    grid_list = [mesh_h_2, mesh_w_2]
    if w_xy_coords:
        mesh_h = mesh_h.float() / h
        mesh_w = mesh_w.float() / w
        grid_list = grid_list + [mesh_h, mesh_w]
    grid_map = torch.stack(grid_list, 2)
    grid_map = grid_map.permute(2, 0, 1)
    return grid_map


class SpatialTemporalAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpatialTemporalAttentionBlock, self).__init__()
        self.C1 = in_channels
        self.C2 = out_channels
        kz = kernel_size
        # 1x1
        self.conv1x1 = nn.Conv2d(self.C1, self.C2, kernel_size=1, stride=1, padding=0, bias=False)
        # 3x3: Params: C1*C2*3*3
        self.conv3x3 = nn.Conv2d(self.C1, self.C2, kernel_size=3, stride=1, padding=1, bias=False)
        # 1x1 9x9 1x1: Params: C1*C1/9 + C1*C1 + C1/9*C2
        self.conv1x1_1 = nn.Conv2d(self.C1, self.C1 // kz, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_kxk = nn.Conv2d(self.C1 // kz, self.C1 // kz, kernel_size=kz, stride=1, padding=kz // 2, bias=False)
        self.conv1x1_2 = nn.Conv2d(self.C1 // kz, self.C2, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        #
        x0 = F.relu(self.conv1x1(x))
        # 3x3
        x1 = F.relu(self.conv3x3(x))
        # 1x1 9x9 1x1
        x2 = F.relu(self.conv1x1_1(x))
        x2 = F.relu(self.conv_kxk(x2))
        x2 = F.relu(self.conv1x1_2(x2))
        # result
        y = x0 + x1 + x2
        return y


class SwinTransformerEncoderBlock(nn.Module):
    def __init__(self, dim, depth, heads, windows_size) -> None:
        super().__init__()
        # model = EncoderLayer(dim=96, depth=4, num_heads=8, num_frames=1, window_size=(4, 4))
        windows_size = (windows_size, windows_size)
        self.model = swin_transformer_layers.EncoderLayer(
            dim=dim, depth=depth, num_heads=heads, num_frames=1, window_size=windows_size
        )
        info(f"SwinTransformerEncoderBlock model size: {model_size(self.model)}")

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, 1, C, H, W)
        x = self.model(x)
        x = x.view(B, C, H, W)
        return x


class SwinTransformerDecoderBlock(nn.Module):
    def __init__(self, dim, depth, heads, windows_size) -> None:
        super().__init__()
        # model = EncoderLayer(dim=96, depth=4, num_heads=8, num_frames=1, window_size=(4, 4))
        windows_size = (windows_size, windows_size)
        self.model = swin_transformer_layers.DecoderLayer(
            dim=dim, depth=depth, num_heads=heads, num_frames=1, window_size=windows_size
        )
        info(f"SwinTransformerDecoderBlock:")
        info(f"  dim         : {dim}")
        info(f"  depth       : {depth}")
        info(f"  heads       : {heads}")
        info(f"  windows_size: {windows_size}")
        info(f"  model size  : {model_size(self.model)}")

    def forward(self, x, y):
        # embdding y featue to x by attentions.
        B, C, H, W = x.shape
        x = x.view(B, 1, C, H, W)
        y = y.view(B, 1, C, H, W)
        z = self.model(x, y)
        z = z.view(B, C, H, W)
        return z


class _SparseEncoder(nn.Module):
    def __init__(self, C1, C2, loop, sparse_encoder_config):
        super(_SparseEncoder, self).__init__()
        assert sparse_encoder_config.type in [
            "conv",
            "spatial_temporal_attention",
            "swin_transformer_encoder_block",
            "swin_transformer_decoder_block",
        ]
        if sparse_encoder_config.type in ["swin_transformer_encoder_block", "swin_transformer_decoder_block"]:
            assert C1 == C2, f"C1 {C1} and C2 {C2} should be equal for swin_transformer_encoder_block."

        self.C1 = C1
        self.C2 = C2
        self.loop = loop
        self.encoder_type = sparse_encoder_config.type

        if sparse_encoder_config.type == "conv":
            self.W1 = nn.Conv2d(C1, C2, 3, 1, 1, bias=False)
            self.S1 = nn.Conv2d(C2, C1, 3, 1, 1, groups=1, bias=False)
            self.S2 = nn.Conv2d(C1, C2, 3, 1, 1, groups=1, bias=False)
        elif sparse_encoder_config.type == "spatial_temporal_attention":
            self.W1 = SpatialTemporalAttentionBlock(C1, C2, kernel_size=sparse_encoder_config.kernel_size)
            self.S1 = SpatialTemporalAttentionBlock(C2, C1, kernel_size=sparse_encoder_config.kernel_size)
            self.S2 = SpatialTemporalAttentionBlock(C1, C2, kernel_size=sparse_encoder_config.kernel_size)
        elif sparse_encoder_config.type == "swin_transformer_encoder_block":
            cfg = sparse_encoder_config
            self.W1 = SwinTransformerEncoderBlock(
                dim=C1, depth=cfg.depth, heads=cfg.heads, windows_size=cfg.windows_size
            )
            self.S1 = SwinTransformerEncoderBlock(
                dim=C2, depth=cfg.depth, heads=cfg.heads, windows_size=cfg.windows_size
            )
            self.S2 = SwinTransformerEncoderBlock(
                dim=C1, depth=cfg.depth, heads=cfg.heads, windows_size=cfg.windows_size
            )
        elif sparse_encoder_config.type == "swin_transformer_decoder_block":
            cfg = sparse_encoder_config
            self.W1 = SwinTransformerDecoderBlock(
                dim=C1, depth=cfg.depth, heads=cfg.heads, windows_size=cfg.windows_size
            )
            self.S1 = SwinTransformerDecoderBlock(
                dim=C2, depth=cfg.depth, heads=cfg.heads, windows_size=cfg.windows_size
            )
            self.S2 = SwinTransformerDecoderBlock(
                dim=C1, depth=cfg.depth, heads=cfg.heads, windows_size=cfg.windows_size
            )

        self.hsw = HSwish()

    def _swin_decoding(self, frames, events):
        x1 = torch.mul(frames, events)
        z = self.W1(x1, events)
        tmp = z
        for i in range(self.loop):
            ttmp = self.hsw(tmp)
            x = self.S1(ttmp, events)
            x = torch.mul(x, events)
            x = torch.mul(x, events)
            x = self.S2(x, events)
            x = ttmp - x
            tmp = torch.add(x, z)
        c = self.hsw(tmp)
        return c

    def forward(self, frames, events):
        if self.encoder_type == "swin_transformer_decoder_block":
            return self._swin_decoding(frames, events)

        x1 = frames
        x1 = torch.mul(x1, events)
        z = self.W1(x1)
        tmp = z
        for i in range(self.loop):
            ttmp = self.hsw(tmp)
            x = self.S1(ttmp)
            x = torch.mul(x, events)
            x = torch.mul(x, events)
            x = self.S2(x)
            x = ttmp - x
            tmp = torch.add(x, z)
        c = self.hsw(tmp)
        return c


class CosineEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(CosineEmbedding, self).__init__()
        assert input_dim in [1, 2, 4]
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.pe_sigma = torch.tensor(10000, dtype=torch.float32)

    def convert_posenc(self, coords):
        N, C, H, W = coords.shape
        if C != self.input_dim:
            raise ValueError("Input channel must be %d, but got %d" % (self.input_dim, C))
        coords = coords.view(N * C, H, W)
        w = torch.exp(torch.linspace(0, torch.log(self.pe_sigma), self.embed_dim // (2 * C), device=coords.device))
        coords_embed = torch.einsum("nhw,d->nhwd", [coords, w])
        coords_embed = torch.cat([torch.sin(coords_embed), torch.cos(coords_embed)], dim=-1)  # NC*H*W*dim
        coords_embed = coords_embed.view(N, C, H, W, self.embed_dim)
        coords_embed = coords_embed.permute(0, 1, 4, 2, 3)
        coords_embed = coords_embed.reshape(N, C * self.embed_dim, H, W)
        return coords_embed

    def forward(self, coords):
        coords_embed = self.convert_posenc(coords)
        return coords_embed


class PositionEmbedding(nn.Module):
    def __init__(self, position_embedding_type, in_c, em_c):
        super(PositionEmbedding, self).__init__()
        assert position_embedding_type in ["learning:1x1", "learning:1x1+1x1", "sin-cos"]
        if position_embedding_type == "learning:1x1":
            self.pe = nn.Conv2d(
                in_channels=in_c,
                out_channels=em_c,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif position_embedding_type == "learning:1x1+1x1":
            self.bayer_emb_1 = nn.Conv2d(
                in_channels=in_c,
                out_channels=em_c,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.bayer_emb_2 = nn.Conv2d(
                in_channels=in_c + em_c,
                out_channels=em_c,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif position_embedding_type == "sin-cos":
            self.pe = CosineEmbedding(in_c, em_c)
        self.position_embedding_type = position_embedding_type

    def forward(self, x):
        if self.position_embedding_type == "learning:1x1":
            y = self.pe(x)
        elif self.position_embedding_type == "learning:1x1+1x1":
            y = self.bayer_emb_1(x)
            y = torch.cat([x, y], 1)
            y = self.bayer_emb_2(y)
        else:
            y = self.pe(x)
        return y


class ExposureDecodingR1(nn.Module):
    """
    This vision is used in options/SeeDynamicEventDataset/SEENet/Ablation-Cascade/SEENet_SDE-ae3-96-4-8-4-loss-2-epoch-20-using-cascade.yaml
    and this config is efficient.
    """

    def __init__(self, inr_c, mlp_layers, exposure_using_cascade_embedding, embedding_method):
        super(ExposureDecodingR1, self).__init__()

        assert embedding_method in ["ADD", "MULTI"]

        self.embedding_method = embedding_method
        self.using_cascade = exposure_using_cascade_embedding

        self.decoder = nn.Sequential()
        for i in range(mlp_layers - 1):
            self.decoder.append(
                nn.Conv2d(
                    in_channels=inr_c,
                    out_channels=inr_c,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            )
            self.decoder.append(nn.ReLU(inplace=True))
        self.decoder.append(
            nn.Conv2d(
                in_channels=inr_c,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )

    def forward(self, x, light_emb, exposure_prompt):
        y = x
        if self.using_cascade:
            for layer in self.decoder:
                if self.embedding_method == "ADD":
                    y = y + light_emb
                else:
                    y = y * light_emb
                y = layer(y)
        else:
            y = y + light_emb
            y = self.decoder(y)
        return y


class ExposureDecoderLayer(nn.Module):
    def __init__(self, inr_c, act):
        super(ExposureDecoderLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=inr_c,
            out_channels=inr_c,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "sin":
            self.act = torch.sin

    def forward(self, x):
        return self.act(self.conv(x))


class ExposureDecoding(nn.Module):
    def __init__(self, inr_c, mlp_layers, exposure_using_cascade_embedding, is_inr_normal_feature, embedding_method):
        """
        mlp_layers: (Define:3) the number of layers in the mlp.
        exposure_using_cascade_embedding: (Define:False) if using cascade the time embedding will be applied to each layer.
        embedding_method: ADD or Multi, defineation is added.
        """
        super(ExposureDecoding, self).__init__()
        assert embedding_method in ["ADD", "MULTI"]
        self.using_cascade = exposure_using_cascade_embedding
        self.is_inr_normal_feature = is_inr_normal_feature
        self.embedding_method = embedding_method

        if self.using_cascade:
            self.decoder = nn.ModuleList()
        else:
            self.decoder = nn.Sequential()
        for i in range(mlp_layers - 1):
            self.decoder.append(ExposureDecoderLayer(inr_c=inr_c, act="relu"))

        if is_inr_normal_feature:
            self.bn = nn.BatchNorm2d(inr_c)

        self.inr_to_rgb = nn.Conv2d(
            in_channels=inr_c,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x, light_emb, exposure_prompt):
        y = x
        if self.using_cascade:
            for layer in self.decoder:
                if self.embedding_method == "ADD":
                    y = y + light_emb
                else:
                    y = y * light_emb
                y = layer(y)
        else:
            y = y + light_emb
            y = self.decoder(y)
        # if using the y is learning normal feature.
        if self.is_inr_normal_feature:
            y = self.bn(y)
            y = y + exposure_prompt
        # return the rgb image
        y = self.inr_to_rgb(y)
        return y


class SEENet(nn.Module):
    def __init__(self, frames, moments, C1, C2, loop, exposure_sample_num, SEE_config):
        super(SEENet, self).__init__()
        assert SEE_config.position_embedding in ["bayer_pattern", "none"]

        self.frames = frames
        self.in_channel = 3 * frames
        self.moments = moments
        self.C1 = C1
        self.C2 = C2
        self.loop = loop
        self.exposure_sample_num = exposure_sample_num
        self.SEE_config = SEE_config

        # Ablation Study 1: Bayer Pattern Embedding
        #   1.1 Bayer Pattern Embedding only with Bayer Pattern
        #   1.2 Bayer Pattern Embedding with coordinates
        # bayer pattern embedding
        if self.SEE_config.position_embedding == "bayer_pattern":
            pos_channels = 4 if self.SEE_config.w_xy_coords else 2
            self.position_embedding = PositionEmbedding(self.SEE_config.position_embedding_type, pos_channels, C1)

        # Ablation Study 2: Image and Event Head Ablation
        #    2.1 head with original code
        #    2.2 head with 9x9 convolution
        # Image head
        self.image_head, self.event_head = self._event_image_heads()
        # Spare Decoding
        self.scn_1 = _SparseEncoder(C1, C2, loop, sparse_encoder_config=self.SEE_config.sparse_encoder_config)
        # Light Embedding
        self.light_emb_conv = PositionEmbedding(self.SEE_config.exposure_embedding_type, 1, C2)

        if self.SEE_config.ExposureDecoder == "Release-1":
            self.inr_decoder = ExposureDecodingR1(
                C2,
                mlp_layers=self.SEE_config.exposure_mlp_layers,
                exposure_using_cascade_embedding=self.SEE_config.exposure_using_cascade_embedding,
                embedding_method=self.SEE_config.exposure_embedding_method,
            )
        else:
            self.inr_decoder = ExposureDecoding(
                C2,
                mlp_layers=self.SEE_config.exposure_mlp_layers,
                exposure_using_cascade_embedding=self.SEE_config.exposure_using_cascade_embedding,
                is_inr_normal_feature=self.SEE_config.is_inr_normal_feature,
                embedding_method=self.SEE_config.exposure_embedding_method,
            )

    def _decoding(self, inr, exposure_prompt):
        B, C, H, W = inr.shape
        light_emb = torch.zeros(B, 1, H, W).to(inr.device) + exposure_prompt
        light_emb = F.relu(self.light_emb_conv(light_emb))
        out = self.inr_decoder(inr, light_emb, exposure_prompt)
        return out

    def forward(self, batch):
        events = batch[ELB.E]
        images = batch[ELB.LL]
        target = batch[ELB.NL]
        B, CN, H, W = target.shape
        if self.SEE_config.position_embedding == "bayer_pattern":
            xy_pos = get_bayer_pattern_coordinate(h=H, w=W, w_xy_coords=self.SEE_config.w_xy_coords)
            xy_pos = xy_pos.unsqueeze(0).repeat(B, 1, 1, 1).to(images.device)
            xy_pos = self.position_embedding(xy_pos)
            images = torch.cat([images, xy_pos], dim=1)
            events = torch.cat([events, xy_pos], dim=1)

        x1 = self.image_head(images)
        ev = self.event_head(events)
        light_inr = self.scn_1(x1, ev)
        # deocding
        # 1. decoding to target normal
        target_prompt = torch.mean(target, dim=(1, 2, 3), keepdim=True)
        out = self._decoding(light_inr, target_prompt)
        batch[ELB.PRD] = out

        # 2. self-supervised
        self_prompt = torch.mean(images, dim=(1, 2, 3), keepdim=True)
        self_reconstruction = self._decoding(light_inr, self_prompt)
        batch[ELB.SSR] = self_reconstruction
        # 3. more exposure prompt
        if self.exposure_sample_num > 0:
            batch[ELB.NLR] = []
            batch[ELB.NLR_EP] = []
            for i in range(self.exposure_sample_num):
                if self.training:
                    exposure_mean = random.uniform(0.4, 0.7)
                else:
                    exposure_mean = 0.2 + i / self.exposure_sample_num * 0.5
                exposure_prompt = torch.ones(size=(B, 1, 1, 1)) * exposure_mean
                exposure_prompt = exposure_prompt.to(images.device)
                exposure_normal_reconstructed = self._decoding(light_inr, exposure_prompt)
                batch[ELB.NLR_EP].append(exposure_prompt)
                batch[ELB.NLR].append(exposure_normal_reconstructed)
            batch[ELB.NLR] = torch.stack(batch[ELB.NLR], dim=1)
            batch[ELB.NLR_EP] = torch.stack(batch[ELB.NLR_EP], dim=1)

        if not self.training:
            standard_mean = 0.4
            exposure_prompt = torch.ones(size=(B, 1, 1, 1)) * standard_mean
            exposure_prompt = exposure_prompt.to(images.device)
            standard_normal_reconstructed = self._decoding(light_inr, exposure_prompt)
            batch[ELB.SLR] = standard_normal_reconstructed
        return batch

    def _event_image_heads(self):
        position_embedding = self.C1 if self.SEE_config.position_embedding == "bayer_pattern" else 0
        image_in_channel = self.in_channel + position_embedding
        event_in_channel = self.moments + position_embedding
        if self.SEE_config.head == "original:image1x1-event1x1sigmod1x1":
            image_head = nn.Conv2d(
                in_channels=image_in_channel,
                out_channels=self.C1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            event_head = nn.Sequential(
                nn.Conv2d(
                    in_channels=event_in_channel,
                    out_channels=self.C1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.Sigmoid(),
                nn.Conv2d(
                    in_channels=self.C1,
                    out_channels=self.C1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.Sigmoid(),
            )
            return image_head, event_head
        elif self.SEE_config.head == "v1:w-9x9-depth-cpnv":
            image_head = nn.Sequential(
                nn.Conv2d(
                    in_channels=image_in_channel,
                    out_channels=self.C1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=self.C1,
                    out_channels=self.C1,
                    kernel_size=9,
                    stride=1,
                    padding=4,
                    bias=False,
                    groups=self.C1,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=self.C1,
                    out_channels=self.C1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )
            event_head = nn.Sequential(
                nn.Conv2d(
                    in_channels=event_in_channel,
                    out_channels=self.C1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=self.C1,
                    out_channels=self.C1,
                    kernel_size=9,
                    stride=1,
                    padding=4,
                    bias=False,
                    groups=self.C1,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=self.C1,
                    out_channels=self.C1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
            )
            return image_head, event_head
