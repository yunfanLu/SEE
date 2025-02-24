import torch
import torch.nn as nn
from absl.logging import info

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB


class _SCN(nn.Module):
    def __init__(self, hidden_channels, high_dim_channels, loop):
        super(_SCN, self).__init__()
        self.hidden_channels = hidden_channels
        self.high_dim_channels = high_dim_channels
        self.loop = loop

        self.W1 = nn.Conv2d(hidden_channels, high_dim_channels, 3, 1, 1, bias=False)
        self.S1 = nn.Conv2d(high_dim_channels, hidden_channels, 3, 1, 1, groups=1, bias=False)
        self.S2 = nn.Conv2d(hidden_channels, high_dim_channels, 3, 1, 1, groups=1, bias=False)
        self.shlu = nn.ReLU(True)

    def forward(self, blur_image, events):
        x1 = blur_image
        event_input = events
        x1 = torch.mul(x1, event_input)
        z = self.W1(x1)
        tmp = z
        for i in range(self.loop):
            ttmp = self.shlu(tmp)
            x = self.S1(ttmp)
            x = torch.mul(x, event_input)
            x = torch.mul(x, event_input)
            x = self.S2(x)
            x = ttmp - x
            tmp = torch.add(x, z)
        c = self.shlu(tmp)
        return c


class ESLBackBone(nn.Module):
    def __init__(
        self,
        input_frames,
        event_moments,
        hidden_channels,
        high_dim_channels,
        loop,
    ):
        super(ESLBackBone, self).__init__()

        self.input_frames = input_frames
        image_channel = 3
        self.in_channel = image_channel * input_frames
        self.event_moments = event_moments
        self.hidden_channels = hidden_channels
        self.high_dim_channels = high_dim_channels
        self.loop = loop

        self.image_d = nn.Conv2d(
            in_channels=self.in_channel,
            out_channels=self.hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.event_c1 = nn.Conv2d(
            in_channels=event_moments,
            out_channels=self.hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.event_c2 = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)

        self.scn_1 = _SCN(hidden_channels, high_dim_channels, loop)

        self.end_conv = nn.Conv2d(
            in_channels=high_dim_channels,
            out_channels=image_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, batch):
        events = batch[ELB.E]
        image = batch[ELB.LL]

        x1 = self.image_d(image)
        event_out = self.event_c1(events)
        event_out = torch.sigmoid(event_out)
        event_out = self.event_c2(event_out)
        event_out = torch.sigmoid(event_out)
        out = self.scn_1(x1, event_out)
        out = self.end_conv(out)

        batch[ELB.PRD] = out
        return batch
