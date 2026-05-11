import os
from os import makedirs
from os.path import join

import cv2
import numpy as np
import torch
from absl.logging import debug, flags, info

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB
from see.visualize.vis_tools import event_vis

FLAGS = flags.FLAGS


class EventLowLightBatchVisualizer:
    def __init__(self, visdir, tag, vis_intermediate=False):
        self.folder = join(FLAGS.log_dir, visdir)
        makedirs(self.folder, exist_ok=True)
        self.tag = tag
        self.vis_intermediate = vis_intermediate

    def __call__(self, batch):
        E = batch[ELB.E]
        LL = batch[ELB.LL]
        HL = batch[ELB.HL]
        NL = batch[ELB.NL]
        PRD = batch[ELB.PRD]
        SLR = batch[ELB.SLR]
        video_names = batch[ELB.VIDEO_NAME]
        frame_names = batch[ELB.FRAME_NAME]
        tag = self.tag

        B = E.shape[0]
        for b in range(B):
            testdata = join(self.folder, video_names[b])
            frame_name = frame_names[b]
            makedirs(testdata, exist_ok=True)
            # E
            event_voxel = E[b].cpu().numpy()
            event_image = event_vis(event_voxel)
            cv2.imwrite(join(testdata, f"{frame_name}_ev_{tag}.png"), event_image)
            # LL
            if isinstance(LL[b], torch.Tensor):
                NC, H, W = LL[b].shape
                ll_frame = LL[b].cpu().numpy().reshape(NC // 3, 3, H, W)
                for i in range(NC // 3):
                    ll = (ll_frame[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                    ll = cv2.cvtColor(ll, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(join(testdata, f"{frame_name}_l{i}_{tag}.png"), ll)
                    # linear scale mean value to 100
                    ll_lm127 = ll * (127 / ll.mean())
                    cv2.imwrite(join(testdata, f"{frame_name}_g{i}_{tag}.png"), ll_lm127)
            # NL
            if isinstance(NL[b], torch.Tensor):
                NC, H, W = NL[b].shape
                nl_frame = NL[b].cpu().numpy().reshape(NC // 3, 3, H, W)
                for i in range(NC // 3):
                    nl = (nl_frame[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                    nl = cv2.cvtColor(nl, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(join(testdata, f"{frame_name}_n{i}_{tag}.png"), nl)
            # PRED
            if isinstance(PRD[b], torch.Tensor):
                NC, H, W = PRD[b].shape
                prd_frame = PRD[b].cpu().numpy().reshape(NC // 3, 3, H, W)
                for i in range(NC // 3):
                    prd = (prd_frame[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                    prd = cv2.cvtColor(prd, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(join(testdata, f"{frame_name}_p{i}_{tag}.png"), prd)
            # batch[ELB.SLR]
            if isinstance(SLR[b], torch.Tensor):
                NC, H, W = SLR[b].shape
                slr_frame = SLR[b].cpu().numpy().reshape(NC // 3, 3, H, W)
                for i in range(NC // 3):
                    slr = (slr_frame[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                    slr = cv2.cvtColor(slr, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(join(testdata, f"{frame_name}_s{i}_{tag}.png"), slr)

            # batch[ELB.NLR_EP]
            # batch[ELB.NLR]
            if isinstance(batch[ELB.NLR][b], torch.Tensor):
                N, C, H, W = batch[ELB.NLR][b].shape
                info(f"batch[ELB.NLR][b].shape: {batch[ELB.NLR][b].shape}")
                nlr_frame = batch[ELB.NLR][b].cpu().numpy().reshape(N, 3, H, W)
                prompt = batch[ELB.NLR_EP][b].cpu().numpy()
                for i in range(N):
                    info(f"Prompt: {prompt[i]}")
                    pp = prompt[i][0][0][0]
                    nlr = (nlr_frame[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                    nlr = cv2.cvtColor(nlr, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(join(testdata, f"{frame_name}_nlr{i}_p{pp:.2f}_{tag}.png"), nlr)