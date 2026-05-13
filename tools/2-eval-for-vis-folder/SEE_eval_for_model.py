import os
from os import listdir
from os.path import join
from os.path import isfile
from absl import app
from absl import flags
from absl import logging
from absl.logging import info
import cv2
import numpy as np
import torch

from see.losses.psnr import PSNR, NormalizedPSNR
from see.losses.ssim import SSIM


"""
Root
    Vis
        group_name-[normal-normal OR low-normal OR high-normal]-input_videoname-normal_videoname
            1689563525934644_ev_DCE_SDE-v1.png # Event Visualization
            1689563525934644_g0_DCE_SDE-v1.png # Gamma Visualization (Using Linear to Adjust the Brightness)
            1689563525934644_l0_DCE_SDE-v1.png # Low Quality Image (Input)
            1689563525934644_n0_DCE_SDE-v1.png # Normal Quailty Image (Ground Truth)
            1689563525934644_p0_DCE_SDE-v1.png # Prediction (Output)
        o_2
            ...
        ...
    Log.INFO
"""

FLAGS = flags.FLAGS

flags.DEFINE_string("root", None, "Root directory")

class Eval:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.psnr = 0
        self.ssim = 0
        self.npsnr = 0
        self.l1 = 0

    def add(self, psnr, ssim, npsnr, l1):
        self.count += 1
        self.psnr += psnr
        self.ssim += ssim
        self.npsnr += npsnr
        self.l1 += l1

    def get(self):
        if self.count == 0:
            return 0, 0, 0, 0
        return self.psnr / self.count, self.ssim / self.count, self.npsnr / self.count, self.l1 / self.count


def _load_image(image_path):
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32).transpose(2, 0, 1) / 255.0
    frame = torch.from_numpy(frame)
    return frame


def main(_):
    # import pudb

    # pudb.set_trace()

    logging.set_verbosity(logging.DEBUG)
    info(f"log_dir: {FLAGS.log_dir}")
    logging.get_absl_handler().use_absl_log_file()
    #
    vis_folder = join(FLAGS.root, "vis")

    f_psnr = PSNR().cuda()
    f_ssim = SSIM().cuda()
    f_npsnr = NormalizedPSNR("linear").cuda()
    f_l1 = torch.nn.L1Loss().cuda()

    low_nml_eval = Eval("low_nml")
    hgh_nml_eval = Eval("hgh_nml")
    nml_nml_eval = Eval("nml_nml")

    for video_folder_name in sorted(listdir(vis_folder)):
        video_folder = join(vis_folder, video_folder_name)
        in_out_gt = [[], [], []]
        for img_name in sorted(listdir(video_folder)):
            # 1718711180188834_0_0_1718711180188834_1718711180208834_l0_DCE_SDE-v1
            # 1718711220588867_0_0_1718711220588867_1718711220608867_n0_DCE_SDE-v1
            # 1718711181388835_0_0_1718711181388835_1718711181408835_p0_DCE_SDE-v1
            ts1, ts2, ts3, ts4, ts5, tp = img_name.split("_")[:6]
            if tp[:2] == "l0":
                gt_name = img_name.replace("l0", "n0")
                pd_name = img_name.replace("l0", "p0")

                ll_path = join(video_folder, img_name)
                lp_path = join(video_folder, pd_name)
                lg_path = join(video_folder, gt_name)

                if isfile(ll_path) and isfile(lp_path) and isfile(lg_path):

                    info(f"Low Quality Image: {img_name}")
                    info(f"Prediction       : {pd_name}")
                    info(f"Ground Truth     : {gt_name}")

                    in_out_gt[0].append(ll_path)
                    in_out_gt[1].append(lp_path)
                    in_out_gt[2].append(lg_path)

        for _, o, g in zip(in_out_gt[0], in_out_gt[1], in_out_gt[2]):
            oi = _load_image(o)
            gi = _load_image(g)
            oi = oi.cuda().unsqueeze(0)
            gi = gi.cuda().unsqueeze(0)
            psnr = f_psnr(oi, gi).item()
            ssim = f_ssim(oi, gi).item()
            npsnr = f_npsnr(oi, gi).item()
            l1 = f_l1(oi, gi).item()

            if "high-normal" in video_folder_name:
                hgh_nml_eval.add(psnr, ssim, npsnr, l1)
                info(f"high-normal  : {hgh_nml_eval.get()}")
            elif "normal-normal" in video_folder_name:
                nml_nml_eval.add(psnr, ssim, npsnr, l1)
                info(f"normal-normal: {nml_nml_eval.get()}")
            elif "low-normal" in video_folder_name:
                low_nml_eval.add(psnr, ssim, npsnr, l1)
                info(f"low-normal   : {low_nml_eval.get()}")

    info(f"ROOT: {vis_folder}")
    info(f"Low-Normal   : {low_nml_eval.get()}")
    info(f"High-Normal  : {hgh_nml_eval.get()}")
    info(f"Normal-Normal: {nml_nml_eval.get()}")



if __name__ == "__main__":
    app.run(main)
