# SEE: See Everything Every Time

**SEE** is a framework for adaptive brightness adjustment across broad lighting conditions using RGB frames and event-camera data. This repository provides the code for **SEE-Net** and resources for the **SEE-600K** dataset.

SEE-Net uses event guidance to enhance images captured under low light, over-exposure, mixed illumination, and high-contrast scenes while preserving scene structure and natural appearance.

## Key Features

- **Event-guided brightness adjustment**: uses RGB frames together with event data for broad-light-range image enhancement.
- **Continuous exposure control**: supports pixel-level brightness adjustment through an exposure prompt.
- **Compact baseline**: SEE-Net has approximately **1.9M parameters**.
- **Large-scale dataset support**: built around SEE-600K, which contains **610,126 images with corresponding event data** across **202 real-world scenarios**.

## Dataset Download

The datasets supporting this project are publicly available:

- **SEE-600K Dataset**
  - [OneDrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/ylu066_connect_hkust-gz_edu_cn/EkNi59p2uHJFjxyeQraiVhgBSs1GnxK4DyCUP-uZhEspCA?e=ZpwOvY)
  - [Hugging Face](https://huggingface.co/datasets/yunfanlu/SEE-600K)
- **SDE Dataset**: [SDE Dataset GitHub](https://github.com/EthanLiang99/EvLight)

## Pretrained Models

Pretrained checkpoints, evaluation logs, and experiment files for the released baselines are available on Google Drive:

<https://drive.google.com/drive/folders/1SWR9YVIrqFEkGGKrv3wSC5RqmMIEYjV2?usp=sharing>

The folder is organized by method and includes checkpoints for `EIFT_AAAI_SEE`, `eSl_SEE`, `EvLight`, and `SEENet_SEE`, along with related logs and `TEST_RESULTS.md`.

## Documentation

- [Tutorial](docs/TUTORIAL.md): environment setup, dataset preparation, training, evaluation, inference, and FAQ.
- [CodaBench Submission Guide](docs/CODABENCH_SUBMISSION_GUIDE.md): step-by-step instructions for packaging and submitting results to CodaBench.
- [CodaBench Page Description](CODABENCH_README.md): concise standalone text for the SEE Challenge CodaBench page.
- [Test Results](docs/TEST_RESULTS.md): baseline results on the SEE dataset.
- [Mini-GT Test Results](docs/TEST_MINI_RESULTS.md): baseline results on the SEE mini GT dataset.

Chinese versions are also available:

- [README_ZH.md](README_ZH.md)
- [docs/TUTORIAL_ZH.md](docs/TUTORIAL_ZH.md)
- [docs/CODABENCH_SUBMISSION_GUIDE_ZH.md](docs/CODABENCH_SUBMISSION_GUIDE_ZH.md)
- [CODABENCH_README_ZH.md](CODABENCH_README_ZH.md)

## Related Links

- **Workshop Website**: [https://eventbasemultimodalvision.github.io](https://eventbasemultimodalvision.github.io)
- **Competition Page**: [https://www.codabench.org/competitions/16195/](https://www.codabench.org/competitions/16195/#/pages-tab)
- **Competition Forum**: [https://www.codabench.org/forums/15944/](https://www.codabench.org/forums/15944/)
- **Competition Report**: [https://arxiv.org/abs/2502.21120](https://arxiv.org/abs/2502.21120)
- **IMU Registration Tool**: [https://github.com/yunfanLu/IMU-Registration-Tool](https://github.com/yunfanLu/IMU-Registration-Tool)

## Citation

If you use the SEE-600K dataset or SEE-Net in your research, please cite:

```bibtex
@article{lu2025SEE,
  title={SEE: See Everything Every Time - Adaptive Brightness Adjustment for Broad Light Range Images via Events},
  author={Yunfan Lu, Xiaogang Xu, Hao Lu, Yanlin Qian, Pengteng Li, Huizai Yao, Bin Yang, Junyi Li, Qianyi Cai, Weiyu Guo, Hui Xiong},
  year={2025},
}
```

## Contact

- **Yunfan Lu**: <ylu066@connect.hkust-gz.edu.cn>
- **GitHub**: [yunfanLu](https://github.com/yunfanLu)
