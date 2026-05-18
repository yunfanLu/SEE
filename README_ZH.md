# SEE：See Everything Every Time

<p align="center">
  <img src="./images/SEE-logo.jpg" alt="SEE Challenge 2026 logo" width="260">
</p>

**SEE** 是一个利用 RGB 图像与事件相机数据，在宽光照范围内进行自适应亮度调整的框架。本仓库提供 **SEE-Net** 的代码以及 **SEE-600K** 数据集相关资源。

SEE-Net 使用事件信息作为引导，在低光、过曝、混合光照和高对比度场景中增强图像，同时保持场景结构和自然观感。

## 主要特性

- **事件引导的亮度调整**：结合 RGB 图像与事件数据，实现宽光照范围的图像增强。
- **连续曝光控制**：通过曝光提示支持像素级亮度调整。
- **轻量级基线模型**：SEE-Net 约有 **1.9M 参数**。
- **大规模数据集支持**：基于 SEE-600K 构建，该数据集包含 **610,126 张图像及对应事件数据**，覆盖 **202 个真实场景**。

## 视觉亮点

| 问题定义 | 数据集样例 |
| --- | --- |
| <img src="./images/2-Problem-Define.jpg" alt="SEE challenge task definition with challenging input, event guidance, and restored output"> | <img src="./images/0-Dataset-Sample.jpg" alt="SEE-600K dataset samples under low, normal, and high lighting conditions"> |
| 复杂光照下的输入 RGB + 事件表示 -> 亮度调整后的 RGB 输出。 | 示例覆盖低光、正常光、高光、混合光照和事件视图。 |

| 场景覆盖 | 基线可视化 |
| --- | --- |
| <img src="./images/1-MoreExamples-WordCloud.jpg" alt="SEE-600K scene examples and scenario word cloud"> | <img src="./images/3-BaseLine-Release.jpg" alt="SEE baseline visualization results produced by the released code"> |
| 202 个真实世界场景，覆盖广泛场景类别和多个光照组。 | SEE-Net 使用 RGB 图像、事件数据和亮度提示实现可控输出曝光。 |

## 数据集下载

本项目使用的数据集已公开发布：

- **SEE-600K 数据集**
  - [OneDrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/ylu066_connect_hkust-gz_edu_cn/EkNi59p2uHJFjxyeQraiVhgBSs1GnxK4DyCUP-uZhEspCA?e=ZpwOvY)
  - [Hugging Face](https://huggingface.co/datasets/yunfanlu/SEE-600K)
- **SDE 数据集**：[SDE Dataset GitHub](https://github.com/EthanLiang99/EvLight)

## 预训练模型

已发布基线方法的预训练权重、评测日志和实验文件可从 Google Drive 下载：

<https://drive.google.com/drive/folders/1SWR9YVIrqFEkGGKrv3wSC5RqmMIEYjV2?usp=sharing>

该文件夹按方法组织，包含 `EIFT_AAAI_SEE`、`eSl_SEE`、`EvLight` 和 `SEENet_SEE` 的权重，以及相关日志和 `TEST_RESULTS.md`。

## 文档

- [教程](docs/TUTORIAL_ZH.md)：环境配置、数据集准备、训练、评测、推理和常见问题。
- [CodaBench 提交指南](docs/CODABENCH_SUBMISSION_GUIDE_ZH.md)：将结果打包并提交到 CodaBench 的逐步说明。
- [CodaBench 页面说明](CODABENCH_README_ZH.md)：用于 SEE Challenge CodaBench 页面的一份简洁独立介绍。
- [测试结果](docs/TEST_RESULTS.md)：SEE 数据集上的基线结果。
- [Mini-GT 测试结果](docs/TEST_MINI_RESULTS.md)：SEE mini GT 数据集上的基线结果。

英文版本也可查看：

- [README.md](README.md)
- [docs/TUTORIAL.md](docs/TUTORIAL.md)
- [docs/CODABENCH_SUBMISSION_GUIDE.md](docs/CODABENCH_SUBMISSION_GUIDE.md)
- [CODABENCH_README.md](CODABENCH_README.md)

## 相关链接

- **Workshop Website**: [https://eventbasemultimodalvision.github.io](https://eventbasemultimodalvision.github.io)
- **Competition Page**: [https://www.codabench.org/competitions/16195/](https://www.codabench.org/competitions/16195/#/pages-tab)
- **Competition Forum**: [https://www.codabench.org/forums/15944/](https://www.codabench.org/forums/15944/)
- **Competition Report**: [https://arxiv.org/abs/2502.21120](https://arxiv.org/abs/2502.21120)
- **IMU Registration Tool**: [https://github.com/yunfanLu/IMU-Registration-Tool](https://github.com/yunfanLu/IMU-Registration-Tool)

## 引用

如果您在研究中使用 SEE-600K 数据集或 SEE-Net，请引用：

```bibtex
@article{lu2025SEE,
  title={SEE: See Everything Every Time - Adaptive Brightness Adjustment for Broad Light Range Images via Events},
  author={Yunfan Lu, Xiaogang Xu, Hao Lu, Yanlin Qian, Pengteng Li, Huizai Yao, Bin Yang, Junyi Li, Qianyi Cai, Weiyu Guo, Hui Xiong},
  year={2025},
}
```

## 联系方式

- **Yunfan Lu**: <ylu066@connect.hkust-gz.edu.cn>
- **GitHub**: [yunfanLu](https://github.com/yunfanLu)
