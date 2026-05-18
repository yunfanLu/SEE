# SEE Challenge 2026：事件引导的低层视觉成像

<p align="center">
  <img src="./images/SEE-logo.jpg" alt="SEE Challenge 2026 logo" width="260">
</p>
欢迎参加 **SEE Challenge 2026**。本挑战隶属于 **EBMV @ ECCV 2026：Event-Based Multimodal Vision: Imaging, Perception, and Understanding**。

本挑战关注**宽光照条件下的事件引导亮度调整**。参赛者将获得在复杂光照下采集的 RGB 图像及其对应的事件流或事件表示，并生成曝光良好、结构忠实且视觉清晰的亮度调整后 RGB 图像。

## 任务

给定：

- 一张在复杂光照下采集的 RGB 图像；
- 对应的事件流或事件表示；
- 如提供，则还包括可选的亮度提示或元数据；

参赛者应输出一张**亮度调整后的 RGB 图像**。

输出应保持场景结构，恢复有用细节，维持自然色彩表现，并避免不真实的幻觉内容。

## 数据集

本挑战使用 **SEE-600K**，这是一个大规模 RGB-事件数据集，包含：

- 610,126 张图像及对应事件数据；
- 202 个真实世界场景；
- 低光、正常光和高光条件；
- 每个场景包含多个光照组；
- 面向事件引导图像增强的宽范围光照变化。

数据集：<https://huggingface.co/datasets/yunfanlu/SEE-600K>

## 评测

官方评测将在测试集上进行。

| 指标 | 含义 | 排名依据 |
| --- | --- | --- |
| PSNR | 像素级重建精度 | 主要指标，越高越好 |
| SSIM | 结构相似性 | 次要指标，越高越好 |
| LPIPS | 感知距离 | 次要指标，越低越好 |

## 时间安排

| 日期 | 事件 |
| --- | --- |
| 2026 年 5 月 10 日 | 挑战网站开放 |
| 2026 年 5 月 25 日 | 验证服务器上线 |
| 2026 年 6 月 25 日 | 测试数据发布，测试服务器上线 |
| 2026 年 7 月 3 日 | 测试提交截止 |
| 2026 年 7 月 10 日 | 结果公布，暂定 |

## 视觉概览

| 问题定义 | 数据集样例 |
| --- | --- |
| <img src="./images/2-Problem-Define.jpg" alt="SEE challenge task definition with challenging input, event guidance, and restored output"> | <img src="./images/0-Dataset-Sample.jpg" alt="SEE-600K dataset samples under low, normal, and high lighting conditions"> |
| 复杂光照下的输入 RGB + 事件表示 -> 亮度调整后的 RGB 输出。 | 示例覆盖低光、正常光、高光、混合光照和事件视图。 |

| 场景覆盖 | 基线可视化 |
| --- | --- |
| <img src="./images/1-MoreExamples-WordCloud.jpg" alt="SEE-600K scene examples and scenario word cloud"> | <img src="./images/3-BaseLine-Release.jpg" alt="SEE baseline visualization results produced by the released code"> |
| 202 个真实世界场景，覆盖广泛场景类别和多个光照组。 | SEE-Net 使用 RGB 图像、事件数据和亮度提示实现可控输出曝光。 |

## 快速开始

1. 访问 GitHub 仓库获取代码和文档：<https://github.com/yunfanLu/SEE>。
2. 从 Hugging Face 下载 SEE-600K：<https://huggingface.co/datasets/yunfanlu/SEE-600K>。
3. 按照仓库教程配置环境和数据集：<https://github.com/yunfanLu/SEE/blob/main/docs/TUTORIAL_ZH.md>。

### 基线流程

1. **训练模型。** 请参考教程中的训练章节：<https://github.com/yunfanLu/SEE/blob/main/docs/TUTORIAL_ZH.md#5-训练-see-net>。
2. **运行推理。** 请参考教程中的评测和推理章节生成 `vis` 文件夹：<https://github.com/yunfanLu/SEE/blob/main/docs/TUTORIAL_ZH.md#6-评测或运行推理>。
3. **将 mini 测试集预测结果提交到 CodaBench。** 按照 CodaBench 提交指南，从 `vis` 中收集所需预测结果、打包并上传 zip 文件：<https://github.com/yunfanLu/SEE/blob/main/docs/CODABENCH_SUBMISSION_GUIDE_ZH.md>。

## 链接

- **GitHub Repository**: <https://github.com/yunfanLu/SEE>
- **Workshop Website**: <https://eventbasemultimodalvision.github.io>
- **Competition Page**: <https://www.codabench.org/competitions/16195/>
- **Competition Forum**: <https://www.codabench.org/forums/15944/>
- **Competition Report**: <https://arxiv.org/abs/2502.21120>
