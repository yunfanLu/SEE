# SEE Challenge 2026: Event-Guided Low-Level Imaging

<p align="center">
  <img src="./images/SEE-logo.jpg" alt="SEE Challenge 2026 logo" width="260">
</p>
Welcome to the **SEE Challenge 2026**, associated with **EBMV @ ECCV 2026: Event-Based Multimodal Vision: Imaging, Perception, and Understanding**.

This challenge focuses on **event-guided brightness adjustment under broad lighting conditions**. Participants receive RGB frames captured under challenging illumination together with corresponding event streams or event representations, and produce brightness-adjusted RGB images that are well exposed, structurally faithful, and visually clear.

## Task

Given:

- an RGB image captured under challenging illumination;
- the corresponding event stream or event representation;
- an optional brightness prompt or metadata if provided;

participants should output a **brightness-adjusted RGB image**.

The output should preserve scene structure, recover useful details, maintain natural color appearance, and avoid unrealistic hallucination.

## Dataset

The challenge uses **SEE-600K**, a large-scale RGB-event dataset containing:

- 610,126 images with corresponding event data;
- 202 real-world scenarios;
- low-light, normal-light, and high-light conditions;
- multiple lighting groups per scene;
- broad illumination variations for event-guided image enhancement.

Dataset: <https://huggingface.co/datasets/yunfanlu/SEE-600K>

## Evaluation

Official evaluation is conducted on the test set.

| Metric | Meaning | Ranking |
| --- | --- | --- |
| PSNR | Pixel-level reconstruction accuracy | Primary, higher is better |
| SSIM | Structural similarity | Secondary, higher is better |
| LPIPS | Perceptual distance | Secondary, lower is better |

## Timeline

| Date | Event |
| --- | --- |
| May 10, 2026 | Challenge website opens |
| May 25, 2026 | Validation server online |
| June 25, 2026 | Test data released and test server online |
| July 3, 2026 | Test submission deadline |
| July 10, 2026 | Results announcement, tentative |

## Visual Overview

| Problem Definition | Dataset Samples |
| --- | --- |
| <img src="./images/2-Problem-Define.jpg" alt="SEE challenge task definition with challenging input, event guidance, and restored output"> | <img src="./images/0-Dataset-Sample.jpg" alt="SEE-600K dataset samples under low, normal, and high lighting conditions"> |
| Input RGB under challenging illumination + event representation -> brightness-adjusted RGB output. | Examples cover low-light, normal-light, high-light, mixed illumination, and event views. |

| Scenario Coverage | Baseline Visualization |
| --- | --- |
| <img src="./images/1-MoreExamples-WordCloud.jpg" alt="SEE-600K scene examples and scenario word cloud"> | <img src="./images/3-BaseLine-Release.jpg" alt="SEE baseline visualization results produced by the released code"> |
| 202 real-world scenarios with broad scene categories and multiple lighting groups. | SEE-Net uses RGB frames, event data, and a brightness prompt for controllable output exposure. |

## Get Started

1. Visit the GitHub repository for code and documentation: <https://github.com/yunfanLu/SEE>.
2. Download SEE-600K from Hugging Face: <https://huggingface.co/datasets/yunfanlu/SEE-600K>.
3. Follow the repository tutorial to set up the environment and run the baseline.
4. Follow the repository CodaBench submission guide when preparing your submission package.

## Links

- **GitHub Repository**: <https://github.com/yunfanLu/SEE>
- **Workshop Website**: <https://eventbasemultimodalvision.github.io>
- **Competition Page**: <https://www.codabench.org/competitions/16195/>
- **Competition Forum**: <https://www.codabench.org/forums/15944/>
- **Competition Report**: <https://arxiv.org/abs/2502.21120>
