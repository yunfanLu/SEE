# SEE Challenge 2026: Event-Guided Brightness Adjustment on SEE-600K

Welcome to the **SEE Challenge 2026**, held with **EBMV @ ECCV 2026**. This competition studies **event-guided brightness adjustment under broad lighting conditions**.

The goal is to recover a well-exposed RGB image from a challenging RGB input and its corresponding event-camera signal. Participants may use the official baseline code, the SEE-600K dataset, external data, and pretrained models, as long as external resources are clearly reported in the final technical report.

## Start Here

| Need | Link |
| --- | --- |
| Code, baseline, dataset links, pretrained models | [GitHub: yunfanLu/SEE](https://github.com/yunfanLu/SEE) |
| Full setup, training, inference, and evaluation tutorial | [docs/TUTORIAL.md](https://github.com/yunfanLu/SEE/blob/main/docs/TUTORIAL.md) |
| CodaBench packaging and upload instructions | [docs/CODABENCH_SUBMISSION_GUIDE.md](https://github.com/yunfanLu/SEE/blob/main/docs/CODABENCH_SUBMISSION_GUIDE.md) |
| Chinese competition guide | [docs/COMPETITION_GUIDE_ZH.md](https://github.com/yunfanLu/SEE/blob/main/docs/COMPETITION_GUIDE_ZH.md) |
| Baseline results | [docs/TEST_RESULTS.md](https://github.com/yunfanLu/SEE/blob/main/docs/TEST_RESULTS.md) |
| Dataset | [SEE-600K on Hugging Face](https://huggingface.co/datasets/yunfanlu/SEE-600K) |

Recommended workflow:

1. Clone the repository and install dependencies.
2. Download SEE-600K.
3. Train or adapt a model using the official training split.
4. Run inference at the original image size, `346 x 260`.
5. Collect prediction images with the provided CodaBench script.
6. Zip the collected folder and upload it to CodaBench.

## Eval Phase

The Eval Phase uses the final test dataset:
<https://drive.google.com/file/d/1-qQ_rKt5nB_8bb0kCZGhW_Epqkpi_I34/view?usp=drive_link>

Before submitting to the Eval Phase, please update the repository and follow the Eval Phase workflow in [docs/CODABENCH_SUBMISSION_GUIDE.md](https://github.com/yunfanLu/SEE/blob/main/docs/CODABENCH_SUBMISSION_GUIDE.md#3-eval-phase-submission-workflow). In short:

- run inference on the `SEE-600K-eval/DVS346-eval` test data at `346 x 260`;
- set `all_groups_as_testing: true` in the testing dataset config;
- collect predictions with `codabench/collect_eval_phase_pred.py`;
- zip the collected folder and upload it to the CodaBench Eval Phase page.

## Task

Given:

- an RGB image captured under difficult illumination;
- the corresponding event stream or event representation;
- optional brightness or exposure metadata when available;

participants should output a **brightness-adjusted RGB image**.

The output should:

- preserve scene structure and image details;
- produce natural color and exposure;
- avoid unrealistic hallucinated content;
- match the original SEE image resolution for submission.

## Dataset: SEE-600K

The challenge uses **SEE-600K**, a large-scale RGB-event dataset for broad-light-range image enhancement.

| Property | Value |
| --- | --- |
| Images | 610,126 |
| Real-world scenarios | 202 |
| Modalities | RGB frames + event data |
| Lighting conditions | low light, normal light, high light, mixed illumination |
| Image resolution | `346 x 260` |

Download:

- Hugging Face: [yunfanlu/SEE-600K](https://huggingface.co/datasets/yunfanlu/SEE-600K)
- Additional mirrors and notes: [GitHub README](https://github.com/yunfanLu/SEE#dataset-download)

Expected dataset structure and setup instructions are documented in [docs/TUTORIAL.md](https://github.com/yunfanLu/SEE/blob/main/docs/TUTORIAL.md#2-prepare-the-dataset).

## Train/Test Split

The following scenario groups are reserved for official testing and **must not be used for training**:

| Type | Reserved groups |
| --- | --- |
| Indoor | 000, 001, 002, 006, 012, 018, 030, 042, 048, 054, 060, 065, 070, 074, 075 |
| Outdoor | 100, 106, 112, 118, 124, 130, 136, 142, 148, 154, 160, 166, 173, 184, 189, 194, 200, 206, 212, 217, 222, 225 |

All other groups are available for training. See the split details in [docs/TUTORIAL.md](https://github.com/yunfanLu/SEE/blob/main/docs/TUTORIAL.md#23-traintest-split).

## Baseline: SEE-Net

The official repository provides **SEE-Net**, a compact baseline with approximately **1.9M parameters**. It uses:

- an RGB frame;
- event data;
- a brightness/exposure prompt.

The baseline supports controllable brightness adjustment through the prompt value `B`:

- `B ~= 0.5`: normal exposure;
- `B > 0.5`: brighter output;
- `B < 0.5`: darker output.

Baseline setup, training, inference, and evaluation are documented here:

- Setup: [docs/TUTORIAL.md#3-set-up-the-python-environment](https://github.com/yunfanLu/SEE/blob/main/docs/TUTORIAL.md#3-set-up-the-python-environment)
- Training: [docs/TUTORIAL.md#5-train-see-net](https://github.com/yunfanLu/SEE/blob/main/docs/TUTORIAL.md#5-train-see-net)
- Inference and evaluation: [docs/TUTORIAL.md#6-evaluate-or-run-inference](https://github.com/yunfanLu/SEE/blob/main/docs/TUTORIAL.md#6-evaluate-or-run-inference)
- Pretrained checkpoints: [GitHub README: Pretrained Models](https://github.com/yunfanLu/SEE#pretrained-models)

## Quick Commands

Clone and install:

```bash
git clone https://github.com/yunfanLu/SEE.git
cd SEE

conda create -n see python=3.10 -y
conda activate see
pip install -r requirements.txt
```

Download SEE-600K with Hugging Face CLI:

```bash
pip install huggingface_hub
hf download yunfanlu/SEE-600K --local-dir ./SEE-600K
```

Before inference for submission, make sure the YAML crop size is the original image size:

```yaml
crop_w: 346
crop_h: 260
```

Run inference with a trained checkpoint:

```bash
python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/vis/" \
  --alsologtostderr=True \
  --RESUME_PATH="./logs/SEE/SEENet_SEE/checkpoint.pth.tar" \
  --TEST_ONLY=True \
  --VISUALIZE=True \
  --VAL_BATCH_SIZE=4
```

## Submission Format

After inference, the repository will produce a `vis` folder. Use the provided script to collect only the prediction images required by CodaBench:

```bash
python codabench/collect_codabench_pred.py ${prediction_vis_directory} ${output_directory}
```

Example:

```bash
python codabench/collect_codabench_pred.py /path/to/vis /path/to/output_dir
```

Then create the zip file:

```bash
cd /path/to/output_dir
zip -r submission.zip ./*
```

Upload `submission.zip` on the CodaBench submission page.

Before uploading, check that:

- prediction images are `.png` files;
- prediction images are saved at `346 x 260`;
- the zip contains the collected prediction folder structure, not an extra parent directory;
- filenames and folders are preserved by the collection script;
- the submitted images correspond to the main prediction output used for evaluation. If you use the official `vis` naming convention, the evaluator prioritizes prediction files containing `_p0_`.

Full submission instructions: [docs/CODABENCH_SUBMISSION_GUIDE.md](https://github.com/yunfanLu/SEE/blob/main/docs/CODABENCH_SUBMISSION_GUIDE.md).

## Evaluation

Ranking is based on reconstruction quality against the official reference images.

| Metric | Role | Direction |
| --- | --- | --- |
| PSNR | Primary ranking metric | Higher is better |
| SSIM | Secondary metric | Higher is better |

The scorer reports overall PSNR/SSIM and category-level scores for `high-normal`, `low-normal`, and `normal-normal` samples. Missing or invalid predictions receive zero-score penalties for the affected samples.

The validation server is intended for development and format checking. Final ranking is determined by the official test phase.

## Important Dates

| Date | Event |
| --- | --- |
| May 10, 2026 | Competition website opens |
| May 25, 2026 | Validation server online |
| June 25, 2026 | Test data released and test server online |
| July 3, 2026 | Test submission deadline |
| July 10, 2026 | Results announcement, tentative |

## Rules

- Official SEE-600K training and validation data may be used.
- External training data, synthetic data, and pretrained models are allowed, but must be clearly described in the technical report.
- Reserved test groups and hidden ground-truth images must not be used for training or manual tuning.
- Do not exploit the evaluation server or submission system.
- Top-ranked teams may be asked to submit a short technical report and provide reproducibility details.

## FAQ

**Which repository should I use?**  
Use the official repository: [https://github.com/yunfanLu/SEE](https://github.com/yunfanLu/SEE).

**Where are the training and inference commands?**  
They are in [docs/TUTORIAL.md](https://github.com/yunfanLu/SEE/blob/main/docs/TUTORIAL.md).

**How do I prepare the CodaBench zip?**  
Follow [docs/CODABENCH_SUBMISSION_GUIDE.md](https://github.com/yunfanLu/SEE/blob/main/docs/CODABENCH_SUBMISSION_GUIDE.md). The key steps are: generate `vis`, collect predictions with `codabench/collect_codabench_pred.py`, then zip the collected output folder.

**Why must inference use `346 x 260`?**  
This is the original SEE image size. Submissions with a different output size may fail evaluation or produce invalid scores.

**Can I use external data or pretrained models?**  
Yes, but all external resources must be reported in the final technical report.

## Links

- Competition page: [https://www.codabench.org/competitions/16195/](https://www.codabench.org/competitions/16195/)
- Competition forum: [https://www.codabench.org/forums/15944/](https://www.codabench.org/forums/15944/)
- GitHub repository: [https://github.com/yunfanLu/SEE](https://github.com/yunfanLu/SEE)
- Dataset: [https://huggingface.co/datasets/yunfanlu/SEE-600K](https://huggingface.co/datasets/yunfanlu/SEE-600K)
- Workshop website: [https://eventbasemultimodalvision.github.io](https://eventbasemultimodalvision.github.io)
- Competition report: [https://arxiv.org/abs/2502.21120](https://arxiv.org/abs/2502.21120)

## Citation

If you use SEE-600K, SEE-Net, or the official baseline code, please cite:

```bibtex
@article{lu2025SEE,
  title={SEE: See Everything Every Time - Adaptive Brightness Adjustment for Broad Light Range Images via Events},
  author={Yunfan Lu, Xiaogang Xu, Hao Lu, Yanlin Qian, Pengteng Li, Huizai Yao, Bin Yang, Junyi Li, Qianyi Cai, Weiyu Guo, Hui Xiong},
  year={2025},
}
```

## Contact

- Yunfan Lu: <ylu066@connect.hkust-gz.edu.cn>
- Mingchao Xu：<mingchao.xu.casia@gmail.com>
