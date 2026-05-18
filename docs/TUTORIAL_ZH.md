# SEE 教程

本教程说明如何配置 SEE 仓库、准备 SEE-600K 数据集、运行基线训练、评测模型权重并生成推理结果。

项目概览、数据集链接、预训练模型链接和相关资源请见 [README_ZH.md](../README_ZH.md)。CodaBench 打包和上传说明请见 [CODABENCH_SUBMISSION_GUIDE_ZH.md](./CODABENCH_SUBMISSION_GUIDE_ZH.md)。

---

## 1. 克隆仓库

```bash
git clone https://github.com/yunfanLu/SEE.git
cd SEE
```

---

## 2. 准备数据集

请从 [README_ZH.md](../README_ZH.md#数据集下载) 中列出的链接下载 SEE-600K。Hugging Face 版本已经完成对齐和分类，因此不需要额外的配准步骤。

### 2.1 使用 Hugging Face CLI 下载

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载完整数据集
hf download yunfanlu/SEE-600K --local-dir ./SEE-600K
```

如果存储空间或带宽有限，可以只下载所需子集：

```bash
# 下载指定子集（例如 RoboticArm）
hf download yunfanlu/SEE-600K --include "RoboticArm/*" --local-dir ./SEE-600K
```

### 2.2 预期数据集结构

下载后，数据集结构应如下所示：

```text
SEE-600K/
└── RoboticArm/
    ├── 000-indoor_ceiling_table_light/       # 组文件夹（场景）
    │   ├── registrate_result.json            # IMU 配准时间戳
    │   ├── exposure_state.json               # 每个视频的曝光分类
    │   ├── video_name_1/                     # 一段视频录制
    │   │   ├── frame_event/
    │   │   │   ├── <timestamp>_<ts_start>_<ts_end>_<ts_exp_start>_<ts_exp_end>.png  # RGB 图像帧
    │   │   │   ├── <ts_start>_<ts_end>.npy          # 事件数据（N x 4：t, x, y, polarity）
    │   │   │   └── <ts_start>_<ts_end>_vis.png      # 事件可视化
    │   │   └── imu.npy                       # IMU 数据
    │   ├── video_name_2/
    │   │   └── ...
    │   └── ...
    ├── 001-indoor_wall_displayboard_wood_luggage/
    └── ...
```

### 2.3 训练/测试划分

数据集包含 202 个场景。以下组保留为**测试集**，不得用于训练：

| 类型 | 组编号 |
| --- | --- |
| 室内 | 000, 001, 002, 006, 012, 018, 030, 042, 048, 054, 060, 065, 070, 074, 075 |
| 室外 | 100, 106, 112, 118, 124, 130, 136, 142, 148, 154, 160, 166, 173, 184, 189, 194, 200, 206, 212, 217, 222, 225 |

其余所有组用于**训练**。

### 2.4 可选：原始数据处理

如果下载的是原始 `.aedat4` 文件，而不是已经预处理的 Hugging Face 版本，请按如下步骤处理：

```bash
# Step 0: 从 .aedat4 文件中提取图像帧、事件和 IMU
python tools/0-dataset-preprocess/0-extract_data_from_aedat4/extract_aedat4.py \
    --aedat_folder="dataset/raw_aedat4/" \
    --video_folder="dataset/processed/"

# Step 1-2: IMU 标定（用于配准）
python tools/0-dataset-preprocess/2-imu-calibration/cal_constant_bias.py

# Step 3: 基于 IMU 的时间配准
python registration/main.py --video_group_root="dataset/RoboticArm/"

# Step 4: 曝光分类
python registration/exposure_classifier.py --root="dataset/RoboticArm/"
```

IMU 配准工具也可在 <https://github.com/yunfanLu/IMU-Registration-Tool> 获取。

---

## 3. 配置 Python 环境

### 3.1 创建 Conda 环境

```bash
conda create -n see python=3.10 -y
conda activate see
```

### 3.2 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 文件包含：

```text
PyYAML
torch
absl-py
easydict
pudb
opencv-python
numpy
timm
expecttest
tensorboard
scipy
numba
einops
kornia
```

### 3.3 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

进行 GPU 训练时，CUDA 应可用。

---

## 4. 配置基线模型

编辑 `options/SEE/SEENet_SEE.yaml`，将 `DATASET.root` 更新为本地数据集路径：

```yaml
DATASET:
  NAME: see_everything_everytime_dataset
  root: /path/to/your/SEE-600K/RoboticArm/    # <-- 将这里改为你的数据集路径
```

例如：

```yaml
DATASET:
  root: /home/user/data/SEE-600K/RoboticArm/
```

`options/SEE/SEENet_SEE.yaml` 中的默认基线配置如下：

| 参数 | 数值 | 说明 |
| --- | --- | --- |
| `TRAIN_BATCH_SIZE` | 2 | 每张 GPU 的 batch size |
| `VAL_BATCH_SIZE` | 2 | 验证 batch size |
| `START_EPOCH` | 0 | 起始 epoch |
| `END_EPOCH` | 20 | 训练总 epoch 数 |
| `OPTIMIZER.NAME` | Adam | 优化器 |
| `OPTIMIZER.LR` | 0.0001 | 学习率 |
| `OPTIMIZER.LR_SCHEDULER` | cosine | 学习率调度器 |
| `LOSS` | L1-Charbonnier + Gradient | 损失函数 |
| `DATASET.crop_h` | 128 | 裁剪高度 |
| `DATASET.crop_w` | 256 | 裁剪宽度 |
| `MODEL.NAME` | SEENet | 模型结构 |
| `MODEL.loop` | 10 | 稀疏编码器迭代次数 |
| `MODEL.C1/C2` | 96 | 通道维度 |
| `MIX_PRECISION` | true | 混合精度训练 |

---

## 5. 训练 SEE-Net

运行基线训练命令：

```bash
export PYTHONPATH="./":$PYTHONPATH

python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True
```

也可以启动提供的脚本：

```bash
sh options/SEE/SEENet_SEE.sh
```

多 GPU 训练时，先设置可见 GPU 编号，再运行相同命令：

```bash
export CUDA_VISIBLE_DEVICES="0,1"

python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True
```

代码会自动使用 `nn.DataParallel` 进行多 GPU 训练。

### 5.1 使用 TensorBoard 监控训练

```bash
tensorboard --logdir=./logs/SEE/SEENet_SEE/ --port=6006 --bind_all
```

随后在浏览器中打开 `http://localhost:6006`。如果在远程服务器上运行，请使用 SSH 端口转发：

```bash
ssh -L 6006:localhost:6006 user@server_ip
```

### 5.2 从检查点恢复训练

```bash
python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True \
  --RESUME_PATH="./logs/SEE/SEENet_SEE/checkpoint.pth.tar" \
  --RESUME_SET_EPOCH=True
```

---

## 6. 评测或运行推理

训练完成后，可使用训练好的检查点运行验证：

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

主要测试参数如下：

| 参数 | 说明 |
| --- | --- |
| `--TEST_ONLY=True` | 跳过训练，只运行验证 |
| `--VISUALIZE=True` | 在验证期间保存输出图像 |
| `--RESUME_PATH=<path>` | 训练好检查点的路径 |
| `--VAL_BATCH_SIZE=4` | 在显存允许时增大 batch size 以加快推理 |

评测可视化文件夹时，运行：

```bash
python tools/2-eval-for-vis-folder/SEE_eval_for_model.py \
    --root="/path/to/your/vis/folder" \
    --log_dir="/path/to/your/log/dir" \
    --alsologtostderr=True
```

使用显式 ground-truth 根目录进行评测时，运行：

```bash
python tools/2-eval-for-vis-folder/SEE_eval_for_model_with_gt_root.py \
    --root="/path/to/your/eval/folder" \
    --gt_root="/path/to/your/gt/root" \
    --log_dir="/path/to/your/log/dir" \
    --alsologtostderr=True
```

基线指标汇总见 [TEST_RESULTS.md](./TEST_RESULTS.md) 和 [TEST_MINI_RESULTS.md](./TEST_MINI_RESULTS.md)。

---

## 7. 使用曝光提示进行亮度调整

SEE-Net 通过曝光提示 **B** 支持连续亮度控制：

- **B ≈ 0.5**：正常曝光，也是典型工作范围；
- **B > 0.5**：输出更亮；
- **B < 0.5**：输出更暗。

模型在解码阶段应用该提示：

```python
exposure_prompt = torch.ones(size=(B, 1, 1, 1)) * exposure_B
exposure_prompt = exposure_prompt.to(images.device)
exposure_normal_reconstructed = self._decoding(light_inr, exposure_prompt)
```

测试时，模型会生成三类输出：

- **PRD**：使用目标曝光提示得到的预测结果；
- **SLR**：使用输入均值进行的自监督重建；
- **NLR**：使用固定 `B=0.4` 的标准光照重建。

**B** 的极端取值主要用于分析和可视化，而不是作为目标输出。该框架面向亮度调整，而不是完整 HDR 重建。

---

## 8. 常见问题

### 需要多少 GPU 显存？

基线 SEENet 模型约有 **1.9M 参数**。在 batch size 为 2、裁剪尺寸为 128×256 时，约需要 **6-8 GB** GPU 显存。可根据硬件情况调整 `TRAIN_BATCH_SIZE` 和裁剪尺寸。

### 可以使用外部数据吗？

可以。允许使用外部训练数据、预训练模型和合成数据，但必须在最终技术报告中**明确说明**。

### 训练需要多长时间？

使用单张 A100 GPU 时，20 个 epoch 的基线训练大约需要 **10-15 小时**。训练时间会随 GPU 型号而变化。

### 为什么 TensorBoard 是空的？

请检查以下内容：

1. `--logdir` 指向包含 `events.out.tfevents.*` 文件的目录。
2. 训练已经开始，并至少记录了一个 epoch；可检查配置中的 `LOG_INTERVAL`。
3. 如果使用远程服务器，请确认 SSH 端口转发配置正确。

---

## 9. 引用

如果使用 SEE-600K 数据集或 SEE-Net，请引用：

```bibtex
@article{lu2025SEE,
  title={SEE: See Everything Every Time - Adaptive Brightness Adjustment for Broad Light Range Images via Events},
  author={Yunfan Lu, Xiaogang Xu, Hao Lu, Yanlin Qian, Pengteng Li, Huizai Yao, Bin Yang, Junyi Li, Qianyi Cai, Weiyu Guo, Hui Xiong},
  year={2025},
}
```

---

## 10. 联系方式

- **Yunfan Lu**: <ylu066@connect.hkust-gz.edu.cn>
- **GitHub**: [yunfanLu](https://github.com/yunfanLu)
