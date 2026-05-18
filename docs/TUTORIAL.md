# EBMV 2026 Low-Level Imaging Challenge: Getting Started Tutorial

This tutorial guides you through the entire process of participating in the **EBMV 2026 Low-Level Imaging Challenge**, from dataset preparation to running the baseline model.

- **Competition Page**: <https://www.codabench.org/competitions/16195/>
- **Competition Report**: <https://arxiv.org/abs/2502.21120>
- **Baseline Code**: <https://github.com/yunfanLu/SEE>
- **Dataset**: <https://huggingface.co/datasets/yunfanlu/SEE-600K>
- **Workshop**: <https://eventbasemultimodalvision.github.io>

***

## 1. Download and Prepare the Dataset

### 1.1 Download SEE-600K

The SEE-600K dataset is available on HuggingFace and OneDrive:

| Source      | Link                                                                                                                                            |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| HuggingFace | <https://huggingface.co/datasets/yunfanlu/SEE-600K>                                                                                             |
| OneDrive    | [Link](https://hkustgz-my.sharepoint.com/:f:/g/personal/ylu066_connect_hkust-gz_edu_cn/EkNi59p2uHJFjxyeQraiVhgBSs1GnxK4DyCUP-uZhEspCA?e=ZpwOvY) |

**Option A: Download from HuggingFace (Recommended)**

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download the entire dataset
hf download yunfanlu/SEE-600K --local-dir ./SEE-600K
```

**Option B: Download specific subsets**

If you have limited storage or bandwidth, you can download only the data you need:

```bash
# Download a specific split (e.g., RoboticArm)
hf download yunfanlu/SEE-600K --include "RoboticArm/*" --local-dir ./SEE-600K
```

> **Note**: If you download from HuggingFace, the data has already been aligned and can be used directly without additional registration.

### 1.2 Dataset Structure

After downloading, the dataset should have the following structure:

```
SEE-600K/
└── RoboticArm/
    ├── 000-indoor_ceiling_table_light/       # Group folder (scene)
    │   ├── registrate_result.json            # IMU registration timestamps
    │   ├── exposure_state.json               # Exposure classification per video
    │   ├── video_name_1/                     # One video recording
    │   │   ├── frame_event/
    │   │   │   ├── <timestamp>_<ts_start>_<ts_end>_<ts_exp_start>_<ts_exp_end>.png  # RGB frame
    │   │   │   ├── <ts_start>_<ts_end>.npy          # Events (N x 4: t, x, y, polarity)
    │   │   │   └── <ts_start>_<ts_end>_vis.png      # Event visualization
    │   │   └── imu.npy                       # IMU data
    │   ├── video_name_2/
    │   │   └── ...
    │   └── ...
    ├── 001-indoor_wall_displayboard_wood_luggage/
    └── ...
```

### 1.3 Train/Test Split

The dataset contains 202 scenarios. The following groups are reserved for **testing** (do NOT use them for training):

| Type    | Groups                                                                                                       |
| ------- | ------------------------------------------------------------------------------------------------------------ |
| Indoor  | 000, 001, 002, 006, 012, 018, 030, 042, 048, 054, 060, 065, 070, 074, 075                                    |
| Outdoor | 100, 106, 112, 118, 124, 130, 136, 142, 148, 154, 160, 166, 173, 184, 189, 194, 200, 206, 212, 217, 222, 225 |

All other groups are used for **training**.

### 1.4 (Optional) Raw Data Processing

If you download raw `.aedat4` files instead of the pre-processed HuggingFace version, you need to process them:

```bash
# Step 0: Extract frames, events, and IMU from .aedat4 files
python tools/0-dataset-preprocess/0-extract_data_from_aedat4/extract_aedat4.py \
    --aedat_folder="dataset/raw_aedat4/" \
    --video_folder="dataset/processed/"

# Step 1-2: IMU calibration (for registration)
python tools/0-dataset-preprocess/2-imu-calibration/cal_constant_bias.py

# Step 3: IMU-based temporal registration
python registration/main.py --video_group_root="dataset/RoboticArm/"

# Step 4: Exposure classification
python registration/exposure_classifier.py --root="dataset/RoboticArm/"
```

> **Note**: If you download from HuggingFace, these steps are **NOT** needed — the data is already aligned and classified.

***

## 2. Clone the SEE Codebase

```bash
git clone https://github.com/yunfanLu/SEE.git
cd SEE
```

***

## 3. Install Python Environment

### 3.1 Create a Conda Environment (Recommended)

```bash
conda create -n see python=3.10 -y
conda activate see
```

### 3.2 Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

```
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

### 3.3 Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Make sure CUDA is available for GPU training.

***

## 4. Run a Baseline Experiment

### 4.1 Configure the Dataset Path

Before training, you need to modify the dataset path in the YAML configuration file to match your local setup.

Edit `options/SEE/SEENet_SEE.yaml` and update the `DATASET.root` field:

```yaml
DATASET:
  NAME: see_everything_everytime_dataset
  root: /path/to/your/SEE-600K/RoboticArm/    # <-- Change this to your dataset path
```

For example, if your dataset is at `/home/user/data/SEE-600K/RoboticArm/`:

```yaml
DATASET:
  root: /home/user/data/SEE-600K/RoboticArm/
```

### 4.2 Training

Run the baseline training:

```bash
export PYTHONPATH="./":$PYTHONPATH

python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True
```

**Multi-GPU training** (if you have multiple GPUs):

```bash
export CUDA_VISIBLE_DEVICES="0,1"

python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True
```

The code uses `nn.DataParallel` for multi-GPU training automatically.

### 4.3 Key Training Configuration

The default configuration in `options/SEE/SEENet_SEE.yaml`:

| Parameter                | Value                     | Description               |
| ------------------------ | ------------------------- | ------------------------- |
| `TRAIN_BATCH_SIZE`       | 2                         | Batch size per GPU        |
| `VAL_BATCH_SIZE`         | 2                         | Validation batch size     |
| `START_EPOCH`            | 0                         | Starting epoch            |
| `END_EPOCH`              | 20                        | Total training epochs     |
| `OPTIMIZER.NAME`         | Adam                      | Optimizer                 |
| `OPTIMIZER.LR`           | 0.0001                    | Learning rate             |
| `OPTIMIZER.LR_SCHEDULER` | cosine                    | LR scheduler              |
| `LOSS`                   | L1-Charbonnier + Gradient | Loss functions            |
| `DATASET.crop_h`         | 128                       | Crop height               |
| `DATASET.crop_w`         | 256                       | Crop width                |
| `MODEL.NAME`             | SEENet                    | Model architecture        |
| `MODEL.loop`             | 10                        | Sparse encoder iterations |
| `MODEL.C1/C2`            | 96                        | Channel dimensions        |
| `MIX_PRECISION`          | true                      | Mixed precision training  |

### 4.4 Monitor Training with TensorBoard

```bash
tensorboard --logdir=./logs/SEE/SEENet_SEE/ --port=6006 --bind_all
```

Then open `http://localhost:6006` in your browser.

If you are on a remote server, use SSH port forwarding:

```bash
ssh -L 6006:localhost:6006 user@server_ip
```

### 4.5 Testing / Validation

After training, run validation with a trained checkpoint:

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

Key flags for testing:

| Flag                   | Description                                  |
| ---------------------- | -------------------------------------------- |
| `--TEST_ONLY=True`     | Skip training, run validation only           |
| `--VISUALIZE=True`     | Save output images during validation         |
| `--RESUME_PATH=<path>` | Path to trained checkpoint                   |
| `--VAL_BATCH_SIZE=4`   | Can increase batch size for faster inference |

### 4.6 Brightness Adjustment with Exposure Prompts

The SEE model supports continuous brightness control via an exposure prompt **B**:

- **B ≈ 0.5**: Normal exposure (recommended)
- **B > 0.5**: Brighter output
- **B < 0.5**: Darker output

This is controlled in the model's forward pass:

```python
exposure_prompt = torch.ones(size=(B, 1, 1, 1)) * exposure_B
output = model._decoding(light_inr, exposure_prompt)
```

During testing, the model generates three types of output:

- **PRD**: Prediction using target exposure prompt
- **SLR**: Self-supervised reconstruction (using input's own mean)
- **NLR**: Standard light reconstruction (using fixed B=0.4)

***

## 5. Prepare Submission

### 5.1 Submission Format

Each submission should be a ZIP file with the following structure:

```
submission.zip
└── results/
    ├── scene_000001.png
    ├── scene_000002.png
    ├── scene_000003.png
    └── ...
```

### 5.2 Requirements

1. Output filenames must match the input sample IDs
2. Images should be saved as **PNG** files
3. Output images should be **RGB** images
4. Output resolution must match the input/reference resolution
5. Do not change the folder structure
6. Do not include extra files unless explicitly required

### 5.3 Evaluation Metrics

| Metric | Description                         | Priority              |
| ------ | ----------------------------------- | --------------------- |
| PSNR   | Pixel-level reconstruction accuracy | **Primary** (ranking) |
| SSIM   | Structural similarity               | Secondary             |
| LPIPS  | Perceptual similarity               | Secondary             |

***

## 6. Tips and FAQ

### Q: How much GPU memory is needed?

The baseline model (SEENet) has only **1.9M parameters**. With batch size 2 and crop size 128×256, it requires approximately **6-8 GB** GPU memory. You can adjust `TRAIN_BATCH_SIZE` and crop size based on your hardware.

### Q: Can I use external data?

Yes, external training data, pretrained models, and synthetic data are allowed, but they must be **clearly reported** in the final technical report.

### Q: How long does training take?

With a single A100 GPU, the baseline training (20 epochs) takes approximately **10-15 hours**. Training time varies depending on your GPU.

### Q: The TensorBoard page is empty / no curves

Make sure:

1. The `--logdir` path points to the correct directory containing `events.out.tfevents.*` files
2. Training has started and logged at least one epoch (check `LOG_INTERVAL` in the config)
3. If using a remote server, ensure port forwarding is set up correctly

### Q: How to resume training from a checkpoint?

```bash
python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True \
  --RESUME_PATH="./logs/SEE/SEENet_SEE/checkpoint.pth.tar" \
  --RESUME_SET_EPOCH=True
```

***

## 7. Important Dates

| Date          | Event                                     |
| ------------- | ----------------------------------------- |
| May 15, 2026  | Challenge website opens                   |
| May 25, 2026  | Validation server online                  |
| June 25, 2026 | Test data released and test server online |
| July 3, 2026  | Test submission deadline                  |
| July 10, 2026 | Results announcement (tentative)          |

***

## 8. Citation

If you use the SEE-600K dataset or SEE-Net, please cite:

```bibtex
@article{lu2025SEE,
  title={SEE: See Everything Every Time - Adaptive Brightness Adjustment for Broad Light Range Images via Events},
  author={Yunfan Lu, Xiaogang Xu, Hao Lu, Yanlin Qian, Pengteng Li, Huizai Yao, Bin Yang, Junyi Li, Qianyi Cai, Weiyu Guo, Hui Xiong},
  year={2025},
}
```

***

## 9. Contact

- **Yunfan Lu**: <ylu066@connect.hkust-gz.edu.cn>
- **GitHub**: [yunfanLu](https://github.com/yunfanLu)
- **Competition Forum**: <https://www.codabench.org/forums/15944/>

