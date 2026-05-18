# SEE Tutorial

This tutorial explains how to set up the SEE repository, prepare the SEE-600K dataset, run baseline training, evaluate checkpoints, and generate inference outputs.

For project overview, dataset links, pretrained model links, and related resources, see [README.md](../README.md). For CodaBench packaging and upload instructions, see [CODABENCH_SUBMISSION_GUIDE.md](./CODABENCH_SUBMISSION_GUIDE.md).

---

## 1. Clone the Repository

```bash
git clone https://github.com/yunfanLu/SEE.git
cd SEE
```

---

## 2. Prepare the Dataset

Download SEE-600K from one of the links listed in [README.md](../README.md#dataset-download). The Hugging Face version is already aligned and classified, so no additional registration is required.

### 2.1 Download with Hugging Face CLI

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download the entire dataset
hf download yunfanlu/SEE-600K --local-dir ./SEE-600K
```

If you have limited storage or bandwidth, download only the required subset:

```bash
# Download a specific split (e.g., RoboticArm)
hf download yunfanlu/SEE-600K --include "RoboticArm/*" --local-dir ./SEE-600K
```

### 2.2 Expected Dataset Structure

After downloading, the dataset should have the following structure:

```text
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

### 2.3 Train/Test Split

The dataset contains 202 scenarios. The following groups are reserved for **testing** and must not be used for training:

| Type | Groups |
| --- | --- |
| Indoor | 000, 001, 002, 006, 012, 018, 030, 042, 048, 054, 060, 065, 070, 074, 075 |
| Outdoor | 100, 106, 112, 118, 124, 130, 136, 142, 148, 154, 160, 166, 173, 184, 189, 194, 200, 206, 212, 217, 222, 225 |

All other groups are used for **training**.

### 2.4 Optional Raw Data Processing

If you download raw `.aedat4` files instead of the pre-processed Hugging Face version, process them as follows:

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

The IMU registration tool is also available at <https://github.com/yunfanLu/IMU-Registration-Tool>.

---

## 3. Set Up the Python Environment

### 3.1 Create a Conda Environment

```bash
conda create -n see python=3.10 -y
conda activate see
```

### 3.2 Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:

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

### 3.3 Verify the Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

CUDA should be available for GPU training.

---

## 4. Configure the Baseline

Edit `options/SEE/SEENet_SEE.yaml` and update `DATASET.root` to match your local dataset path:

```yaml
DATASET:
  NAME: see_everything_everytime_dataset
  root: /path/to/your/SEE-600K/RoboticArm/    # <-- Change this to your dataset path
```

For example:

```yaml
DATASET:
  root: /home/user/data/SEE-600K/RoboticArm/
```

The default baseline configuration in `options/SEE/SEENet_SEE.yaml` is:

| Parameter | Value | Description |
| --- | --- | --- |
| `TRAIN_BATCH_SIZE` | 2 | Batch size per GPU |
| `VAL_BATCH_SIZE` | 2 | Validation batch size |
| `START_EPOCH` | 0 | Starting epoch |
| `END_EPOCH` | 20 | Total training epochs |
| `OPTIMIZER.NAME` | Adam | Optimizer |
| `OPTIMIZER.LR` | 0.0001 | Learning rate |
| `OPTIMIZER.LR_SCHEDULER` | cosine | LR scheduler |
| `LOSS` | L1-Charbonnier + Gradient | Loss functions |
| `DATASET.crop_h` | 128 | Crop height |
| `DATASET.crop_w` | 256 | Crop width |
| `MODEL.NAME` | SEENet | Model architecture |
| `MODEL.loop` | 10 | Sparse encoder iterations |
| `MODEL.C1/C2` | 96 | Channel dimensions |
| `MIX_PRECISION` | true | Mixed precision training |

---

## 5. Train SEE-Net

Run the baseline training command:

```bash
export PYTHONPATH="./":$PYTHONPATH

python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True
```

You can also launch the provided script:

```bash
sh options/SEE/SEENet_SEE.sh
```

For multi-GPU training, set the visible GPU IDs before running the same command:

```bash
export CUDA_VISIBLE_DEVICES="0,1"

python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True
```

The code uses `nn.DataParallel` for multi-GPU training automatically.

### 5.1 Monitor Training with TensorBoard

```bash
tensorboard --logdir=./logs/SEE/SEENet_SEE/ --port=6006 --bind_all
```

Then open `http://localhost:6006` in your browser. On a remote server, use SSH port forwarding:

```bash
ssh -L 6006:localhost:6006 user@server_ip
```

### 5.2 Resume Training from a Checkpoint

```bash
python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True \
  --RESUME_PATH="./logs/SEE/SEENet_SEE/checkpoint.pth.tar" \
  --RESUME_SET_EPOCH=True
```

---

## 6. Evaluate or Run Inference

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

Key testing flags:

| Flag | Description |
| --- | --- |
| `--TEST_ONLY=True` | Skip training and run validation only |
| `--VISUALIZE=True` | Save output images during validation |
| `--RESUME_PATH=<path>` | Path to the trained checkpoint |
| `--VAL_BATCH_SIZE=4` | Increase batch size for faster inference if memory allows |

To evaluate a visualization folder, run:

```bash
python tools/2-eval-for-vis-folder/SEE_eval_for_model.py \
    --root="/path/to/your/vis/folder" \
    --log_dir="/path/to/your/log/dir" \
    --alsologtostderr=True
```

To evaluate with an explicit ground-truth root, run:

```bash
python tools/2-eval-for-vis-folder/SEE_eval_for_model_with_gt_root.py \
    --root="/path/to/your/eval/folder" \
    --gt_root="/path/to/your/gt/root" \
    --log_dir="/path/to/your/log/dir" \
    --alsologtostderr=True
```

Baseline metric summaries are available in [TEST_RESULTS.md](./TEST_RESULTS.md) and [TEST_MINI_RESULTS.md](./TEST_MINI_RESULTS.md).

---

## 7. Use Exposure Prompts for Brightness Adjustment

SEE-Net supports continuous brightness control through an exposure prompt **B**:

- **B ≈ 0.5**: normal exposure and the typical operating range;
- **B > 0.5**: brighter output;
- **B < 0.5**: darker output.

The model applies the prompt during decoding:

```python
exposure_prompt = torch.ones(size=(B, 1, 1, 1)) * exposure_B
exposure_prompt = exposure_prompt.to(images.device)
exposure_normal_reconstructed = self._decoding(light_inr, exposure_prompt)
```

During testing, the model generates three types of output:

- **PRD**: prediction using the target exposure prompt;
- **SLR**: self-supervised reconstruction using the input mean;
- **NLR**: standard-light reconstruction using fixed `B=0.4`.

Extreme values of **B** are mainly intended for analysis and visualization rather than as target outputs. The framework is designed for brightness adjustment rather than full HDR reconstruction.

---

## 8. FAQ

### How much GPU memory is needed?

The baseline SEENet model has approximately **1.9M parameters**. With batch size 2 and crop size 128×256, it requires approximately **6-8 GB** of GPU memory. Adjust `TRAIN_BATCH_SIZE` and crop size based on your hardware.

### Can I use external data?

Yes. External training data, pretrained models, and synthetic data are allowed, but they must be **clearly reported** in the final technical report.

### How long does training take?

With a single A100 GPU, baseline training for 20 epochs takes approximately **10-15 hours**. Training time varies by GPU.

### Why is TensorBoard empty?

Check the following:

1. `--logdir` points to the directory that contains `events.out.tfevents.*` files.
2. Training has started and logged at least one epoch; check `LOG_INTERVAL` in the config.
3. If using a remote server, SSH port forwarding is configured correctly.

---

## 9. Citation

If you use the SEE-600K dataset or SEE-Net, please cite:

```bibtex
@article{lu2025SEE,
  title={SEE: See Everything Every Time - Adaptive Brightness Adjustment for Broad Light Range Images via Events},
  author={Yunfan Lu, Xiaogang Xu, Hao Lu, Yanlin Qian, Pengteng Li, Huizai Yao, Bin Yang, Junyi Li, Qianyi Cai, Weiyu Guo, Hui Xiong},
  year={2025},
}
```

---

## 10. Contact

- **Yunfan Lu**: <ylu066@connect.hkust-gz.edu.cn>
- **GitHub**: [yunfanLu](https://github.com/yunfanLu)
