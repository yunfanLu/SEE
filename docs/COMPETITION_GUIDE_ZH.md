# SEE Challenge 2026 参赛完整指南

> **比赛主页**：https://www.codabench.org/competitions/16195/
> **GitHub 仓库**：https://github.com/yunfanLu/SEE
> **Workshop**：EBMV @ ECCV 2026

---

## 一、比赛概述

### 1.1 任务定义

SEE Challenge 2026 是 ECCV 2026 Workshop **EBMV（Event-Based Multimodal Vision）** 的配套竞赛，聚焦于**宽光照条件下的事件引导亮度调整**。

真实世界中，相机在低光、过曝、混合光照、高对比度等场景下常常失效。事件相机（Event Camera）具有高时间分辨率和高动态范围，能在困难光照下捕获有效的结构和运动信息。

**输入：**
- 一张在复杂光照下拍摄的 RGB 图像
- 对应的事件流或事件表示（Event Voxel）
- 可选的亮度提示值 B（brightness prompt）

**输出：**
- 一张亮度调整后的 RGB 图像，要求：
  - 保留场景结构和细节
  - 自然的色彩表现
  - 避免幻觉（hallucination）内容

### 1.2 评估指标

| 指标 | 含义 | 权重 |
|------|------|------|
| **PSNR** | 像素级重建精度 | **主要指标，越高越好** |
| **SSIM** | 结构相似性 | 次要指标，越高越好 |

最终排名以 **PSNR** 为主要依据。

### 1.3 重要时间节点

| 日期 | 事件 |
|------|------|
| 2026年5月10日 | 比赛网站开放 |
| 2026年5月25日 | 验证服务器上线 |
| 2026年6月25日 | 测试数据发布，测试服务器上线 |
| 2026年7月3日 | 测试提交截止 |
| 2026年7月10日 | 结果公布（暂定） |

### 1.4 奖项

顶级团队将在 ECCV 2026 Workshop 上获得表彰，设有：
- 第一、二、三名
- 最佳学生团队
- 最佳技术报告

---

## 二、数据集：SEE-600K

### 2.1 数据集概况

| 属性 | 数值 |
|------|------|
| 图像总数 | 610,126 张 |
| 真实场景数 | 202 个 |
| 光照类型 | 低光、正常光、高光、混合光 |
| 数据格式 | RGB 图像 + 事件 Voxel（.npy）+ 事件可视化（.png）|
| 图像分辨率 | 346 × 260 |

**下载地址：**
- Hugging Face：https://huggingface.co/datasets/yunfanlu/SEE-600K
- OneDrive：见 GitHub README

### 2.2 数据集目录结构

```
SEE-600K/
└── RoboticArm/
    ├── 000-indoor_ceiling_table_light/     ← 场景组（含光照描述）
    │   ├── registrate_result.json          ← 配准结果
    │   ├── exposure_state.json             ← 曝光状态
    │   └── video_name_1/                   ← 视频序列
    │       ├── frame_event/
    │       │   ├── <timestamp>_...png      ← RGB 帧
    │       │   ├── <ts_start>_<ts_end>.npy ← 事件 Voxel
    │       │   └── <ts_start>_<ts_end>_vis.png  ← 事件可视化
    │       └── imu.npy
    └── ...
```

### 2.3 训练/测试划分（重要）

以下场景组**不得用于训练**，为官方测试集：

- **室内（Indoor）**：000, 001, 002, 006, 012, 018, 030, 042, 048, 054, 060, 065, 070, 074, 075
- **室外（Outdoor）**：100, 106, 112, 118, 124, 130, 136, 142, 148, 154, 160, 166, 173, 184, 189, 194, 200, 206, 212, 217, 222, 225

### 2.4 下载数据集

```bash
pip install huggingface_hub

# 下载全量数据集
hf download yunfanlu/SEE-600K --local-dir ./SEE-600K

# 只下载部分场景（推荐先试用）
hf download yunfanlu/SEE-600K --include "RoboticArm/*" --local-dir ./SEE-600K
```

---

## 三、环境配置

### 3.1 克隆仓库

```bash
git clone https://github.com/yunfanLu/SEE.git
cd SEE
```

### 3.2 创建 Python 环境

```bash
conda create -n see python=3.10 -y
conda activate see
pip install -r requirements.txt
```

主要依赖：PyTorch、OpenCV、NumPy、timm、einops、kornia、tensorboard、scipy、numba

### 3.3 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 四、基线模型：SEE-Net

### 4.1 模型简介

SEE-Net 是官方提供的基线模型，约 **1.9M 参数**，接受 RGB 图像、事件 Voxel 和亮度提示 B 作为输入，输出亮度调整后的 RGB 图像。

**亮度提示 B 的含义：**
- B ≈ 0.5：正常曝光
- B > 0.5：输出更亮
- B < 0.5：输出更暗

**测试阶段输出类型：**
| 前缀 | 含义 |
|------|------|
| `_p0_`（PRD） | 使用目标曝光提示的预测结果 |
| `_s0_`（SLR） | 自监督重建（使用输入均值） |
| `_n0_`（NLR） | 标准光照重建（固定 B=0.4） |

### 4.2 预训练模型下载

Google Drive：https://drive.google.com/drive/folders/1SWR9YVIrqFEkGGKrv3wSC5RqmMIEYjV2

包含：EIFT_AAAI_SEE、eSl_SEE、EvLight、SEENet_SEE 的权重及训练日志。

---

## 五、训练

### 5.1 配置数据集路径

编辑 `options/SEE/SEENet_SEE.yaml`：

```yaml
DATASET:
  root: /path/to/your/SEE-600K/RoboticArm/
```

默认训练配置：
- batch size：2
- epoch：20
- 学习率：0.0001
- 裁剪尺寸：128 × 256
- 损失函数：L1-Charbonnier + Gradient

### 5.2 启动训练

```bash
export PYTHONPATH="./":$PYTHONPATH
python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True
```

### 5.3 多 GPU 训练

```bash
export CUDA_VISIBLE_DEVICES="0,1"
python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True
```

### 5.4 从检查点恢复训练

```bash
python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True \
  --RESUME_PATH="./logs/SEE/SEENet_SEE/checkpoint.pth.tar" \
  --RESUME_SET_EPOCH=True
```

**训练时长参考**：单张 A100，20 epoch 约 10–15 小时。显存需求：batch size=2，裁剪 128×256 时约需 6–8 GB。

---

## 六、推理与评估

### 6.1 运行推理（生成 vis 文件夹）

**注意**：提交前必须将裁剪尺寸改为原始图像大小：

```yaml
# options/SEE/SEENet_SEE.yaml
crop_w: 346
crop_h: 260
```

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

推理结果保存在 `--log_dir` 指定的 `vis` 文件夹中。

### 6.2 本地评估

```bash
python tools/2-eval-for-vis-folder/SEE_eval_for_model.py \
    --root="/path/to/your/vis/folder" \
    --log_dir="/path/to/your/log/dir"
```

---

## 七、提交到 CodaBench

### 7.1 完整提交流程

```
训练模型 → 运行推理（crop 346×260）→ 收集预测图像 → 打包 zip → 上传 CodaBench
```

### 7.2 第一步：收集预测图像

```bash
python codabench/collect_codabench_pred.py \
    ${prediction_vis_directory} \
    ${output_directory}
```

- `${prediction_vis_directory}`：推理生成的 `vis` 文件夹路径
- `${output_directory}`：收集后的输出目录

可选参数：
```bash
# 使用自定义列表文件
python codabench/collect_codabench_pred.py /path/to/vis /path/to/output --list /path/to/list.txt

# 覆盖已有输出目录
python codabench/collect_codabench_pred.py /path/to/vis /path/to/output --overwrite
```

默认使用 `codabench/SEE_gt_mini.txt` 作为文件列表。

### 7.3 第二步：打包 zip

```bash
cd ${output_directory}
zip -r submission.zip ./*
```

**上传前检查清单：**
- [ ] 文件数量正确
- [ ] 文件名格式符合要求
- [ ] zip 内部目录结构正确
- [ ] 图像分辨率为原始尺寸（346 × 260）
- [ ] 图像格式为 `.png`

### 7.4 第三步：上传到 CodaBench

1. 打开 https://www.codabench.org/competitions/16195/
2. 登录账号，进入 **My Submissions** 页面
3. 点击上传，选择 `submission.zip`
4. 等待评测结果（验证集结果通常几分钟内返回）

### 7.5 Eval Phase 测试提交流程

测试数据集: https://drive.google.com/file/d/1-qQ_rKt5nB_8bb0kCZGhW_Epqkpi_I34/view?usp=drive_link

Eval Phase 使用最终测试数据。提交前请先更新项目代码，并参考
`options/SEE/SEENet_SEE_replicate.yaml` 修改测试配置，尤其需要确认：

```yaml
DATASET:
  root: /path/to/SEE-600K-eval/DVS346-eval
  crop_h: 260
  crop_w: 346
  all_groups_as_testing: true
```

`all_groups_as_testing: true` 表示将测试数据目录下的所有 group 都作为测试集，
避免由于 group 名称不在训练代码内置 testing split 中而导致测试样本数为 0。

运行测试脚本生成 `vis` 目录：

```bash
python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE_replicate.yaml" \
  --log_dir="${eval_log_directory}" \
  --alsologtostderr=True \
  --RESUME_PATH="${checkpoint_path}" \
  --TEST_ONLY=True \
  --VISUALIZE=True \
  --VAL_BATCH_SIZE=1
```

测试完成后，从生成的 `vis` 文件夹中收集需要提交的预测图像：

```bash
python codabench/collect_eval_phase_pred.py \
  ${prediction_vis_directory} \
  ${output_directory}
```

- `${prediction_vis_directory}`：测试脚本生成的 `vis` 文件夹
- `${output_directory}`：收集后的提交目录

最后打包并上传：

```bash
cd ${output_directory}
zip -r submission.zip ./*
```

将 `submission.zip` 上传到 CodaBench 的 Eval Phase 页面即可。

---

## 八、比赛规则

1. 可使用官方 SEE-600K 训练和验证数据
2. **允许**使用外部训练数据、预训练模型、合成数据，但必须在技术报告中明确说明
3. **禁止**在隐藏测试集上手动调参
4. **严禁**使用隐藏测试集的真值图像
5. 禁止利用评估服务器或提交系统漏洞
6. 排名靠前的团队需提交简短技术报告
7. 组织者保留验证顶级提交可重现性的权利

---

## 九、常见问题

**Q：验证集和测试集有什么区别？**
A：验证服务器（5月25日上线）主要用于开发调试和格式检查；测试服务器（6月25日上线）才是最终排名依据。

**Q：可以使用外部数据吗？**
A：可以，但必须在技术报告中说明数据来源和使用方式。

**Q：提交频率有限制吗？**
A：CodaBench 通常有每日提交次数限制，具体以平台显示为准。

**Q：crop 尺寸为什么要改成 346×260？**
A：训练时为节省显存使用了较小的裁剪尺寸（128×256），但提交的预测图像必须与参考图像分辨率一致（346×260），否则评测会失败。

**Q：提交的是哪一路输出（PRD/SLR/NLR）？**
A：提交 `_p0_`（PRD，使用目标曝光提示的预测结果），这是与 ground truth 对齐的主要输出。

---

## 十、相关链接

| 资源 | 链接 |
|------|------|
| 比赛主页 | https://www.codabench.org/competitions/16195/ |
| 比赛论坛 | https://www.codabench.org/forums/15944/ |
| GitHub 仓库 | https://github.com/yunfanLu/SEE |
| 数据集（HuggingFace） | https://huggingface.co/datasets/yunfanlu/SEE-600K |
| 预训练模型 | https://drive.google.com/drive/folders/1SWR9YVIrqFEkGGKrv3wSC5RqmMIEYjV2 |
| Workshop 官网 | https://eventbasemultimodalvision.github.io |
| 技术报告（arXiv） | https://arxiv.org/abs/2502.21120 |

---

## 十一、引用

```bibtex
@article{lu2025SEE,
  title={SEE: See Everything Every Time - Adaptive Brightness Adjustment for Broad Light Range Images via Events},
  author={Yunfan Lu, Xiaogang Xu, Hao Lu, Yanlin Qian, Pengteng Li, Huizai Yao, Bin Yang, Junyi Li, Qianyi Cai, Weiyu Guo, Hui Xiong},
  year={2025},
}
```

---

## 十二、联系方式

- **Yunfan Lu**（主要联系人）：ylu066@connect.hkust-gz.edu.cn
- **Mingchao Xu**：mingchao.xu.casia@gmail.com
- **GitHub**：https://github.com/yunfanLu
