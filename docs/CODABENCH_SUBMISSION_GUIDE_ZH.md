# CodaBench 提交指南

本指南说明如何准备 SEE 预测结果并提交到 CodaBench。环境配置、模型训练、评测和推理请见 [TUTORIAL\_ZH.md](./TUTORIAL_ZH.md)。

***

## 1. 生成推理结果

使用训练好的模型运行推理并保存可视化结果。输出目录通常命名为 `vis`；它包含推理阶段保存的所有图像。具体推理命令和测试参数请见 [TUTORIAL\_ZH.md](./TUTORIAL_ZH.md#6-评测或运行推理)。

在生成提交预测结果之前，请确保对应 YAML 文件中的推理裁剪尺寸设置为 SEE 原始图像大小：

```yaml
crop_w: 346
crop_h: 260
```

这对应于：

- 宽度：`346`
- 高度：`260`

如果推理裁剪尺寸没有设置为原始图像大小，输出可能不符合比赛测试集要求。

***

## 2. 收集比赛预测图像

使用收集脚本抽取与比赛测试集 ground-truth 样本对应的预测图像：

```bash
python codabench/collect_codabench_pred.py ${prediction_vis_directory} ${output_directory}
```

示例：

```bash
python codabench/collect_codabench_pred.py /path/to/vis /path/to/output_dir
```

参数说明：

- `${prediction_vis_directory}`：模型推理后生成的 `vis` 文件夹。该文件夹保存推理输出的所有图像；生成方法请见 [TUTORIAL\_ZH.md](./TUTORIAL_ZH.md#6-评测或运行推理)；
- `${output_directory}`：用于保存收集后 CodaBench 提交结果的文件夹。

脚本默认使用列表 `codabench/SEE_gt_mini.txt`。如果要使用其他列表，请传入 `--list`：

```bash
python codabench/collect_codabench_pred.py /path/to/vis /path/to/output_dir --list /path/to/list.txt
```

如果输出目录已经存在且需要替换，请传入 `--overwrite`：

```bash
python codabench/collect_codabench_pred.py /path/to/vis /path/to/output_dir --overwrite
```

脚本执行完成后，输出目录中会包含整理好的提交结果。

***

## 3. Eval Phase 测试提交流程

Eval Phase 使用最终测试数据。提交前请先更新项目代码，并参考 `options/SEE/SEENet_SEE_replicate.yaml` 修改测试配置，尤其需要确认：

```yaml
DATASET:
  root: /path/to/SEE-600K-eval/DVS346-eval
  crop_h: 260
  crop_w: 346
  all_groups_as_testing: true
```

`all_groups_as_testing: true` 表示将测试数据目录下的所有 group 都作为测试集，避免由于 group 名称不在训练代码内置 testing split 中而导致测试样本数为 0。

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

参数说明：

- `${prediction_vis_directory}`：测试脚本生成的 `vis` 文件夹；
- `${output_directory}`：收集后的提交目录。

最后按照下一节打包，并上传到 CodaBench 的 Eval Phase 页面。

***

## 4. 打包提交文件

将收集后的输出目录打包为 zip 文件：

```bash
cd /path/to/output_dir
zip -r submission.zip ./* 
```

上传前，请检查：

- 输出文件数量是否正确；
- 文件名格式是否符合比赛要求；
- zip 文件内部目录结构是否正确；
- 图像是否保存为 SEE 原始图像大小。

***

## 5. 上传到 CodaBench

1. 打开 CodaBench 比赛页面：<https://www.codabench.org/competitions/16195/>。
2. 进入 **submission** 页面。
3. 上传 `submission.zip`。
4. 提交文件。

如果 CodaBench 提交状态异常，请先检查 zip 文件内部目录结构和文件命名。

***

## 6. 流程总结

```bash
# 1. 收集用于提交的预测图像
python codabench/collect_codabench_pred.py ${prediction_vis_directory} ${output_directory}

# Eval Phase 使用专用收集脚本
python codabench/collect_eval_phase_pred.py ${prediction_vis_directory} ${output_directory}

# 2. 打包收集后的文件夹
cd ${output_directory}
zip -r submission.zip ./* 

# 3. 在 CodaBench 上传 submission.zip
```
