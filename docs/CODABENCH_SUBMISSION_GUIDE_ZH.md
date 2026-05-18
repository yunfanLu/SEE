# Codabench 测试提交流程中文说明

本文档用于说明 SEE 项目中，从模型训练、推理测试，到整理比赛测试集预测结果并提交到 Codabench 的完整流程。

---

## 1. 模型训练

模型训练统一通过 `options/SEE/` 目录下对应模型的 shell 脚本启动。

以 **SEENet** 为例，训练命令为：

```bash
sh options/SEE/SEENet_SEE.sh
```

说明：
- 不同模型通常都有各自对应的脚本；
- 如果需要训练其他模型，只需要替换成对应的 `.sh` 文件即可；
- 训练相关配置通常写在该脚本所引用的 `.yaml` 文件中。

---

## 2. 模型推理 / Inference

模型训练完成后，进行测试推理时，仍然调用对应模型脚本中的**测试部分**。

以 **SEENet** 为例，测试代码同样参考：

```bash
sh options/SEE/SEENet_SEE.sh
```

需要注意：

- 推理时请使用该脚本中的 **Test / 测试部分**；
- 推理前需要检查对应的 YAML 配置；
- **inference 时，yaml 中的 crop 图像宽高必须设置为原图大小**。

对于 SEE 数据集，这里需要设置为：

```yaml
crop_w: 346
crop_h: 260
```

也就是原图大小：

- 宽 `346`
- 高 `260`

如果推理时 crop 大小不是原图大小，可能会导致输出结果与比赛测试集要求不一致。

---

## 3. 从可视化结果中抽取比赛测试集对应预测图

完成推理后，通常会得到模型的可视化结果目录（`vis` 文件夹）。

接下来，需要从这些测试结果中，抽取出**本次比赛 test 集对应 GT 的预测图像**。

这里使用的脚本是：

```bash
codabench/collect_codabench_pred.py
```

使用方式：

```bash
python codabench/collect_codabench_pred.py ${predict的vis目录} ${输出文件夹路径}
```

例如：

```bash
python codabench/collect_codabench_pred.py /path/to/vis /path/to/output_dir
```

参数说明：

- `${predict的vis目录}`：模型推理后生成的 `vis` 文件夹；
- `${输出文件夹路径}`：用于保存抽取后的比赛提交结果。

执行完成后，输出文件夹中会保存整理好的、可用于比赛提交的预测结果。

---

## 4. 打包并提交到 Codabench

当第 3 步得到输出文件夹后，需要将该文件夹打包为 zip 文件，然后上传到 Codabench 的 submission 页面。

可以使用如下命令进行打包：

```bash
zip -r submission.zip ${输出文件夹路径}
```

例如：

```bash
zip -r submission.zip /path/to/output_dir
```

完成后：

1. 打开 Codabench 比赛页面；
2. 进入 **submission** 页面；
3. 上传打包好的 `submission.zip`；
4. 提交即可。

---

## 5. 流程总结

完整流程如下：

1. 训练模型
   ```bash
   sh options/SEE/SEENet_SEE.sh
   ```
2. 使用脚本中的测试部分进行 inference；
3. 确保 YAML 中推理时的图像大小设置为原图大小：
   ```yaml
   crop_w: 346
   crop_h: 260
   ```
4. 使用如下命令抽取比赛提交所需预测结果：
   ```bash
   python codabench/collect_codabench_pred.py ${predict的vis目录} ${输出文件夹路径}
   ```
5. 将输出文件夹打包为 zip：
   ```bash
   zip -r submission.zip ${输出文件夹路径}
   ```
6. 将 zip 上传到 Codabench submission 页面。

---

## 6. 备注

- 训练和测试都建议优先查看对应模型脚本中的注释；
- 提交前建议先检查输出文件数量、命名格式以及目录结构是否符合比赛要求；
- 如果 Codabench 提交后状态异常，可优先检查 zip 内部目录结构和文件命名是否正确。
