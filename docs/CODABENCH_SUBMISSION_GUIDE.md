# CodaBench Submission Guide

This guide explains how to prepare SEE prediction results and submit them to CodaBench. For environment setup, model training, evaluation, and inference, see [TUTORIAL.md](./TUTORIAL.md).

---

## 1. Generate Inference Results

Run inference with your trained model and save visualization results. The output directory is usually named `vis`.

Before generating submission predictions, make sure the inference crop size in the corresponding YAML file is set to the original SEE image size:

```yaml
crop_w: 346
crop_h: 260
```

This corresponds to:

- Width: `346`
- Height: `260`

If the inference crop size is not set to the original image size, the output may not match the competition test-set requirements.

---

## 2. Collect Competition Prediction Images

Use the collection script to extract the prediction images that correspond to the competition test-set ground-truth samples:

```bash
python codabench/collect_codabench_pred.py ${prediction_vis_directory} ${output_directory}
```

Example:

```bash
python codabench/collect_codabench_pred.py /path/to/vis /path/to/output_dir
```

Arguments:

- `${prediction_vis_directory}`: the `vis` folder generated after model inference;
- `${output_directory}`: the folder used to save the collected prediction results for CodaBench submission.

The script uses the default list `codabench/SEE_gt_mini.txt`. To use another list, pass `--list`:

```bash
python codabench/collect_codabench_pred.py /path/to/vis /path/to/output_dir --list /path/to/list.txt
```

If the output directory already exists and you want to replace it, pass `--overwrite`:

```bash
python codabench/collect_codabench_pred.py /path/to/vis /path/to/output_dir --overwrite
```

After the script finishes, the output directory contains the organized prediction results for submission.

---

## 3. Package the Submission

Create a zip file from the collected output directory:

```bash
zip -r submission.zip ${output_directory}
```

Example:

```bash
zip -r submission.zip /path/to/output_dir
```

Before uploading, check that:

- the number of output files is correct;
- the filename format matches the expected competition format;
- the directory structure inside the zip file is correct;
- the images are saved at the original SEE image size.

---

## 4. Upload to CodaBench

1. Open the CodaBench competition page: <https://www.codabench.org/competitions/16195/>.
2. Go to the **submission** page.
3. Upload `submission.zip`.
4. Submit the file.

If the CodaBench submission status is abnormal, first check the zip file's internal directory structure and file naming.

---

## 5. Workflow Summary

```bash
# 1. Collect prediction images for submission
python codabench/collect_codabench_pred.py ${prediction_vis_directory} ${output_directory}

# 2. Package the collected folder
zip -r submission.zip ${output_directory}

# 3. Upload submission.zip on CodaBench
```
