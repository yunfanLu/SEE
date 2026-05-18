# Codabench Test Submission Workflow

This document explains the complete workflow for the SEE project, including model training, inference/testing, organizing prediction results for the competition test set, and submitting them to Codabench.

---

## 1. Model Training

Model training is launched through the corresponding shell script under the `options/SEE/` directory.

Taking **SEENet** as an example, the training command is:

```bash
sh options/SEE/SEENet_SEE.sh
````

Notes:

* Different models usually have their own corresponding scripts;
* To train another model, simply replace it with the corresponding `.sh` file;
* Training-related configurations are usually defined in the `.yaml` file referenced by the script.


## 2. Model Inference / Testing

After model training is completed, testing/inference is still performed using the **testing part** of the corresponding model script.

Taking **SEENet** as an example, the testing command also refers to:

```bash
sh options/SEE/SEENet_SEE.sh
```


Notes:

- During inference, use the **Test / testing section** in the script;
- Before inference, check the corresponding YAML configuration;
- **During inference, the crop width and height in the YAML file must be set to the original image size**.

For the SEE dataset, they should be set as:

crop_w: 346
crop_h: 260

This corresponds to the original image size:

* Width: `346`
* Height: `260`

If the crop size during inference is not set to the original image size, the output results may not meet the requirements of the competition test set.


## 3. Collect Prediction Images Corresponding to the Competition Test Set

After inference is completed, a visualization result directory, usually named `vis`, will be generated.

Next, you need to extract the prediction images corresponding to the **GT samples of the competition test set** from these testing results.

The script used for this step is:

```bash
codabench/collect_codabench_pred.py
```

Usage:

```bash
python codabench/collect_codabench_pred.py ${prediction_vis_directory} ${output_directory}
```

For example:

```bash
python codabench/collect_codabench_pred.py /path/to/vis /path/to/output_dir
```

Arguments:

* `${prediction_vis_directory}`: the `vis` folder generated after model inference;
* `${output_directory}`: the folder used to save the collected prediction results for competition submission.

After the script finishes, the output directory will contain the organized prediction results ready for competition submission.

---

## 4. Package and Submit to Codabench

After obtaining the output directory from Step 3, you need to package this folder into a zip file and upload it to the submission page on Codabench.

You can use the following command to create the zip file:

```bash
zip -r submission.zip ${output_directory}
```

For example:

```bash
zip -r submission.zip /path/to/output_dir
```

After that:

1. Open the Codabench competition page;
2. Go to the **submission** page;
3. Upload the packaged `submission.zip`;
4. Submit it.

---

## 5. Workflow Summary

The complete workflow is as follows:

1. Train the model:

   ```bash
   sh options/SEE/SEENet_SEE.sh
   ```

2. Use the testing section in the script for inference;

3. Make sure the image size during inference is set to the original size in the YAML file:

   ```yaml
   crop_w: 346
   crop_h: 260
   ```

4. Use the following command to collect the prediction results required for competition submission:

   ```bash
   python codabench/collect_codabench_pred.py ${prediction_vis_directory} ${output_directory}
   ```

5. Package the output folder into a zip file:

   ```bash
   zip -r submission.zip ${output_directory}
   ```

6. Upload the zip file to the Codabench submission page.

---

## 6. Notes

* It is recommended to first check the comments in the corresponding model script before training and testing;
* Before submission, it is recommended to check whether the number of output files, filename format, and directory structure meet the competition requirements;
* If the Codabench submission status is abnormal, first check whether the internal directory structure of the zip file and the file naming are correct.
