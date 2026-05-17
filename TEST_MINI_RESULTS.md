# Test Results on SEE Mini GT Dataset

This document summarizes baseline test results on the SEE mini GT dataset.

## Overall Results

| Model | Count | PSNR | PSNR-Linear_N | SSIM |
| --- | ---: | ---: | ---: | ---: |
| eSL | 537 | 12.142816955150838 | 14.328414840127724 | 0.45363404060210133 |
| EIFT | 537 | 13.269945744029636 | 17.10539889624221 | 0.508263058797413 |
| EvLight | 537 | 13.245802241774689 | 18.101232314242996 | 0.5166970032164948 |
| SEENet | 537 | 17.841153858760215 | 18.173663222811964 | 0.6576141682028549 |

## Results by Lighting Condition

### 1. eSL

| Condition | Count | PSNR | SSIM | PSNR-Linear_N | L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low-Normal | 285 | 11.55674770003871 | 0.39312689436738496 | 12.900011945084522 | 0.24832917113314595 |
| High-Normal | 90 | 12.36504225730896 | 0.4638807381192843 | 16.70895741780599 | 0.21702626355820231 |
| Normal-Normal | 162 | 13.050406217575073 | 0.5543891881351117 | 15.518822204919509 | 0.19671123964643036 |

### 2. EIFT

| Condition | Count | PSNR | SSIM | PSNR-Linear_N | L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low-Normal | 285 | 13.48764405334205 | 0.5022773237604844 | 16.915871705506976 | 0.19095519332770716 |
| High-Normal | 90 | 12.124863004684448 | 0.47061427721960675 | 18.376021263334486 | 0.23017980994449722 |
| Normal-Normal | 162 | 13.52311505506068 | 0.53970950835005 | 16.732925046373296 | 0.21958736354416167 |

### 3. EvLight

| Condition | Count | PSNR | SSIM | PSNR-Linear_N | L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low-Normal | 285 | 13.469529933260198 | 0.5044110481153454 | 17.87595701803241 | 0.19867764842091945 |
| High-Normal | 90 | 13.03145105573866 | 0.476916192099452 | 19.214617029825845 | 0.21189862017830213 |
| Normal-Normal | 162 | 12.971291221218344 | 0.5604116341076147 | 17.87900290077115 | 0.21628154033542046 |

### 4. SEENet

| Condition | Count | PSNR | SSIM | PSNR-Linear_N | L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low-Normal | 285 | 16.70454464962608 | 0.6044091157223049 | 17.171287319116423 | 0.120221282685535 |
| High-Normal | 90 | 17.67813867992825 | 0.6236959202422037 | 17.893183824751112 | 0.09910740231474241 |
| Normal-Normal | 162 | 19.93130812232877 | 0.7700591205078878 | 20.09292420045829 | 0.07185170612079494 |

## Notes

- Overall metrics are reported as Count, PSNR, PSNR-Linear_N, and SSIM.
- Per-lighting-condition results are reported as `Count + (PSNR, SSIM, PSNR-Linear_N, L1)`.
- The SEE mini GT dataset used here contains 537 samples in total:
  - Low-Normal: 285
  - High-Normal: 90
  - Normal-Normal: 162
- Among the tested baselines, SEENet achieves the best overall performance on the SEE mini GT dataset.

## How to Run Evaluation

To evaluate your own model with an explicit GT root:

```bash
python tools/2-eval-for-vis-folder/SEE_eval_for_model_with_gt_root.py \
    --root="/path/to/your/eval/folder" \
    --gt_root="/path/to/your/gt/root" \
    --log_dir="/path/to/your/log/dir" \
    --alsologtostderr=True
```

For more details, see [TUTORIAL.md](./TUTORIAL.md).
