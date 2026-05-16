# Test Results on SEE Dataset

This document summarizes baseline test results on the SEE dataset.

## Overall Results

| Model | PSNR | PSNR-Linear_N | SSIM |
| --- | ---: | ---: | ---: |
| eSL | 12.497933116976917 | 14.678264411503823 | 0.4590008483887293 |
| EIFT | 13.793279392580926 | 18.140406656151995 | 0.5255946642388964 |
| EvLight | 14.778939572681521 | 20.437348261096336 | 0.5589390925097292 |
| SEENet | 18.827914236034424 | 19.15370335625842 | 0.6413754544794725 |

## Results by Lighting Condition

### 1. eSL

| Condition | PSNR | SSIM | PSNR-Linear_N | L1 |
| --- | ---: | ---: | ---: | ---: |
| Low-Normal | 11.71686303696365 | 0.38487610627512764 | 12.709430027276838 | 0.24746266433433184 |
| High-Normal | 12.684757618040875 | 0.47192720484373896 | 17.228681301248486 | 0.2086358498460774 |
| Normal-Normal | 13.329043261117713 | 0.5699684469679067 | 16.056801360152487 | 0.19423581311522528 |

### 2. EvLight

| Condition | PSNR | SSIM | PSNR-Linear_N | L1 |
| --- | ---: | ---: | ---: | ---: |
| Low-Normal | 13.70890295421383 | 0.5149924498364116 | 18.4315754091346 | 0.19609124236910377 |
| High-Normal | 13.457695973430436 | 0.4918908255242249 | 19.53202949158133 | 0.19902556211289663 |
| Normal-Normal | 13.637796657133238 | 0.5924541240694041 | 18.542610409209697 | 0.20045347673571184 |

### 3. EIFT

| Condition | PSNR | SSIM | PSNR-Linear_N | L1 |
| --- | ---: | ---: | ---: | ---: |
| Low-Normal | 13.489086465148022 | 0.5068513168834802 | 17.309244948008033 | 0.1946338162979781 |
| High-Normal | 12.30855265234721 | 0.4767232469373172 | 18.65585968086691 | 0.2221921297367226 |
| Normal-Normal | 13.706325046170178 | 0.5475036669206746 | 17.095937220482895 | 0.21513682325284328 |

### 4. SEENet

| Condition | PSNR | SSIM | PSNR-Linear_N | L1 |
| --- | ---: | ---: | ---: | ---: |
| Low-Normal | 16.963360673031247 | 0.6132844679897682 | 17.403425003376114 | 0.11817919201483273 |
| High-Normal | 18.303280793387316 | 0.6427289049044765 | 18.504418790340424 | 0.09054435048942808 |
| Normal-Normal | 20.109399302061213 | 0.7873336093370304 | 20.286927988362866 | 0.06820999052440531 |

## Notes

- Overall metrics are reported as PSNR, PSNR-Linear_N, and SSIM.
- Per-lighting-condition results are reported as `(PSNR, SSIM, PSNR-Linear_N, L1)`.
- Among the tested baselines, SEENet achieves the best overall performance on the SEE dataset.

## How to Run Evaluation

To evaluate your own model:

```bash
python tools/2-eval-for-vis-folder/SEE_eval_for_model.py \
    --root="/path/to/your/vis/folder" \
    --log_dir="/path/to/your/log/dir" \
    --alsologtostderr=True
```

For more details, see [TUTORIAL.md](./TUTORIAL.md).
