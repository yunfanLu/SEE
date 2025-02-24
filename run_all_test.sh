#!/bin/bash

#SBATCH -N 1 # 指定node的数量
#SBATCH -p i64m512r #  i64m1tga40u, i64m1tga800u
#SBATCH --gres=gpu:1 # 需要使用多少GPU，n是需要的数量
#SBATCH --time=7-00:00:00 # 指定任务运⾏的上限时间（我们的HPC为7天
#SBATCH --job-name=DCE_SEE-v1
#SBATCH --output=./logs/SeeEverythingEverytime/DeepCurveEstimation/DCE_SEE-v1/%j-job.out # slurm的输出文件，%j是jobid
#SBATCH --error=./logs/SeeEverythingEverytime/DeepCurveEstimation/DCE_SEE-v1/%j-job.err # 指定错误输出的格式
#SBATCH --cpus-per-task=1

# echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
# echo "Job run at:"
# echo "$(hostnamectl)"
# echo "Python version: $(python --version)"
# echo "              : $(which python)"
# echo "CUDA version: $(nvcc --version)"
# nvidia-smi


export CUDA_VISIBLE_DEVICES="2"
export PATH="/hpc2hdd/home/ylu066/miniconda3/bin/":$PATH
export PYTHONPATH="./":$PYTHONPATH


# python tools/2-paper-figure/2-eval-for-vis-folder/make_SDE_results_to_visualization.py \
#     --make_csv=False \
#     --choise_best_png=True \
#     --make_video=False &

# python tools/2-paper-figure/2-eval-for-vis-folder/make_SDE_results_to_visualization.py \
#     --make_csv=False \
#     --choise_best_png=False \
#     --make_video=True &

python tools/2-paper-figure/2-eval-for-vis-folder/make_SEE_results_to_visualization.py \
    --make_csv=True \
    --choise_best_png=True \
    --make_video=True