#!/bin/bash

#SBATCH -N 1 # 指定node的数量
#SBATCH -p i64m1tga800u #  i64m1tga40u, i64m1tga800u
#SBATCH --gres=gpu:1 # 需要使用多少GPU，n是需要的数量
#SBATCH --time=7-00:00:00 # 指定任务运⾏的上限时间（我们的HPC为7天
#SBATCH --job-name=SEENet_SEE-ar1-96-4-8-4-loss-2-epoch-20
#SBATCH --output=./logs/SeeEverythingEverytime/SEENet/Ablation-Release/SEENet_SEE-ar1-96-4-8-4-loss-2-epoch-20/%j-job.out # slurm的输出文件，%j是jobid
#SBATCH --error=./logs/SeeEverythingEverytime/SEENet/Ablation-Release/SEENet_SEE-ar1-96-4-8-4-loss-2-epoch-20/%j-job.err # 指定错误输出的格式
#SBATCH --cpus-per-task=1

export CUDA_VISIBLE_DEVICES="0,1"
export PATH="/hpc2hdd/home/ylu066/miniconda3/bin":$PATH
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"
echo "Python version: $(python --version)"
echo "              : $(which python)"
echo "CUDA version: $(nvcc --version)"
nvidia-smi



python see/main.py \
  --yaml_file="options/SEE/SEENet_SEE.yaml" \
  --log_dir="./logs/SEE/SEENet_SEE/" \
  --alsologtostderr=True