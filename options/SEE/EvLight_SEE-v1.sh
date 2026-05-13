#!/bin/bash

#SBATCH -N 1 # 指定node的数量
#SBATCH -p i64m1tga40u #  i64m1tga40u, i64m1tga800u
#SBATCH --gres=gpu:1 # 需要使用多少GPU，n是需要的数量
#SBATCH --time=7-00:00:00 # 指定任务运⾏的上限时间（我们的HPC为7天
#SBATCH --job-name=eSl_SEE-v1
#SBATCH --output=./logs/SeeEverythingEverytime/EvLight/EvLight_SEE-v1/%j-job.out # slurm的输出文件，%j是jobid
#SBATCH --error=./logs/SeeEverythingEverytime/EvLight/EvLight_SEE-v1/%j-job.err # 指定错误输出的格式
#SBATCH --cpus-per-task=1

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"
echo "Python version: $(python --version)"
echo "              : $(which python)"
echo "CUDA version: $(nvcc --version)"
nvidia-smi


export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python see/main.py \
  --yaml_file="options/SEE/EvLight_SEE-v1.yaml" \
  --log_dir="./logs/SEE/EvLight_SEE-v1/" \
  --alsologtostderr=True

# Test
# python see/main.py \
  # --yaml_file=<YAML> \
  # --log_dir=<LOG> \
  # --alsologtostderr=True  \
  # --RESUME_PATH=<PRETRAIN_MODEL> \
  # --TEST_ONLY=True \
  # --VISUALIZE=True