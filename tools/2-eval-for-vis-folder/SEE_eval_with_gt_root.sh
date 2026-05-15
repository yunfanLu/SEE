echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"
echo "Python version: $(python --version)"
echo "              : $(which python)"
echo "CUDA version: $(nvcc --version)"
nvidia-smi

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="./":$PYTHONPATH

# Example:
# EVAL_ROOT="/root/autodl-tmp/models/SEENet_SEE/eval_0513"
# GT_ROOT="/root/autodl-tmp/datasets/SEE_gt"
# python tools/2-eval-for-vis-folder/SEE_eval_for_model_with_gt_root.py \
#     --root="$EVAL_ROOT" \
#     --gt_root="$GT_ROOT" \
#     --log_dir="$EVAL_ROOT" \
#     --alsologtostderr=True

parallel python tools/2-eval-for-vis-folder/SEE_eval_for_model_with_gt_root.py \
    --root={} \
    --gt_root=/root/autodl-tmp/data/SEE_gt \
    --log_dir={} \
    --alsologtostderr=True ::: \
    "/root/autodl-tmp/models/SEENet_SEE/eval_0513/"
