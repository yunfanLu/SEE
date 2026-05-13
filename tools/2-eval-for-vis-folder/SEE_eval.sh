
echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"
echo "Python version: $(python --version)"
echo "              : $(which python)"
echo "CUDA version: $(nvcc --version)"
nvidia-smi


export CUDA_VISIBLE_DEVICES="0"
# export PATH="/hpc2hdd/home/ylu066/miniconda3/bin/":$PATH
export PYTHONPATH="./":$PYTHONPATH

# # Training in SDE and testing in SEE

# # DIR="logs/SeeDynamicEventDataset/DeepCurveEstimation/DCE_SDE-v1/SEE-test-vis-036/"
# # python tools/2-paper-figure/2-eval-for-vis-folder/SEE_eval_for_model.py \
# #     --root=$DIR \
# #     --log_dir=$DIR \
# #     --alsologtostderr=True

# DIR="logs/SeeDynamicEventDataset/EIFT_AAAI/EIFT_AAAI_SDE-v1/SEE-test-vis-033/"
# python tools/2-paper-figure/2-eval-for-vis-folder/SEE_eval_for_model.py \
#     --root=$DIR \
#     --log_dir=$DIR \
#     --alsologtostderr=True

# DIR="logs/SeeDynamicEventDataset/eSL/eSl_SDE-v1/SEE-test-vis-050/"
# python tools/2-paper-figure/2-eval-for-vis-folder/SEE_eval_for_model.py \
#     --root=$DIR \
#     --log_dir=$DIR \
#     --alsologtostderr=True

# DIR="logs/SeeDynamicEventDataset/EvLight/EvLight_SDE-v1/SEE-test-vis-041/"
# python tools/2-paper-figure/2-eval-for-vis-folder/SEE_eval_for_model.py \
#     --root=$DIR \
#     --log_dir=$DIR \
#     --alsologtostderr=True

# DIR="logs/SeeDynamicEventDataset/EvLowLight/EvLowLight_SDE-v1/SEE-test-vis-007/"
# python tools/2-paper-figure/2-eval-for-vis-folder/SEE_eval_for_model.py \
#     --root=$DIR \
#     --log_dir=$DIR \
#     --alsologtostderr=True


# # Training in SEE and testing in SEE

# DIR="logs/SeeEverythingEverytime/EIFT_AAAI/EIFT_AAAI_SEE-v1-docker/vis-010"
# python tools/2-paper-figure/2-eval-for-vis-folder/SEE_eval_for_model.py \
#     --root=$DIR \
#     --log_dir=$DIR \
#     --alsologtostderr=True

# DIR="logs/SeeEverythingEverytime/eSL/eSl_SEE-v1/vis-005/"
# python tools/2-paper-figure/2-eval-for-vis-folder/SEE_eval_for_model.py \
#     --root=$DIR \
#     --log_dir=$DIR \
#     --alsologtostderr=True

# DIR="logs/SeeEverythingEverytime/EvLight/EvLight_SEE-v1-docker/vis-001/"
# python tools/2-paper-figure/2-eval-for-vis-folder/SEE_eval_for_model.py \
#     --root=$DIR \
#     --log_dir=$DIR \
#     --alsologtostderr=True

parallel python tools/2-eval-for-vis-folder/SEE_eval_for_model.py --root={} --log_dir={} --alsologtostderr=True ::: \
    "/root/autodl-tmp/models/eSl_SEE/eval_0513/" 
    # "/root/autodl-tmp/models/SEENet_SEE/eval_0513/" 
    # "/root/autodl-tmp/models/EvLight/eval_0513/" 
    # "/root/autodl-tmp/models/EIFT_AAAI_SEE/eval_0513/" 
    # "logs/SeeEverythingEverytime/SEENet/Ablation-Release/SEENet_SEE-ar103-96-4-8-4-loss-2-epoch-20-A40-no9x9-loop-10/vis-014/"