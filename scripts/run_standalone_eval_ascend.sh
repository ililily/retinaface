
echo "=============================================================================================================="
echo "Please run the script as: "
echo "for example: bash run_standalone_eval_ascend.sh [CKPT_FILE]"
echo "=============================================================================================================="

export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=1
export RANK_ID=0

python eval.py --backbone_name 'ResNet50' --val_model $1 > ./eval.log 2>&1 &
