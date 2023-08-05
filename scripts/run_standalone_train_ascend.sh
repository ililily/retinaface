
echo "Usage: bash ./scripts/run_standalone_train_ascend.sh"

export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1

python train.py --backbone_name 'ResNet50' > train.log 2>&1 &
