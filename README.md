# retinaface描述

Retinaface人脸检测模型于2019年提出，应用于WIDER FACE数据集时效果最佳。RetinaFace论文：RetinaFace: Single-stage Dense Face Localisation in the Wild。与S3FD和MTCNN相比，RetinaFace显著提上了小脸召回率，但不适合多尺度人脸检测。为了解决这些问题，RetinaFace采用RetinaFace特征金字塔结构进行不同尺度间的特征融合，并增加了SSH模块。

[论文](https://arxiv.org/abs/1905.00641v2)：  Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, Stefanos Zafeiriou. "RetinaFace: Single-stage Dense Face Localisation in the Wild". 2019.

# 预训练模型

RetinaFace可以使用ResNet50或MobileNet0.25骨干提取图像特征进行检测。使用ResNet50充当backbone时需要使用./src/resnet.py作为模型文件，然后从ModelZoo中获取ResNet50的训练脚本（使用默认的参数配置）在ImageNet2012上训练得到ResNet50的预训练模型。

# 模型架构

具体来说，RetinaFace是基于RetinaNet的网络，采用了RetinaNet的特性金字塔结构，并增加了SSH结构。网络中除了传统的检测分支外，还增加了关键点预测分支和自监控分支。结果表明，这两个分支可以提高模型的性能。这里我们不介绍自我监控分支。

# 数据集

使用的数据集： [WIDERFACE](<http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html>)

- 数据集目录结构如下所示：

    ```bash
    ├── data/
        ├── widerface/
            ├── ground_truth/
            │   ├──wider_easy_val.mat
            │   ├──wider_face_val.mat
            │   ├──wider_hard_val.mat
            │   ├──wider_medium_val.mat
            ├── train/
            │   ├──images/
            │   │   ├──0--Parade/
            │   │   │   ├──0_Parade_marchingband_1_5.jpg
            │   │   │   ├──...
            │   │   ├──.../
            │   ├──label.txt
            ├── val/
            │   ├──images/
            │   │   ├──0--Parade/
            │   │   │   ├──0_Parade_marchingband_1_20.jpg
            │   │   │   ├──...
            │   │   ├──.../
            │   ├──label.txt
    ```

# 环境要求

- 硬件（Ascend）
    - 使用ResNet50作为backbone时用Ascend来搭建硬件环境。

- Ascend处理器环境运行（使用ResNet50作为backbone）

  ```python
  # 训练
  python train.py --backbone_name 'ResNet50' > train.log 2>&1 &

  # 评估
  python eval.py --backbone_name 'ResNet50' --val_model [CKPT_FILE] > ./eval.log 2>&1 &

  # 推理
  bash run_infer_310.sh ../retinaface.mindir /home/dataset/widerface/val/ 0
  ```

## 脚本及样例代码

```bash
    ├── retinaface
        ├── README_CN.md                           // Retinaface相关说明
        ├── ascend310_infer                        // 实现310推理源代码
        ├── scripts
        │   ├──run_distribution_train_ascend.sh    // 分布式到Ascend的shell脚本
        │   ├──run_infer_310.sh                    // Ascend推理的shell脚本（使用ResNet50作为backbone时）
        │   ├──run_standalone_eval_ascend.sh       // Ascend评估的shell脚本
        │   ├──run_standalone_train_ascend.sh      // Ascend单卡训练的shell脚本
        ├── src
        │   ├──augmentation.py                     // 数据增强方法
        │   ├──config.py                           // 参数配置
        │   ├──dataset.py                          // 创建数据集
        │   ├──loss.py                             // 损失函数
        │   ├──lr_schedule.py                      // 学习率衰减策略
        │   ├──network_with_resnet.py              // 使用ResNet50作为backbone的RetinaFace架构
        │   ├──resnet.py                           // 使用ResNet50作为backbone时预训练要用到的ResNet50架构
        │   ├──utils.py                            // 数据预处理
        ├── data
        │   ├──widerface                           // 数据集
        │   ├──resnet-90_625.ckpt                  // ResNet50 ImageNet预训练模型
        │   ├──ground_truth                        // 评估标签
        ├── eval.py                                // 评估脚本
        ├── export.py                              // 将checkpoint文件导出到air/mindir（使用ResNet50作为backbone时）
        ├── postprocess.py                         // 310推理后处理脚本
        ├── preprocess.py                          // 310推理前处理脚本
        ├── train.py                               // 训练脚本
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置使用ResNet50作为backbone的RetinaFace和WIDER FACE数据集

  ```python
    'variance': [0.1, 0.2],                                   # 方差
    'clip': False,                                            # 裁剪
    'loc_weight': 2.0,                                        # Bbox回归损失权重
    'class_weight': 1.0,                                      # 置信度/类回归损失权重
    'landm_weight': 1.0,                                      # 地标回归损失权重
    'batch_size': 8,                                          # 训练批次大小
    'num_workers': 16,                                        # 数据集加载数据的线程数量
    'num_anchor': 29126,                                      # 矩形框数量，取决于图片大小
    'nnpu': 8,                                                # 训练的NPU数量
    'image_size': 840,                                        # 训练图像大小
    'match_thresh': 0.35,                                     # 匹配框阈值
    'optim': 'sgd',                                           # 优化器类型
    'momentum': 0.9,                                          # 优化器动量
    'weight_decay': 1e-4,                                     # 优化器权重衰减
    'epoch': 60,                                              # 训练轮次数量
    'decay1': 20,                                             # 首次权重衰减的轮次数
    'decay2': 40,                                             # 二次权重衰减的轮次数
    'initial_lr':0.04                                         # 初始学习率，八卡并行训练时设置为0.04
    'warmup_epoch': -1,                                       # 热身大小，-1表示无热身
    'gamma': 0.1,                                             # 学习率衰减比
    'ckpt_path': './checkpoint/',                             # 模型保存路径
    'keep_checkpoint_max': 8,                                 # 预留检查点数量
    'resume_net': None,                                       # 重启网络，默认为None
    'training_dataset': '../data/widerface/train/label.txt',  # 训练数据集标签路径
    'pretrain': True,                                         # 是否基于预训练骨干进行训练
    'pretrain_path': '../data/resnet-90_625.ckpt',            # 预训练的骨干检查点路径
    # 验证
    'val_model': './train_parallel3/checkpoint/ckpt_3/RetinaFace-56_201.ckpt', # 验证模型路径
    'val_dataset_folder': './data/widerface/val/',            # 验证数据集路径
    'val_origin_size': True,                                  # 是否使用全尺寸验证
    'val_confidence_threshold': 0.02,                         # 验证置信度阈值
    'val_nms_threshold': 0.4,                                 # 验证NMS阈值
    'val_iou_threshold': 0.5,                                 # 验证IOU阈值
    'val_save_result': False,                                 # 是否保存结果
    'val_predict_save_folder': './widerface_result',          # 结果保存路径
    'val_gt_dir': './data/ground_truth/',                     # 验证集ground_truth路径
    # 推理
    'infer_dataset_folder': '/home/dataset/widerface/val/',   # 310进行推理时验证数据集路径
    'infer_gt_dir': '/home/dataset/widerface/ground_truth/',  # 310进行推理时验证集ground_truth路径
  ```
## 训练过程

### 用法

- Ascend处理器环境运行（使用ResNet50作为backbone）

  ```bash
  # 将src/config.py文件中nnpu参数改为1
  python train.py --backbone_name 'ResNet50' > train.log 2>&1 &
  OR
  bash ./scripts/run_standalone_train_ascend.sh
  ```

  上述python命令在后台运行，可通过`train.log`文件查看结果。

  训练结束后，可以得到损失值：

  ```bash
  epoch: 7 step: 1609, loss is 5.327434
  epoch time: 466281.709 ms, per step time: 289.796 ms
  epoch: 8 step: 1609, loss is 4.7512465
  epoch time: 466995.237 ms, per step time: 290.239 ms
  ```

- GPU处理器环境运行（使用MobileNet0.25作为backbone）

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python train.py --backbone_name 'MobileNet025' > train.log 2>&1 &
  OR
  bash ./scripts/run_standalone_train_gpu.sh
  ```

  上述python命令在后台运行，可通过`train.log`文件查看结果。

  训练结束后，可在默认文件夹`./checkpoint/`中找到检查点文件。

## 评估过程

### 评估

- Ascend环境运行评估WIDER FACE数据集（使用ResNet50作为backbone）

  CKPT_FILE是用于评估的检查点路径。如'./train_parallel3/checkpoint/ckpt_3/RetinaFace-56_201.ckpt'。

  ```bash
  python eval.py --backbone_name 'ResNet50' --val_model [CKPT_FILE] > ./eval.log 2>&1 &
  OR
  bash run_standalone_eval_ascend.sh [CKPT_FILE]
  ```

  上述python命令在后台运行，可通过"eval.log"文件查看结果。测试数据集的准确率如下：

  ```python
  # grep "Val AP" eval.log
  Easy   Val AP : 0.9516
  Medium Val AP : 0.9381
  Hard   Val AP : 0.8403
  ```


## 预测过程

### 预测

  当前预测支持ResNet50作为backbone输入单张图片进行预测。

  CKPT_FILE是用于评估的检查点路径。如'./train_parallel3/checkpoint/ckpt_3/RetinaFace-56_201.ckpt'。

  IMG_PATH是用于预测的图片。如'./test.png'

  ```bash
  python predict.py --ckpt_file [CKPT_FILE] --img_path [IMG_PATH] > ./predict.log 2>&1 &
  ```

  上述python命令在后台运行，可通过"predict.log"文件查看结果，样例如下：

  ```python
  Prediction res: [[60.0, 40.0, 67.0, 94.0, 0.9995]]
  Prediction avg time: 11.9745 ms
  ```

## 导出过程

### 导出

将checkpoint文件导出成mindir格式模型。（使用ResNet50作为backbone）

  ```shell
  python export.py --ckpt_file [CKPT_FILE]
  ```

### 推理

在进行推理之前我们需要先导出模型。mindir可以在任意环境上导出，air模型只能在昇腾910环境上导出。以下展示了使用mindir模型执行推理的示例。

- 使用WIDER FACE数据集进行推理（使用ResNet50作为backbone）

  执行推理的命令如下所示，其中'MINDIR_PATH'是mindir文件路径；'DATASET_PATH'是使用的推理数据集所在路径，如'/home/dataset/widerface/val/'；'DEVICE_ID'可选，默认值为0。

  ```shell
  bash run_infer_cpp.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_TYPE] [DEVICE_ID]
  ```

  推理的精度结果保存在scripts目录下，在acc.log日志文件中可以找到类似以下的分类准确率结果。推理的性能结果保存在scripts/time_Result目录下，在test_perform_static.txt文件中可以找到类似以下的性能结果。

  ```bash
  Easy   Val AP : 0.9498
  Medium Val AP : 0.9351
  Hard   Val AP : 0.8306
  NN inference cost average time: 365.584 ms of infer_count 3226
  ```

