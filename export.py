
import argparse
import numpy as np

from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.config import cfg_res50, cfg_mobile025

parser = argparse.ArgumentParser(description='retinaface export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--backbone_name", type=str, default='ResNet50', help="Backbone name")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="retinaface", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target(default: Ascend)")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)


def export_net():
    """export net"""
    if args.backbone_name == 'ResNet50':
        from src.network_with_resnet import RetinaFace, resnet50
        cfg = cfg_res50

        backbone = resnet50(1001)
        network = RetinaFace(phase='predict', backbone=backbone)

    elif args.backbone_name == 'MobileNet025':
        from src.network_with_mobilenet import RetinaFace, resnet50, mobilenet025
        cfg = cfg_mobile025

        if cfg['name'] == 'ResNet50':
            backbone = resnet50(1001)
        elif cfg['name'] == 'MobileNet025':
            backbone = mobilenet025(1000)
        network = RetinaFace(phase='predict', backbone=backbone, cfg=cfg)

    if cfg['val_origin_size']:
        height, width = 5568, 1056
    else:
        height, width = 2176, 2176

    backbone.set_train(False)
    network.set_train(False)

    assert args.ckpt_file is not None, "checkpoint_path is None."
    param_dict = load_checkpoint(args.ckpt_file)
    network.init_parameters_data()
    load_param_into_net(network, param_dict)
    input_arr = Tensor(np.zeros([args.batch_size, 3, height, width], np.float32))
    export(network, input_arr, file_name=args.file_name, file_format=args.file_format)

if __name__ == '__main__':
    export_net()
