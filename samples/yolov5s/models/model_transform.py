import argparse
import os
import sys

import tcim
import torch
import numpy as np
import torchvision.transforms as transforms
from pyasn1_modules.rfc2985 import sequenceNumber
from tvm.relay.frontend.onnx import onnx_input

''' Add dependencies '''
BASE_DIR=os.path.dirname(os.path.abspath(__file__))

def add_python_path(base_dir, relative_path):
    pythonlib_dir = os.path.join(base_dir, relative_path)
    sys.path.append(pythonlib_dir)
    print("add pythonlib:", pythonlib_dir)

#add hmassist
add_python_path(BASE_DIR, '../../deps/python/hmassist')
# add hmmodel
add_python_path(BASE_DIR, '../../deps/python/hmodel')

from hmquant.api import quant_single_onnx_network, generate_golden, quantize_profiling
from hmquant.tools.dataset.preprocess.transform import ToTensorNotNormal

dataset_path = os.path.join(BASE_DIR, '../../../datasets')

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        dest='model_path',
                        type=str,
                        default=os.path.join(os.getenv("MODEL_PATH", default=""),
                        "yolov5s_clip.onnx"),
                        help='path to model path',
                        )

    parser.add_argument(
        '--model_name',
        dest='model_name',
        type=str,
        default='yolov5s',
        help = 'model name',
    )

    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default=os.path.join('output', os.getenv('HOMO_TARGET', ''), 'result'),
        help='model name',
    )

    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=1,
        help='batch size',
    )
    parser.add_argument(
        '--stage',
        dest='stage',
        type=str,
        default="all",
        help='build stage choise=["build", "test", "all"]',
    )

    args = parser.parse_args()
    return args

def calibrate(args=None):
    model_path = args.model_path
    model_name = args.model_name
    output_path = args.model_dir

    env_dict = os.environ

    def preprocess(filepath):
        import cv2
        from hmassist.utils import utils
        from hmassist.utils.box_utils import letterbox

        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, _, _ = letterbox(image, [640,640], stride=64, auto=False) #HWC
        image = np.transpose(image, (2, 0, 1)) #CHW
        image = np.expand_dims(image, axis=0) #NCHW
        data = torch.tensor(image.astype(np.float32))
        return data

    calib_num = 20 # 20张图片量化
    calib_files = []
    calib_dir = os.path.join(dataset_path, 'coco2017/val2017')
    file_list = os.listdir(calib_dir)
    for filename in file_list:
        _, ext = os.path.splitext(filename)
        if ext in ['.jpg', '.JPG', '.bmp', '.png', '.jpeg', '.BMP', '.bin']:
            calib_files.append(filename)
            if len(calib_files) == calib_num:
                break
    print("calib_files=", calib_files)
    calib_dataset = [
        preprocess(os.path.join(calib_dir, file_path)) for file_path in calib_files
    ]

    quanttool_config = {
        "inputs_cfg": {
            'ALL': {
                'data_format': 'RGB',
                'first_layer_weight_denorm_mean': [0, 0, 0],
                'first_layer_weight_denorm_std':[1, 1, 1],
                'resizer_crop':{'top':0, 'left':0, 'height':640, 'width': 640},
                "resizer_resize": {
                    'height': 640,
                    'width':640,
                    'align_corners': False,
                    'method': 'bilinear',
                },
                'toYUV_format':'YUV420',
            },
        },
        'graph_opt_cfg': {},
    }

    onnx_input = calib_dataset[0]

    print("start calibrating ...")
    sequencer = quant_single_onnx_network(
        quanttool_config,
        calib_dataset,
        model_path,
        device='cpu',
    )

    print("start save model and generate golden ...")
    generate_golden(
        sequencer=sequencer,
        calibset=onnx_input,
        save_path=output_path,
        model_name = model_name,
        batch_size = 1,
        device = 'cpu'
    )

    print('start quantize profiling ...')
    quantize_profiling(sequencer, [onnx_input])
    print("calibrate completed")

def build_hmmodel(args=None):
    model_name = args.model_name
    batch = args.batch
    stage = args.stage
    model_dir = args.model_dir
    quant_name = "hmquant_" + model_name + "_with_act"
    onnx_name = quant_name + ".onnx"
    model_path = os.path.join(model_dir, onnx_name)
    model_dir = os.path.dirname(model_path)

    # build
    import tcim
    from hmassist.utils.dist_metrics import cosine_distance
    import onnx

    # build_model
    if stage == 'build' or stage =='all':
        onnx_model = onnx.load(model_path)
        compile_config = {}
        if batch % 4 ==0:
            compile_config["tcim.core_num"] = 4;
        input_cfg = {}
        inputs = onnx_model.graph.input
        for input in inputs:
            dims = input.type.tensor_type.shape.dim
            input_shape = [dim.dim_value for dim in dims]
            input_shape[0] *= batch
            input_cfg[input.name]= tcim.HMInput(shape=input_shape)
        tcim.build.build_from_hmonnx(onnx_model, model_name=model_name, input=input_cfg, compiler_cfg=compile_config)
        print(model_name, 'build_completed.')
    # test model
    if stage == 'test' or stage == 'all':
        # 2.1 load model
        module = tcim.runtime.load(model_name + ".hmm")
        # 2.2 set input with golden
        input_num = module.get_num_inputs()
        for id in range(input_num):
            input_name = module.get_input_name(id)
            input_info = module.get_input_info(input_name)
            print("input_info[{}] shape={}, dtype={}, format={}".format(input_name,
                                                                        input_info.shape,
                                                                        input_info.dtype,
                                                                        input_info.format.name))
            input_file_name = 'hmquant_' + model_name + '_' + input_name + '_input.npy'
            input_data_path = os.path.join(model_dir, input_file_name)
            input_data = np.load(input_data_path).astype(input_info.dtype)
            input_data = np.concatenate([input_data for i in range(batch)], axis=0)
            print("golden input[{}] shape={}, dtype={}".format(input_name, input_data.shape,
                                                               input_data.dtype))
            module.set_input(input_name, input_data)
        # 2.3 infer model
        module.run()
        module.sync()

        # 2.4 get output and compare with golden
        result_check = True
        output_num = module.get_num_outputs()
        for id in range(output_num):
            output_name = module.get_output_name(id)
            output_info = module.get_output_info(output_name, is_quanted=True)
            print("output_info[{}] shape = {}, dtype = {}, format = {}".format(output_name, output_info.shape,
                                                                               output_info.dtype,
                                                                               output_info.format.name))
            output_data = module.get_output(output_name, is_quanted=True)
            print("output[{}] shape = {}, dtype = {}".format(output_name, output_data.shape, output_data.dtype))
            output_data_path = os.path.join(model_dir, 'hmquant_' + model_name + '_' + output_name + '_output.npy')
            if os.path.exists(output_data_path):
                golden_output = np.load(output_data_path)
                golden_output = np.concatenate([golden_output for i in range(batch)], axis=0)
            else:
                result_check = False
                print("[warning] compare canceled while golden data not found -> {}".format(output_data_path))
                continue
            if golden_output.shape == output_data.shape:
                cosine_dist = cosine_distance(golden_output, output_data)
                is_match = (golden_output == output_data).all()
                print("[compare] golden output [{}] match={}, similarity={:.6f}"
                      .format(output_name, is_match, cosine_dist))
                if cosine_dist < 0.99:
                    result_check = False
            else:
                result_check = False
                print("[compare] golden output [{}] shape not match {} vs {}"
                      .format(output_name, golden_output.shape, output_data.shape))
        if not result_check:
            print("[error] result check failed.")
            exit(-1)

if __name__ == '__main__':
    print("current file is: ", __file__)
    args = get_args()
    calibrate(args)
    build_hmmodel(args)

