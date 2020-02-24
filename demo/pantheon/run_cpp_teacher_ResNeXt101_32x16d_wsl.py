#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle.fluid as fluid

from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

from utils import parse_args, sample_generator, sample_list_generator, batch_generator
from paddleslim.pantheon import Teacher


def run(args):
    if args.out_path and args.out_port:
        raise ValueError("args.out_path and args.out_port should not be valid "
                         "at the same time")
    if not args.out_path and not args.out_port:
        raise ValueError("One of args.out_path and args.out_port be valid")

    model_file = "ResNeXt101_32x16d_wsl_inference_model"
    gpu_id = 0
    config = AnalysisConfig(model_file)
    config.enable_use_gpu(100, gpu_id)
    predictor = create_paddle_predictor(config)

    teacher = Teacher(out_path=args.out_path, out_port=args.out_port)
    teacher.start()

    reader = batch_generator(max_n=1000000, batch_size=args.batch_size)

    schema = {}
    schema["feed_image"] = {
        "shape": [args.batch_size, 3, 224, 224],
        "dtype": "float32",
        "lod_level": 0
    }
    schema["feed_label"] = {
        "shape": [args.batch_size, 1],
        "dtype": "int64",
        "lod_level": 0
    }
    schema["knowledge"] = {
        "shape": [args.batch_size, 1000],
        "dtype": "float32",
        "lod_level": 0
    }

    teacher.start_cpp_knowledge_service(
        schema=schema,
        predictor=predictor,
        reader=reader,
        times=args.serving_times)


if __name__ == '__main__':
    args = parse_args()
    run(args)
