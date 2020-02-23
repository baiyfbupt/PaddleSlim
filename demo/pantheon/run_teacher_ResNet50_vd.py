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

from utils import parse_args, sample_generator, sample_list_generator, batch_generator
from paddleslim.pantheon import Teacher


def run(args):
    if args.out_path and args.out_port:
        raise ValueError("args.out_path and args.out_port should not be valid "
                         "at the same time")
    if not args.out_path and not args.out_port:
        raise ValueError("One of args.out_path and args.out_port be valid")

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    program, feed_vars, fetch_vars = fluid.io.load_inference_model(
        'ResNet50_vd_inference_model', exe)
    data = program.global_block().var('feed_image')
    label = program.global_block().var('feed_label')
    logits = program.global_block().var('fc_0.tmp_1')
    teacher_vars = []
    for v in program.list_vars():
        try:
            teacher_vars.append((v.name, v.shape))
        except:
            pass
    #print(teacher_vars)

    teacher = Teacher(out_path=args.out_path, out_port=args.out_port)
    teacher.start()

    if args.generator_type == "sample_generator":
        reader_config = {
            "sample_generator": sample_generator(max_n=1000000),
            "batch_size": args.batch_size,
            "drop_last": False
        }
    elif args.generator_type == "sample_list_generator":
        reader_config = {
            "sample_list_generator": sample_list_generator(
                max_n=1000000, batch_size=args.batch_size)
        }
    else:
        reader_config = {
            "batch_generator": batch_generator(
                max_n=1000000, batch_size=args.batch_size)
        }

    teacher.start_knowledge_service(
        feed_list=['feed_image', 'feed_label'],
        schema={"feed_image": data,
                "feed_label": label,
                "knowledge": logits},
        program=program,
        reader_config=reader_config,
        exe=exe,
        times=args.serving_times)


if __name__ == '__main__':
    args = parse_args()
    run(args)
