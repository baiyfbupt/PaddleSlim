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

import argparse
import os
import sys
import time
import numpy as np
import paddle.fluid as fluid
from paddleslim.pantheon import Student
from utils import str2bool
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
import models
model_list = [m for m in dir(models) if "__" not in m]


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--in_address0",
        type=str,
        default=None,
        help="Input address for teacher 0. (default: %(default)s)")
    parser.add_argument(
        "--in_path0",
        type=str,
        default=None,
        help="Input file path for teacher 0. (default: %(default)s)")
    parser.add_argument(
        "--in_address1",
        type=str,
        default=None,
        help="Input address for teacher 1. (default: %(default)s)")
    parser.add_argument(
        "--in_path1",
        type=str,
        default=None,
        help="Input file path for teacher 1. (default: %(default)s)")
    parser.add_argument(
        "--use_cuda",
        type=str2bool,
        default=False,
        help="Whether to use GPU for prediction. (default: %(default)s)")
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet50_vd",
        help="Student model. (default: %(default)s)")
    parser.add_argument(
        "--test_send_recv",
        type=str2bool,
        default=False,
        help="Whether to test send/recv interfaces. (default: %(default)s)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The batch size of student model. (default: %(default)s)")
    args = parser.parse_args()
    return args


def run(args):
    if args.in_address0 and args.in_path0:
        raise ValueError(
            "args.in_address0 and args.in_path0 should not be valid "
            "at the same time!")
    if not args.in_address0 and not args.in_path0:
        raise ValueError(
            "One of args.in_address0 and args.in_path0 must be valid!")

    #student = Student(merge_strategy={"result": "sum"})

    #student.register_teacher(
    #    in_address=args.in_address0, in_path=args.in_path0)
    #student.start()

    if args.test_send_recv:
        for t in xrange(2):
            for i in xrange(3):
                print(student.recv(t))
        student.send("message from student!")

    #knowledge_desc = student.get_knowledge_desc()
    #data_generator = student.get_knowledge_generator(
    #    batch_size=args.batch_size, drop_last=False)

    def data_reader():
        def reader():
            for data in data_generator():
                yield data['feed_image'], data['feed_label'], data['knowledge']

        return reader

    def sample_generator(max_n):
        def wrapper():
            for i in range(max_n):
                yield [
                    np.random.rand(3, 224, 224), np.random.rand(1),
                    np.random.rand(1000)
                ]

        return wrapper

    def batch_generator(max_n, batch_size=128):
        def wrapper():
            data_batch = []
            label_batch = []
            knowledge_batch = []
            for sample in sample_generator(max_n)():
                if len(data_batch) < batch_size:
                    data_batch.append(sample[0])
                    label_batch.append(sample[1])
                    knowledge_batch.append(sample[2])
                if len(data_batch) == batch_size:
                    yield [
                        np.array(data_batch).astype('float32').reshape(
                            (-1, 3, 224, 224)),
                        np.array(label_batch).astype('int64').reshape((-1, 1)),
                        np.array(knowledge_batch).astype('float32').reshape(
                            (-1, 1000))
                    ]
                    data_batch = []
                    label_batch = []
                    knowledge_batch = []
            if len(data_batch) > 0:
                yield [
                    np.array(data_batch).astype('float32').reshape(
                        (-1, 3, 224, 224)),
                    np.array(label_batch).astype('int64').reshape((-1, 1)),
                    np.array(knowledge_batch).astype('float32').reshape(
                        (-1, 1000))
                ]

        return wrapper

    assert args.model in model_list, "{} is not in lists: {}".format(
        args.model, model_list)
    student_main = fluid.Program()
    student_startup = fluid.Program()
    with fluid.program_guard(student_main, student_startup):
        with fluid.unique_name.guard():
            image = fluid.layers.data(
                name='feed_image', shape=[3, 224, 224], dtype='float32')
            label = fluid.layers.data(
                name='feed_label', shape=[1], dtype='int64')
            knowledge = fluid.layers.data(
                name='knowledge', shape=[1000], dtype='float32')
            train_loader = fluid.io.DataLoader.from_generator(
                feed_list=[image, label, knowledge],
                capacity=64,
                use_double_buffer=True,
                iterable=True)
            # model definition
            model = models.__dict__[args.model]()
            out = model.net(input=image, class_dim=1000)
            student_feature = student_main.global_block().var('fc_0.tmp_1')
            distill_loss = fluid.layers.reduce_mean(
                fluid.layers.square(knowledge - student_feature),
                name='distill_loss')
            cost = fluid.layers.cross_entropy(input=out, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            loss = distill_loss + avg_cost
            opt = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.1, momentum=0.9)
            opt.minimize(loss)

    student_vars = []
    for v in student_main.list_vars():
        try:
            student_vars.append((v.name, v.shape))
        except:
            pass
    #print(student_vars)

    build_strategy = fluid.BuildStrategy()
    build_strategy.fuse_all_reduce_ops = False
    student_main = fluid.CompiledProgram(student_main).with_data_parallel(
        loss_name=loss.name, build_strategy=build_strategy)

    places = fluid.cuda_places() if args.use_cuda else fluid.cpu_places()
    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(student_startup)
    train_loader.set_batch_generator(
        batch_generator(
            1000000, batch_size=128), places=places)

    for step_id, batch_data in enumerate(train_loader):
        fetch_np = exe.run(student_main, batch_data, fetch_list=[loss.name])
        if step_id % 10 == 0:
            print("Step {}".format(step_id))
        if step_id == 100:
            start_time = time.time()
        elif step_id == 200:
            end_time = time.time()
            break
    print("Runs 100 steps costs {:.6f} s".format(end_time - start_time))


if __name__ == '__main__':
    args = parse_args()
    run(args)
