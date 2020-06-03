# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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
"dygraph transformer layers"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import json
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, LayerNorm, Linear, to_variable, Layer, guard
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid import ParamAttr
from paddle.fluid.initializer import MSRA
from .transformer_encoder import EncoderLayer
from paddle.fluid.dygraph.base import to_variable


class BertModelLayer(Layer):
    def __init__(self,
                 emb_size=128,
                 hidden_size=768,
                 n_layer=12,
                 voc_size=30522,
                 max_position_seq_len=512,
                 sent_types=2,
                 return_pooled_out=True,
                 initializer_range=1.0,
                 conv_type="conv_bn",
                 search_layer=False,
                 use_fp16=False):
        super(BertModelLayer, self).__init__()

        self._emb_size = emb_size
        self._hidden_size = hidden_size
        self._n_layer = n_layer
        self._voc_size = voc_size
        self._max_position_seq_len = max_position_seq_len
        self._sent_types = sent_types
        self.return_pooled_out = return_pooled_out

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._dtype = "float16" if use_fp16 else "float32"

        self._conv_type = conv_type
        self._search_layer = search_layer
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=initializer_range)

        self._src_emb = Embedding(
            size=[self._voc_size, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            dtype=self._dtype)

        self._pos_emb = Embedding(
            size=[self._max_position_seq_len, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer),
            dtype=self._dtype)

        self._sent_emb = Embedding(
            size=[self._sent_types, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._sent_emb_name, initializer=self._param_initializer),
            dtype=self._dtype)

        # BERT_BASE_PATH = "./data/pretrained_models/uncased_L-12_H-768_A-12/"
        # dir_path = BERT_BASE_PATH + "/dygraph_params/"

        # def load_numpy_weight(file_name):
        #     if six.PY2:
        #         res = np.load(
        #             os.path.join(dir_path, file_name), allow_pickle=True)
        #     else:
        #         res = np.load(
        #             os.path.join(dir_path, file_name),
        #             allow_pickle=True,
        #             encoding='latin1')
        #     assert res is not None
        #     return res

        # # load word embedding
        # _param = load_numpy_weight("word_embedding")
        # self._src_emb.set_dict({"weight": _param})
        # print("INIT word embedding")

        # _param = load_numpy_weight("pos_embedding")
        # self._pos_emb.set_dict({"weight": _param})
        # print("INIT pos embedding")

        # _param = load_numpy_weight("sent_embedding")
        # self._sent_emb.set_dict({"weight": _param})
        # print("INIT sent embedding")

        self._emb_fac = Linear(
            input_dim=self._emb_size,
            output_dim=self._hidden_size,
            param_attr=fluid.ParamAttr(name="s_emb_factorization"))

        # self.pooled_fc = Linear(
        #     input_dim=self._hidden_size,
        #     output_dim=self._hidden_size,
        #     param_attr=fluid.ParamAttr(
        #         name="s_pooled_fc.w_0", initializer=self._param_initializer),
        #     bias_attr="s_pooled_fc.b_0",
        #     act="tanh")

        self._encoder = EncoderLayer(
            n_layer=self._n_layer,
            hidden_size=self._hidden_size,
            search_layer=self._search_layer)

    def max_flops(self):
        return self._encoder.max_flops

    def max_model_size(self):
        return self._encoder.max_model_size

    def arch_parameters(self):
        return [self._encoder.alphas]  #, self._encoder.k]

    def forward(self, data_ids, flops=[], model_size=[]):
        """
        forward
        """

        src_ids_a = data_ids[0]
        position_ids_a = data_ids[1]
        sentence_ids_a = data_ids[2]
        # (bs, seq_len)

        src_emb_a = self._src_emb(src_ids_a)
        pos_emb_a = self._pos_emb(position_ids_a)
        sent_emb_a = self._sent_emb(sentence_ids_a)
        # (bs, seq_len, emb_size)

        emb_out_a = src_emb_a + pos_emb_a
        emb_out_a = emb_out_a + sent_emb_a
        emb_out_a = self._emb_fac(emb_out_a)
        emb_out_b = emb_out_a
        # (bs, seq_len, hidden_size)

        if len(data_ids) > 5:
            src_ids_b = data_ids[4]
            position_ids_b = data_ids[5]
            sentence_ids_b = data_ids[6]

            src_emb_b = self._src_emb(src_ids_b)
            pos_emb_b = self._pos_emb(position_ids_b)
            sent_emb_b = self._sent_emb(sentence_ids_b)

            emb_out_b = src_emb_b + pos_emb_b
            emb_out_b = emb_out_b + sent_emb_b

            emb_out_b = self._emb_fac(emb_out_b)
            # (bs, seq_len, emb_size)

        enc_output = self._encoder(
            emb_out_a, emb_out_b, flops=flops, model_size=model_size)

        return enc_output
