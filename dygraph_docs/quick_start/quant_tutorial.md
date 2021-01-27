#  图像分类模型量化-快速开始

该教程以图像分类模型MobileNetV1为例，说明如何快速使用[PaddleSlim的模型量化接口]()。
该示例包含以下步骤：

1. 导入依赖
2. 构建模型和数据集
3. 进行预训练
4. 量化训练
5. 导出预测模型

以下章节依次次介绍每个步骤的内容。

## 1. 导入依赖

PaddleSlim依赖Paddle2.0-rc1版本，请确认已正确安装Paddle，然后按以下方式导入Paddle和PaddleSlim:

```python
import paddle
import paddle.vision.models as models
from paddle.static import InputSpec as Input
from paddle.vision.datasets import Cifar10
import paddle.vision.transforms as T
from paddleslim.dygraph.quant import QAT
```

## 2. 构建网络和数据集

该章节构造一个用于对CIFAR10数据进行分类的分类模型，选用`MobileNetV1`，并将输入大小设置为`[3, 32, 32]`，输出类别数为10。
为了方便展示示例，我们使用Paddle提供的预定义分类模型，执行以下代码构建分类模型：

```python
net = models.mobilenet_v1(pretrained=False, scale=1.0, num_classes=10)
inputs = [Input([None, 3, 32, 32], 'float32', name='image')]
labels = [Input([None, 1], 'int64', name='label')]
optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1,
        parameters=net.parameters())
model = paddle.Model(net, inputs, labels)
model.prepare(
        optimizer,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5)))

transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
train_dataset = Cifar10(mode='train', backend='cv2', transform=transform)
val_dataset = Cifar10(mode='test', backend='cv2', transform=transform)
```

## 3. 进行预训练

对模型进行预训练，为之后的裁剪做准备。
执行以下代码对模型进行预训练
```python
model.fit(train_dataset, epochs=5, batch_size=256, verbose=1)
model.evaluate(val_dataset, batch_size=256, verbose=1)
```


## 4. 量化训练

### 4.1 将模型转换为模拟量化模型

```python
quant_config = {
    # weight preprocess type, default is None and no preprocessing is performed.
    'weight_preprocess_type': None,
    # activation preprocess type, default is None and no preprocessing is performed.
    'activation_preprocess_type': None,
    # weight quantize type, default is 'channel_wise_abs_max'
    'weight_quantize_type': 'channel_wise_abs_max',
    # activation quantize type, default is 'moving_average_abs_max'
    'activation_quantize_type': 'moving_average_abs_max',
    # weight quantize bit num, default is 8
    'weight_bits': 8,
    # activation quantize bit num, default is 8
    'activation_bits': 8,
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    'dtype': 'int8',
    # window size for 'range_abs_max' quantization. default is 10000
    'window_size': 10000,
    # The decay coefficient of moving average, default is 0.9
    'moving_rate': 0.9,
    # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}

quanter = QAT(config=quant_config)
quanter.quantize(net)
```

### 4.2 训练量化模型

在这里我们对量化模型进行finetune训练，代码如下所示：

```python
model.fit(train_dataset, epochs=2, batch_size=256, verbose=1)
model.evaluate(val_dataset, batch_size=256, verbose=1)
```

在量化训练得到理想的量化模型之后，我们可以将其导出用于预测部署。


## 5. 导出预测模型

通过以下接口，可以直接导出量化预测模型：

```python
path="./quant_inference_model"
quanter.save_quantized_model(
    net,
    path,
    input_spec=inputs)
```

导出之后，可以在`path`路径下找到导出的量化预测模型