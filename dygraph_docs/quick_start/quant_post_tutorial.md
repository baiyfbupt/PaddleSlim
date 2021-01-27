#  图像分类模型离线量化-快速开始

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

训练完成后导出预测模型:
```python
paddle.jit.save(net, "./fp32_inference_model", input_spec=[inputs])
```


## 4.离线量化

调用slim接口将原模型转换为离线量化模型：

```python
paddle.enable_static()
place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
paddleslim.quant.quant_post_static(
        executor=exe,
        model_dir='./',
        model_filename='fp32_inference_model.pdmodel',
        params_filename='fp32_inference_model.pdiparams',
        quantize_model_path='./quant_post_static_model',
        sample_generator=train_dataset,
        batch_nums=10)
```