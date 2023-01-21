# NeuralFlow

Deep learning framework built with numpy (cupy)  
This version supports cuda 11.x ver

## install
```shell
$ git clone git clone https://github.com/augustinLib/neuralflow.git
```
or
```
# cpu-only
$ pip install neuralflow-cpu

# gpu (cuda 11.x)
$ pip install neuralflow
```

## Quick guide
you can build model like this,
```python
from neuralflow.model import Model, DenseLayer, ConvLayer, MaxPoolingLayer
from neuralflow.function_class import ReLU

model = Model(
    DenseLayer(784, 50),
    ReLU(),
    DenseLayer(50, 10)
)

conv_model = Model(
    ConvLayer(input_channel = 1, output_channel = 30, kernel_size = 5, stride = 1, padding=0),
    ReLU(),
    MaxPoolingLayer(kernel_size=2, stride=2),
    DenseLayer(4320, 100),
    ReLU(),
    DenseLayer(100, 10)
)
```
and the training proceeds as follows.
```python
from neuralflow.function_class import ReLU, CrossEntropyLoss
from neuralflow.optimizer import Adam

critic = CrossEntropyLoss()
optim = Adam()
pred = model(x)
loss = critic(pred, y)
model.backward(critic)
optim.update(model)

```
you can also train model with trainer
```python
from neuralflow.trainer import ClassificationTrainer
from neuralflow.data import DataLoader

dataloader = DataLoader(train_data)
trainer = ClassificationTrainer(model,
                                critic,
                                optim,
                                epochs,
                                init_lr = 0.001)
trainer.train(dataloader)
```
when using gpu, set it as follows.
```python
# using gpu
from neuralflow import config
config.GPU = True

# using cpu
from neuralflow import config
config.GPU = False
```

## Structure
- neuralflow
    - \_\_init\_\_.py
    - data.py
    - function.py
    - function_class.py
    - model.py
    - optimizer.py
    - trainer.py
    - utils.py
    - nlp
      - utils.py
    - epoch_notice
      - send_message.py
      - token_generator.py
- dataset
- test
- README.md
- .gitignore

