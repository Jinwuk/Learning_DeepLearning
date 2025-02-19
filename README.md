# Learning_DeepLearning
[toc]



"만들면서 배우는 생성 AI"를 실습하면서 여러가지 테스트 코드를 만든다. 몇 가지 테스트 코드는 실제 사용 및 실습을 염두에 두고 만든다.
환경은 py3.11.4를 중심으로 생성된다. 

## Fundamental Libraries 
다음은, 코드상에서 필요한 기본 라이브러리들이다.
그러나, 각 python 파일에서는 이 중에서 일부만 사용된다.
일부 코드에서는 불필요한 library들을 지웠으나 일부에서는 여전히 남아 있다.
operation.py 에서는 해당 라이브러리들을 모두 살려 놓을 예정이다.
~~~python
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as Transforms
from torchsummary import summary
from matplotlib import pyplot as plt
~~~

## Tensorboard 사용법
- 가급적 별도의 command 혹은 conda 창을 띄워서 사용한다.
~~~bash
tensorboard --logdir=runs
~~~

멈출때는 CTRL+C 혹은 다음과 같이 멈춘다.
~~~bash
pkill -f tensorboard
~~~
### 원격으로  Tensorboard 접속하는 법
예를 들어 Windows PC에서 Linux PC에서 수행한 결과를 Windows Web Browser에서 보고 싶다고 가정하자.
ssh로 접근한 다음 다음과 같이 Tensorboard를 수행한다.
~~~bash 
tensorboard --logdir=./runs --bind_all
~~~
그러면 다음과 같이 ssh에서 server(Linux PC) 에서의 응답이 나온다.
~~~bash
TensorFlow installation not found - running with reduced feature set.

NOTE: Using experimental fast data loading logic. To disable, pass
    "--load_fast=false" and report issues on GitHub. More details:
    https://github.com/tensorflow/tensorboard/issues/4784

TensorBoard 2.12.1 at http://sderoen-System-Product-Name:6006/ (Press CTRL+C to quit)
~~~
맨 하단의 주소 "http://sderoen-System-Product-Name:6006" 을 Windows PC의 Web Browser에서 입력하면 서버에서의 결과를 클라이언트에서 볼 수 있다.


## 원격 SSH로 Windwos conda 환경 실행하기
간단하게 anaconda 혹은 miniconda의 activation.bat 가 실행되면 된다.
그러므로 conda 설치 디렉토리에서 원격으로 다음을 실행시키면 된다.
~~~bash
C:\ProgramData\miniconda3\Scripts\activate.bat C:\ProgramData\miniconda3
~~~

## Numpy 문제
numpy 버전 문제가 있다. 본 프로그램은 numpy 2.0 이상에서는 제대로 작동하지 않는다. 에러가 발생한다.
해당 코드를 수정하면 될 것이나, 일단은 numpy 버전을 내려서 해결하도록 한다.
Numpy의 경우 numpy 1.x 버전은 1.26.3 이 가장 최신 버전이므로 이것을 사용한다.
numpy 2.0 에서 동작이 안되는 부분 코드는 추후 수정한다.

## Fashion MNIST 데이터 Dimension
Fashion MNIST Data를 Pytorch 를 사용하여 읽어들이면 다음의 Dimension을 가진다.
~~~bash
[data index, channel, x, y] = [18, 1, 32, 32]
~~~
그러므로 다음의 transpose를 통해 출력 Dimension이 다음이 되도록 한다.
~~~
[0, 2, 3, 1] -> [18, 32, 32, 1]
~~~

## Test 및 Debug
일반적인 전체 코드 테스트를 위해서라면 다음과 같이 해야 한다.
~~~bash
python main_test.py 
~~~
그러나, 일부 기능의 정상적인 수행을 테스트 하기 위해서라면 다음과 같이 arg를 주는 것이 시간 절약을 위해 좋다. 
~~~bash
python main_test.py -sg -im
~~~

## Pytorch의 함수와 Loss function 문제 (Softmax vs Loss-functions)
- pytorch loss function의 입력은 어떻게 되는가? softmax와 같은 nn.function의 출력을 입력으로 받아들이지 못하는가?
### Answer
PyTorch의 손실 함수는 일반적으로 모델의 출력과 대상(타깃) 값을 입력으로 받습니다. softmax와 같은 torch.nn.functional 함수의 출력을 직접 손실 함수의 입력으로 사용할 수 있습니다.
예를 들어, 분류 문제에서 CrossEntropyLoss를 사용할 때, 모델의 출력을 softmax를 거치지 않고 바로 사용해도 됩니다. CrossEntropyLoss는 내부적으로 softmax와 log를 적용하기 때문입니다. 다음은 그 예제입니다:

- 예제 코드
~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 예제 텐서 생성
predictions = torch.randn(3, 5)  # (배치 크기, 클래스 수)
targets = torch.tensor([1, 0, 4])  # 대상 (타깃) 값

# 손실 함수 정의
loss_fn = nn.CrossEntropyLoss()

# 손실 계산
loss = loss_fn(predictions, targets)
print(f"손실: {loss.item()}")
~~~
이 코드는 모델의 출력 predictions와 타깃 targets를 사용하여 손실 값을 계산합니다. CrossEntropyLoss는 모델의 출력에 softmax를 내부적으로 적용하므로, 추가로 softmax를 적용할 필요가 없습니다.
하지만 MSELoss(Mean Squared Error Loss)와 같은 다른 손실 함수에서는 모델의 출력에 softmax를 명시적으로 적용해야 할 수도 있습니다. 이 경우 다음과 같이 작성할 수 있습니다:

~~~python
# softmax 적용
softmax_output = F.softmax(predictions, dim=1)

# MSELoss 사용
mse_loss_fn = nn.MSELoss()
targets_one_hot = F.one_hot(targets, num_classes=5).float()  # 타깃을 원-핫 인코딩

# 손실 계산
mse_loss = mse_loss_fn(softmax_output, targets_one_hot)
print(f"MSE 손실: {mse_loss.item()}")
~~~
이 예제에서는 softmax를 적용한 후 MSELoss를 계산합니다. 따라서 손실 함수의 입력으로 softmax 출력을 사용할 수 있습니다.


## Saved Model
.\data\saved_model 에 다음 model 파일을 저장하였다.
이 파일의 Dimension은 다음과 같다.
~~~bash
autoencoder_model.pth
classifier_model.pth
~~~
- AutoEncoder
~~~bash
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 16, 16]             320
              ReLU-2           [-1, 32, 16, 16]               0
            Conv2d-3             [-1, 64, 8, 8]          18,496
              ReLU-4             [-1, 64, 8, 8]               0
            Conv2d-5            [-1, 128, 4, 4]          73,856
              ReLU-6            [-1, 128, 4, 4]               0
           Flatten-7                 [-1, 2048]               0
            Linear-8                    [-1, 2]           4,098
           Encoder-9                    [-1, 2]               0
           Linear-10                 [-1, 2048]           6,144
  ConvTranspose2d-11            [-1, 128, 8, 8]         147,584
             ReLU-12            [-1, 128, 8, 8]               0
  ConvTranspose2d-13           [-1, 64, 16, 16]          73,792
             ReLU-14           [-1, 64, 16, 16]               0
  ConvTranspose2d-15           [-1, 32, 32, 32]          18,464
             ReLU-16           [-1, 32, 32, 32]               0
           Conv2d-17            [-1, 1, 32, 32]             289
          Decoder-18            [-1, 1, 32, 32]               0
================================================================
Total params: 343,043
Trainable params: 343,043
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.14
Params size (MB): 1.31
Estimated Total Size (MB): 2.45
----------------------------------------------------------------
~~~
- classifier
~~~bash
----------------------------------------------------------------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                [-1, 1, 64]             192
              ReLU-2                [-1, 1, 64]               0
            Linear-3                [-1, 1, 64]           4,160
              ReLU-4                [-1, 1, 64]               0
            Linear-5                [-1, 1, 10]             650
================================================================
Total params: 5,002
Trainable params: 5,002
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.02
Estimated Total Size (MB): 0.02
----------------------------------------------------------------
~~~

## Comparison Performance 
VAE를 Fashion MNIST에 대하여 학습하는 실험을 하였을 때 다음과 같은 결과가 나왔다

|       GPU          | OP. time     |     OS    | EPOCH | 
|--------------------|--------------|-----------|-------|
| GeForce RTX 3050   |    112.96    | Windows 11|    10 |
| GeForce GTX 1080 Ti|     46.98    | Ubuntu 24 |    10 |   
| A100-PCI           |     61.28    | Ubuntu 20.04 | 10 |     

## Developing Memo
### sys.append(path)
- python에서 개발 디렉토리 구성을 위해 만들어 놓은 path 구조를 sys.path.append를 통해 등록하게 되면 이후 프로그램이 수행되어도 등록이 추가되거나 하지 않는다.

### summary와 forward
summary 를 구하기 위해 한번 pytorch는 전체 구조를 forward 하여 Dimension을 구한다.
따라서 model의 forward 함수가 호출된다.

