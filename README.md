# Learning_DeepLearning
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
그러므로 
