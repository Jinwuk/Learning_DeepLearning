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
