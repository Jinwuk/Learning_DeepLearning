torch.compile 
===
[toc]

## torch.compile 개략

`torch.compile`은 PyTorch의 모델과 연산 그래프를 컴파일하는 데 사용되는 함수입니다. 이 함수는 모델을 더 효율적으로 실행하기 위해 JIT(Just-In-Time) 컴파일을 통해 최적화된 버전을 생성합니다. `torch.compile`은 PyTorch의 고성능 컴퓨팅 기능을 활용하여 모델의 실행 속도를 향상시킬 수 있습니다.

다음은 `torch.compile`을 사용하는 기본적인 방법입니다:

1. **모델 정의**:
   `torch.compile`을 사용하려면 먼저 PyTorch 모델을 정의해야 합니다.
   ```python
   import torch
   import torch.nn as nn

   class SimpleModel(nn.Module):
       def __init__(self):
           super(SimpleModel, self).__init__()
           self.fc = nn.Linear(10, 1)

       def forward(self, x):
           return self.fc(x)
   ```

2. **모델 컴파일**:
   정의된 모델을 `torch.compile` 함수를 사용하여 컴파일할 수 있습니다.
   ```python
   model = SimpleModel()
   compiled_model = torch.compile(model)
   ```

3. **모델 실행**:
   컴파일된 모델을 사용하여 데이터를 처리할 수 있습니다.
   ```python
   input_data = torch.randn(5, 10)
   output = compiled_model(input_data)
   print(output)
   ```

`torch.compile`을 사용하면 PyTorch 모델의 성능을 크게 향상시킬 수 있으며, 특히 큰 모델이나 복잡한 연산 그래프에서 더 큰 이점을 얻을 수 있습니다.

도움이 되셨기를 바랍니다! 추가로 궁금한 점이 있으시면 언제든지 말씀해 주세요. 😊

## torch.compile Parameter

`torch.compile(model, backend="eager")`에서 `eager`는 PyTorch의 이그제큐션 모드를 의미합니다. `eager` 모드는 즉시 실행 모드로, 모든 연산이 즉시 실행되고 결과가 반환됩니다. 이 모드는 디버깅과 개발 중에 유용하며, 코드가 실행되는 순서대로 결과를 볼 수 있습니다.

이제 `torch.compile`의 입력 파라미터에 대해 설명하겠습니다:

1. **`model`**: 컴파일하려는 PyTorch 모델입니다. 이는 `torch.nn.Module` 클래스의 인스턴스여야 합니다.

2. **`backend`**: 컴파일 모드를 지정합니다. 기본적으로 `eager` 모드가 사용되며, 다른 옵션으로는 `graph` 모드가 있습니다.
   - `eager`: 즉시 실행 모드로, 코드가 실행되는 순서대로 결과를 반환합니다. 디버깅과 개발에 유용합니다.
   - `graph`: 그래프 실행 모드로, 전체 연산 그래프를 컴파일하여 최적화된 실행을 제공합니다. 성능 향상에 유리합니다.

3. **`mode`**: 컴파일 모드를 지정합니다. 기본값은 `training` 모드이며, `evaluation` 모드로 설정할 수도 있습니다. 각 모드에 따라 컴파일 최적화가 다르게 적용됩니다.
   - `training`: 훈련 모드로, 모델의 학습을 최적화합니다.
   - `evaluation`: 평가 모드로, 모델의 예측을 최적화합니다.

4. **`optimization_level`**: 최적화 수준을 지정합니다. 기본값은 `O1`이며, `O0`(최적화 없음)부터 `O3`(최대 최적화)까지 설정할 수 있습니다. 최적화 수준에 따라 컴파일러가 적용하는 최적화 기법이 달라집니다.

5. **`device`**: 모델이 실행될 디바이스를 지정합니다. 기본값은 `cpu`이며, `cuda`를 선택하면 GPU에서 실행됩니다.

예시:
```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
compiled_model = torch.compile(model, backend="eager", mode="training", optimization_level="O1", device="cuda")
```

이 코드는 `torch.compile` 함수를 사용하여 지정된 파라미터에 따라 모델을 컴파일합니다. 각 파라미터는 모델의 실행 방식과 최적화에 영향을 미칩니다.

도움이 되셨기를 바랍니다! 추가로 궁금한 점이 있으시면 언제든지 말씀해 주세요. 😊