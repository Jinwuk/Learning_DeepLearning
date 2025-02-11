## Question
pytorch를 사용하여 auto encoder 신경망을 구성할 때, encoder 의 출력, 즉, Latent vector만 따로 빼내고자 한다면 어떻게 코드를 구성해야 하는가?
예제 python 코드를 보여주기 바란다.  

## Answer 
PyTorch를 사용하여 Autoencoder를 구성하고 latent vector만 추출하는 방법은 여러가지가 있지만, 가장 간단한 방법은 `forward` 메서드에서 latent vector를 return하는 것입니다.  다음은 예시 코드입니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() # Output should be between 0 and 1 for many applications
        )

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Decode (optional, if you only need the latent vector, you can omit this)
        decoded = self.decoder(encoded)
        return encoded, decoded # Return both encoded and decoded


# Example usage
input_dim = 784  # Example: MNIST images (28x28)
hidden_dim = 256
latent_dim = 64

autoencoder = Autoencoder(input_dim, hidden_dim, latent_dim)

# Sample input
input_tensor = torch.randn(1, input_dim)  # Batch size of 1

# Get latent vector
latent_vector, decoded = autoencoder(input_tensor)

print("Latent vector shape:", latent_vector.shape) # Output should be (1, 64)
print("Decoded shape:", decoded.shape) # Output should be (1, 784)

#If you only need the latent vector, you can modify the forward function like this:
class Autoencoder_latent_only(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder_latent_only, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        return encoded

autoencoder_latent = Autoencoder_latent_only(input_dim, hidden_dim, latent_dim)
latent_vector_only = autoencoder_latent(input_tensor)
print("Latent vector only shape:", latent_vector_only.shape) # Output should be (1, 64)

```

이 코드는  Autoencoder 클래스를 정의하고, `forward` 메서드에서 인코더의 출력인 latent vector와 디코더의 출력을 모두 반환합니다.  `latent_vector` 변수에 latent vector가 저장됩니다.  두번째 예시는 latent vector만 반환하는 Autoencoder를 보여줍니다.  필요에 따라 `forward` 메서드를 수정하여 원하는 출력만 반환하도록 할 수 있습니다.  `nn.Sigmoid()` 와 같은 활성화 함수는 사용하는 데이터와 목적에 따라 조정해야 합니다.  예를 들어, MNIST 이미지와 같은 0과 1 사이의 값을 갖는 데이터에는 `Sigmoid`를, 다른 데이터에는 다른 활성화 함수를 사용하는 것이 적절할 수 있습니다.
