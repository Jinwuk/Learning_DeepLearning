## Question
평균 $\mu$, 분산이 $\sigma^2$ 인 Normal 분포를 $\mathcal{N}(\mu, \sigma^2)$ 이라 할 때 
$\mathcal{N}(0, 1)$과의 KL Divergence는 어떻게 나타나는가? 

## Answer 
평균이 $\mu$, 분산이 $\sigma^2$인 정규 분포 $\mathcal{N}(\mu, \sigma^2)$를 $P$라 하고, 표준 정규 분포 $\mathcal{N}(0, 1)$를 $Q$라고 할 때, $P$와 $Q$ 사이의 KL Divergence는 다음과 같이 계산됩니다.

**1. KL Divergence의 정의**

KL Divergence는 다음과 같이 정의됩니다.

$$D_{KL}(P||Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx$$

여기서 $p(x)$는 $P$의 확률 밀도 함수이고, $q(x)$는 $Q$의 확률 밀도 함수입니다.

**2. 정규 분포의 확률 밀도 함수**

정규 분포 $\mathcal{N}(\mu, \sigma^2)$의 확률 밀도 함수는 다음과 같습니다.

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

따라서, $P = \mathcal{N}(\mu, \sigma^2)$의 확률 밀도 함수는

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

이고, $Q = \mathcal{N}(0, 1)$의 확률 밀도 함수는

$$q(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right)$$

입니다.

**3. KL Divergence 계산**

이제 KL Divergence를 계산해 봅시다.

$$D_{KL}(P||Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx$$

$$= \int_{-\infty}^{\infty} p(x) \left[ \log p(x) - \log q(x) \right] dx$$

먼저 $\log p(x)$와 $\log q(x)$를 계산합니다.

$$\log p(x) = \log \left( \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \right) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$$

$$\log q(x) = \log \left( \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right) \right) = -\frac{1}{2}\log(2\pi) - \frac{x^2}{2}$$

따라서,

$$\log \frac{p(x)}{q(x)} = \log p(x) - \log q(x) = -\frac{1}{2}\log(\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2} + \frac{x^2}{2}$$

이제 KL Divergence를 계산합니다.

$$D_{KL}(P||Q) = \int_{-\infty}^{\infty} p(x) \left[ -\frac{1}{2}\log(\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2} + \frac{x^2}{2} \right] dx$$

$$= -\frac{1}{2}\log(\sigma^2) \int_{-\infty}^{\infty} p(x) dx - \frac{1}{2\sigma^2} \int_{-\infty}^{\infty} (x-\mu)^2 p(x) dx + \frac{1}{2} \int_{-\infty}^{\infty} x^2 p(x) dx$$

각 항을 계산하면 다음과 같습니다.

*   $\int_{-\infty}^{\infty} p(x) dx = 1$ (확률 밀도 함수의 적분은 1)
*   $\int_{-\infty}^{\infty} (x-\mu)^2 p(x) dx = \sigma^2$ ($P$의 분산)
*   $\int_{-\infty}^{\infty} x^2 p(x) dx = Var(X) + E[X]^2 = \sigma^2 + \mu^2$ ($P$의 2차 모멘트)

따라서,

$$D_{KL}(P||Q) = -\frac{1}{2}\log(\sigma^2) - \frac{1}{2\sigma^2} \cdot \sigma^2 + \frac{1}{2} (\sigma^2 + \mu^2)$$

$$= -\frac{1}{2}\log(\sigma^2) - \frac{1}{2} + \frac{1}{2}\sigma^2 + \frac{1}{2}\mu^2$$

$$= \frac{1}{2} \left( -\log(\sigma^2) - 1 + \sigma^2 + \mu^2 \right)$$

$$= \frac{1}{2} \left( \mu^2 + \sigma^2 - \log(\sigma^2) - 1 \right)$$

**결론**

따라서, $\mathcal{N}(\mu, \sigma^2)$와 $\mathcal{N}(0, 1)$ 사이의 KL Divergence는 다음과 같습니다.

$$D_{KL}(\mathcal{N}(\mu, \sigma^2) || \mathcal{N}(0, 1)) = \frac{1}{2} \left( \mu^2 + \sigma^2 - \log(\sigma^2) - 1 \right)$$

또는 $\log$를 자연로그로 사용한다면,

$$D_{KL}(\mathcal{N}(\mu, \sigma^2) || \mathcal{N}(0, 1)) = \frac{1}{2} \left( \mu^2 + \sigma^2 - \ln(\sigma^2) - 1 \right)$$