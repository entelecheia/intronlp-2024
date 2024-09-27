# 5주차 1세션 - 트랜스포머 소개

## 소개

이번 세션에서는 2017년 Vaswani 등이 소개한 획기적인 모델인 **트랜스포머 아키텍처**에 대해 깊이 알아볼 것입니다. 트랜스포머는 순환 또는 합성곱 계층에 의존하지 않고 장거리 의존성을 효과적으로 포착하여 자연어 처리(NLP)에 혁명을 일으켰습니다.

---

## 전통적인 RNN의 한계

순환 신경망(RNN)과 LSTM, GRU와 같은 변형들은 시퀀스 모델링 작업의 중추였습니다. 하지만 이들에게는 몇 가지 한계가 있습니다:

- **순차적 계산**: RNN은 시퀀스를 단계별로 처리하므로 계산을 병렬화하기 어렵습니다.
- **장기 의존성**: 그래디언트 소실 또는 폭발로 인해 시간적으로 멀리 떨어진 의존성을 포착하기 어렵습니다.
- **고정된 컨텍스트 창**: 멀리 있는 토큰의 컨텍스트를 효과적으로 통합하는 능력이 제한적입니다.

---

## 어텐션 메커니즘의 필요성

이러한 한계를 극복하기 위해 어텐션 메커니즘이 도입되었습니다:

- **병렬화**: 어텐션을 통해 모델이 모든 입력 토큰을 동시에 처리할 수 있습니다.
- **동적 컨텍스트**: 모델이 입력 시퀀스의 관련 부분에 동적으로 집중할 수 있습니다.
- **향상된 장거리 의존성**: 멀리 떨어진 토큰 간의 관계를 더 잘 포착합니다.

---

## 셀프 어텐션 메커니즘

셀프 어텐션 또는 인트라 어텐션은 같은 시퀀스 내의 서로 다른 위치를 연관시켜 시퀀스의 표현을 계산합니다.

### 스케일드 닷-프로덕트 어텐션

셀프 어텐션의 핵심 연산은 **스케일드 닷-프로덕트 어텐션**입니다.

#### 도표: 스케일드 닷-프로덕트 어텐션

![스케일드 닷-프로덕트 어텐션](./figs/attention-calculate.gif)

_그림 1: 스케일드 닷-프로덕트 어텐션 메커니즘._

### 수학적 공식화

쿼리 $ Q $, 키 $ K $, 값 $ V $가 주어졌을 때:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
$$

- $ Q $: 쿼리 행렬
- $ K $: 키 행렬
- $ V $: 값 행렬
- $ d_k $: 키 벡터의 차원

#### 코드 예시: 스케일드 닷-프로덕트 어텐션

```python
#torch는 PyTorch의 핵심 모듈로, 텐서 생성과 다양한 수학적 연산을 지원
import torch
#torch.nn.functional은 신경망에서 사용
import torch.nn.functional as F

# 스케일드 닷 프로덕트 어텐션 함수 정의
def scaled_dot_product_attention(Q, K, V):
    #Q(쿼리)의 마지막 차원의 크기를 가져오는 것으로, 이 값은 스케일링을 위한 계산에 사용됨
    d_k = Q.size(-1)
    #Q(쿼리)와 K(키)의 행렬의 곱을 계산 후 스케일링(내적 값들을 벡터 차원의 제곱근으로 나눔)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    #Softmax 함수를 통해 scores 값을 확률처럼 정규화(가중치의 합이 1이 되도록)
    attention_weights = F.softmax(scores, dim=-1)
    #어텐션 가중치와 밸류(V)를 곱한 결과를 반환
    return torch.matmul(attention_weights, V), attention_weights

# 임의의 Q, K, V 입력 값 설정 (batch_size=1, num_heads=1, sequence_length=3, embedding_dim=4)
Q = torch.rand(1, 3, 4)  # 쿼리 벡터
K = torch.rand(1, 3, 4)  # 키 벡터
V = torch.rand(1, 3, 4)  # 밸류 벡터

# 함수 호출 및 결과 출력
output, attention_weights = scaled_dot_product_attention(Q, K, V)

print("어텐션 출력:", output)
print("어텐션 가중치:", attention_weights)

```

---

## 멀티-헤드 어텐션

트랜스포머는 단일 어텐션 함수를 수행하는 대신 **멀티-헤드 어텐션**을 사용하여 다른 표현 부분 공간에서 정보를 포착합니다.

### 도표: 멀티-헤드 어텐션

![멀티-헤드 어텐션](./figs/attention-multihead.jpeg)

_그림 2: 여러 어텐션 헤드를 가진 멀티-헤드 어텐션 메커니즘._

### 수학적 공식화

각 어텐션 헤드 $ i $에 대해:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

모든 헤드를 연결:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

- $ W_i^Q, W_i^K, W_i^V $: $ i $번째 헤드의 매개변수 행렬
- $ W^O $: 출력 선형 변환 행렬

---

## "Attention is All You Need" 논문 개요

Vaswani 등은 합성곱이나 순환 계층 없이 전적으로 어텐션 메커니즘에 의존하는 트랜스포머 모델을 소개했습니다.

### 주요 기여

- **순환성 제거**: 병렬화를 가능하게 하고 훈련 시간을 단축합니다.
- **위치 인코딩**: 토큰의 상대적 또는 절대적 위치에 대한 정보를 주입합니다.
- **우수한 성능**: 기계 번역 작업에서 최고 수준의 결과를 달성했습니다.

---

## 트랜스포머 구성 요소 분석

### 인코더와 디코더 아키텍처

트랜스포머는 각각 여러 층으로 구성된 인코더와 디코더로 이루어져 있습니다.

#### 도표: 트랜스포머 아키텍처

![트랜스포머 아키텍처](./figs/transformer-architecture.png)

_그림 3: 트랜스포머 모델의 전체 아키텍처._

- **인코더 층**: 각 층은 두 개의 하위 층으로 구성됩니다:
  - 멀티-헤드 셀프 어텐션
  - 위치별 피드포워드 네트워크
- **디코더 층**: 각 층은 세 개의 하위 층으로 구성됩니다:
  - 마스크된 멀티-헤드 셀프 어텐션
  - 인코더 출력에 대한 멀티-헤드 어텐션
  - 위치별 피드포워드 네트워크

### 위치 인코딩

모델에 순환성이 없기 때문에, 시퀀스 순서 정보를 주입하기 위해 입력 임베딩에 위치 인코딩을 추가합니다.

#### 수학적 공식화

위치 $ pos $와 차원 $ i $에 대해:

$$
\text{PE}_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)
$$

$$
\text{PE}_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)
$$

#### 코드 예시: 위치 인코딩

```python
import torch
import math

#positional_encoding함수는 문장의 길이와 단어 임베딩의 차원 크기를 인자로 받는다
def positional_encoding(seq_len, d_model):
    #0으로 채워진 텐서 PE에 각 단어의 포지셔널 인코딩 값을 저장할 예정
    PE = torch.zeros(seq_len, d_model)
    #0부터 -1까지의 정수 배열을 생성, 차원을 늘려 2D 텐서로 만듬
    position = torch.arange(0, seq_len).unsqueeze(1)
    #포지셔널 인코딩에서 위치 정보가 반영되는 방식
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    #각 단어의 위치에 대해 사인, 코사인 값이 반영된 포지셔널 인코딩 생성
    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)
    #최종 인코딩 텐서 PE 반환
    return PE

# 함수를 호출해서 포지셔널 인코딩을 계산
seq_len = 10  # 문장의 길이
d_model = 16  # 임베딩 차원 수

pos_encoding = positional_encoding(seq_len, d_model)

print("포지셔널 인코딩 결과:")
print(pos_encoding)

```

### 피드포워드 네트워크

각 위치에 별도로 동일하게 적용되며, 두 개의 선형 변환과 그 사이의 ReLU 활성화로 구성됩니다.

#### 수학적 공식화

$$
\text{FFN}(x) = \text{max}(0, x W_1 + b_1) W_2 + b_2
$$

---

## 이전 아키텍처에 비해 트랜스포머의 장점

- **병렬화**: GPU를 효율적으로 사용할 수 있습니다.
- **장거리 의존성 포착 향상**: 두 위치 사이의 직접적인 연결.
- **훈련 시간 단축**: 효율적인 계산으로 인한 빠른 수렴.
- **확장성**: 더 큰 데이터셋과 모델로 쉽게 확장 가능.

---

## 결론

트랜스포머 아키텍처는 전통적인 RNN의 한계를 해결하고 어텐션 메커니즘을 활용하여 시퀀스를 더 효과적으로 모델링함으로써 NLP에 새로운 표준을 설정했습니다. 트랜스포머의 각 구성 요소를 이해하는 것은 대규모 언어 모델을 포함한 현대 NLP 모델이 어떻게 작동하는지 파악하는 데 중요합니다.

---

## 참고 문헌

- Vaswani 등, "Attention is All You Need" ([논문 링크](https://arxiv.org/abs/1706.03762))
- Jay Alammar의 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
