# 5주차 3세션 - 트랜스포머의 실제 구현 및 시각화

## 소개

이번 세션에서는 트랜스포머 아키텍처의 실제적인 측면에 초점을 맞출 것입니다. 어텐션 메커니즘이 어떻게 작동하는지 시각화하고 Python과 PyTorch를 사용하여 트랜스포머 모델을 구현하는 과정을 살펴볼 것입니다. 이 실습 접근 방식은 트랜스포머가 언어를 어떻게 처리하고 생성하는지에 대한 이해를 깊게 할 것입니다.

---

## 어텐션 메커니즘 시각화

어텐션이 내부적으로 어떻게 작동하는지 이해하는 것이 중요합니다. 시각화는 신경망의 "블랙박스" 특성을 이해하는 데 도움이 될 수 있습니다.

### 어텐션 점수 시각화

어텐션 점수는 출력의 각 부분을 생성할 때 모델이 입력의 다른 부분에 얼마나 집중하는지를 나타냅니다.

#### 예시: 어텐션 히트맵

다음과 같은 입력 문장이 있다고 가정해 봅시다:

- **입력**: "고양이가 매트 위에 앉았다."

이 문장을 처리할 때, 모델은 각 단어 쌍 사이의 어텐션 점수를 계산합니다.

#### 도표: 어텐션 히트맵 행렬

```
           고양이가  매트   위에   앉았다
        +-------------------------------------
 고양이가 | 0.1   0.2   0.3   0.1   0.2   0.1
    매트 | 0.2   0.1   0.3   0.1   0.2   0.1
    위에 | 0.1   0.2   0.1   0.3   0.2   0.1
  앉았다 | 0.1   0.1   0.2   0.1   0.3   0.2
```

_그림 1: 단어 간 어텐션 점수를 보여주는 어텐션 히트맵 예시._

#### 해석

- **높은 점수**: 단어 간 강한 어텐션을 나타냅니다.
- **대칭성**: 셀프 어텐션에서 행렬은 종종 대칭적입니다.
- **집중**: "가"와 같은 단어는 흔하기 때문에 낮은 어텐션 점수를 가질 수 있습니다.

### 어텐션 맵 해석

시각적 어텐션 맵은 입력의 어떤 부분이 출력에 영향을 미치는지 이해하는 데 도움이 됩니다.

#### 예시: 번역 작업

- **원문**: "Das ist ein Beispiel."
- **번역문**: "이것은 예시입니다."

#### 도표: 교차 어텐션 맵

```
             이것은  예시   입니다
        +-----------------------------
     Das | 0.7   0.1   0.1   0.1
     ist | 0.1   0.7   0.1   0.1
     ein | 0.1   0.1   0.7   0.1
Beispiel | 0.1   0.1   0.1   0.7
```

_그림 2: 원문과 번역문 사이의 교차 어텐션 맵._

#### 해석

- **정렬**: 높은 어텐션 점수는 단어를 그 번역과 정렬합니다.
- **중의성 해소**: 모델이 어순과 구문 차이를 어떻게 처리하는지 식별하는 데 도움이 됩니다.

---

## 트랜스포머의 실제 구현

이해를 공고히 하기 위해 간단한 트랜스포머 모델을 구축해 보겠습니다.

### 처음부터 간단한 트랜스포머 구축하기

PyTorch를 사용하여 주요 구성 요소에 초점을 맞춘 최소한의 트랜스포머 모델을 구현할 것입니다:

- **임베딩 계층**
- **위치 인코딩**
- **멀티헤드 어텐션**
- **피드포워드 네트워크**
- **인코더 및 디코더 계층**

### PyTorch로 구현하기

#### 필요한 라이브러리

```python
import torch
import torch.nn as nn
import math
```

### 코드 설명

#### 1. 위치 인코딩

위치 인코딩은 시퀀스 순서 정보를 주입합니다.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 형태: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x
```

#### 2. 멀티헤드 어텐션 모듈

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model은 n_heads로 나누어 떨어져야 합니다"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # 선형 투영
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)

        # 스케일드 닷-프로덕트 어텐션
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        x = torch.matmul(attention, v)

        # 헤드 연결
        x = x.transpose(1,2).contiguous().view(bs, -1, self.d_k * self.n_heads)
        return self.out_proj(x)
```

#### 3. 피드포워드 네트워크

```python
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
```

#### 4. 인코더 계층

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 셀프 어텐션
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        # 피드포워드 네트워크
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x
```

#### 5. 디코더 계층

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 마스크된 셀프 어텐션
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn_output)
        # 교차 어텐션
        cross_attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + cross_attn_output)
        # 피드포워드 네트워크
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        return x
```

#### 6. 트랜스포머 조립

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, num_layers=6):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 임베딩 및 위치 인코딩
        src = self.pos_encoder(self.src_embedding(src))
        tgt = self.pos_encoder(self.tgt_embedding(tgt))

        # 인코더
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # 디코더
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        output = self.fc_out(tgt)
        return output
```

#### 7. 마스크 생성

마스크는 훈련 중 모델이 미래 토큰에 주의를 기울이는 것을 방지하는 데 필수적입니다.

```python
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
```

---

## 고급 주제

### 트랜스포머 변형

#### BERT (Bidirectional Encoder Representations from Transformers)

- **아키텍처**: 트랜스포머의 인코더 부분만 사용합니다.
- **목적**: 마스크 언어 모델링(MLM)과 다음 문장 예측(NSP).
- **용도**: 질문 응답 및 감성 분석과 같은 문맥 이해가 필요한 작업에 탁월합니다.

#### GPT (Generative Pre-trained Transformer)

- **아키텍처**: 트랜스포머의 디코더 부분을 활용합니다.
- **목적**: 언어 모델링 (다음 단어 예측).
- **용도**: 텍스트 생성 작업에 효과적입니다.

### 트랜스포머의 응용

- **기계 번역**: 언어 간 고품질 번역.
- **텍스트 요약**: 문서의 간결한 요약 생성.
- **질문 응답**: 문맥을 바탕으로 답변 제공.
- **텍스트 생성**: 창의적인 글쓰기, 코드 생성 등.

---

## 선택적 코딩 과제 안내

### 간단한 어텐션 메커니즘 구현하기

#### 셀프 어텐션 함수

```python
def simple_self_attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output
```

#### 사용 예시

```python
# 샘플 입력 텐서
x = torch.rand(1, 5, 64)  # (배치 크기, 시퀀스 길이, d_model)

# 셀프 어텐션에서 Q, K, V가 모두 x라고 가정
output = simple_self_attention(x, x, x)
print(output.shape)  # (1, 5, 64) 출력 예상
```

---

## 결론

이번 세션에서는 트랜스포머를 이해하기 위한 실습 접근 방식을 취했습니다. 어텐션 메커니즘을 시각화하고 처음부터 트랜스포머 모델을 구현함으로써, 이제 이러한 모델들이 내부적으로 어떻게 작동하는지에 대해 더 깊이 있는 이해를 갖게 되었을 것입니다. 이러한 실제 경험은 앞으로 몇 주 동안 대규모 언어 모델과 API를 다루게 될 때 매우 귀중할 것입니다.

---

## 참고 문헌

- Vaswani 외, "Attention is All You Need" ([논문 링크](https://arxiv.org/abs/1706.03762))
- Jay Alammar의 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
