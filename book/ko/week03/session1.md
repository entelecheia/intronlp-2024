# 3주차 세션 1 - 언어 모델과 N-gram 소개

## 1. 언어 모델 소개

언어 모델은 단어 시퀀스에 확률을 할당하는 확률 모델입니다. 이들은 다음과 같은 많은 자연어 처리(NLP) 작업에서 중요한 역할을 합니다:

- 음성 인식
- 기계 번역
- 텍스트 생성
- 맞춤법 교정
- 감정 분석

언어 모델의 주요 목표는 언어에서 단어 시퀀스의 결합 확률 분포를 학습하는 것입니다.

### 1.1 형식적 정의

단어 시퀀스 W = (w₁, w₂, ..., wₙ)가 주어졌을 때, 언어 모델은 확률 P(W)를 계산합니다:

P(W) = P(w₁, w₂, ..., wₙ)

확률의 연쇄 법칙을 사용하여 이를 다음과 같이 분해할 수 있습니다:

P(W) = P(w₁) _ P(w₂|w₁) _ P(w₃|w₁,w₂) _ ... _ P(wₙ|w₁,w₂,...,wₙ₋₁)

### 1.2 언어 모델의 응용

1. **예측 텍스트**: 문장의 다음 단어 제안 (예: 스마트폰 키보드)
2. **기계 번역**: 유창하고 문법적으로 올바른 번역 보장
3. **음성 인식**: 유사한 소리의 구문 구별
4. **텍스트 생성**: 챗봇이나 콘텐츠 생성을 위한 인간과 유사한 텍스트 생성
5. **정보 검색**: 쿼리 의도를 이해하여 검색 결과 개선

## 2. N-gram 모델

N-gram 모델은 마르코프 가정을 기반으로 한 확률적 언어 모델의 한 유형입니다. 이들은 이전 N-1개 단어가 주어졌을 때 단어의 확률을 예측합니다.

### 2.1 N-gram의 유형

- 유니그램: P(wᵢ)
- 바이그램: P(wᵢ|wᵢ₋₁)
- 트라이그램: P(wᵢ|wᵢ₋₂,wᵢ₋₁)
- 4-gram: P(wᵢ|wᵢ₋₃,wᵢ₋₂,wᵢ₋₁)

### 2.2 마르코프 가정

N-gram 모델은 단어의 확률이 오직 이전 N-1개의 단어에만 의존한다는 단순화 가정을 합니다. 이를 마르코프 가정이라고 합니다.

트라이그램 모델의 경우:

P(wᵢ|w₁,w₂,...,wᵢ₋₁) ≈ P(wᵢ|wᵢ₋₂,wᵢ₋₁)

### 2.3 N-gram 확률 계산

N-gram 확률은 일반적으로 최대 우도 추정(MLE)을 사용하여 계산됩니다:

P(wₙ|wₙ₋ᵢ,...,wₙ₋₁) = count(wₙ₋ᵢ,...,wₙ) / count(wₙ₋ᵢ,...,wₙ₋₁)

바이그램 확률을 계산하는 간단한 함수를 구현해 보겠습니다:

```python
from collections import defaultdict
import nltk
nltk.download('punkt')

def calculate_bigram_prob(corpus):
    # 코퍼스 토큰화
    tokens = nltk.word_tokenize(corpus.lower())

    # 바이그램과 유니그램 카운트
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        bigram_counts[bigram] += 1
        unigram_counts[tokens[i]] += 1
    unigram_counts[tokens[-1]] += 1

    # 확률 계산
    bigram_probs = {}
    for bigram, count in bigram_counts.items():
        w1, w2 = bigram
        bigram_probs[bigram] = count / unigram_counts[w1]

    return bigram_probs

# 사용 예
corpus = "고양이가 매트 위에 앉았다. 개가 바닥에 앉았다."
bigram_probs = calculate_bigram_prob(corpus)

print("일부 바이그램 확률:")
for bigram, prob in list(bigram_probs.items())[:5]:
    print(f"P({bigram[1]}|{bigram[0]}) = {prob:.2f}")
```

### 2.4 N-gram 모델의 장단점

장점:

- 이해하고 구현하기 쉬움
- 계산 효율성이 높음
- 충분한 데이터가 있으면 많은 응용 프로그램에서 잘 작동함

단점:

- 제한된 문맥 (오직 이전 N-1개 단어만 고려)
- 데이터 희소성 (많은 가능한 N-gram이 학습 데이터에 나타나지 않음)
- 의미적으로 유사한 문맥에 대한 일반화 없음

## 3. 미등장 N-gram 처리: 스무딩 기법

N-gram 모델의 주요 문제 중 하나는 영확률 문제입니다: 학습 데이터에 나타나지 않은 N-gram에 0 확률이 할당됩니다. 스무딩 기법은 이 문제를 해결합니다.

### 3.1 라플라스 (Add-1) 스무딩

라플라스 스무딩은 모든 N-gram 카운트에 1을 더합니다:

P(wₙ|wₙ₋ᵢ,...,wₙ₋₁) = (count(wₙ₋ᵢ,...,wₙ) + 1) / (count(wₙ₋ᵢ,...,wₙ₋₁) + V)

여기서 V는 어휘 크기입니다.

바이그램에 대한 라플라스 스무딩을 구현해 보겠습니다:

```python
def laplace_smoothed_bigram_prob(corpus):
    tokens = nltk.word_tokenize(corpus.lower())
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        bigram_counts[bigram] += 1
        unigram_counts[tokens[i]] += 1
    unigram_counts[tokens[-1]] += 1

    V = len(set(tokens))  # 어휘 크기

    smoothed_probs = {}
    for w1 in unigram_counts:
        for w2 in unigram_counts:
            bigram = (w1, w2)
            smoothed_probs[bigram] = (bigram_counts[bigram] + 1) / (unigram_counts[w1] + V)

    return smoothed_probs

# 사용 예
smoothed_probs = laplace_smoothed_bigram_prob(corpus)

print("일부 라플라스 스무딩된 바이그램 확률:")
for bigram, prob in list(smoothed_probs.items())[:5]:
    print(f"P({bigram[1]}|{bigram[0]}) = {prob:.4f}")
```

### 3.2 기타 스무딩 기법

- Add-k 스무딩: 라플라스와 유사하지만 k < 1을 더함
- 굿-튜링 스무딩: 한 번 나타난 N-gram의 빈도를 기반으로 미등장 N-gram의 확률을 추정
- 크네서-네이 스무딩: 더 일반적인 N-gram의 빈도를 사용하여 특정 N-gram의 확률을 추정

## 4. 언어 모델 평가: 퍼플렉시티

퍼플렉시티는 언어 모델을 평가하는 일반적인 지표입니다. 확률 분포가 샘플을 얼마나 잘 예측하는지 측정합니다. 낮은 퍼플렉시티는 더 나은 성능을 나타냅니다.

테스트 세트 W = w₁w₂...wₙ에 대해 퍼플렉시티는 다음과 같습니다:

PP(W) = P(w₁w₂...wₙ)^(-1/n)

바이그램 모델의 퍼플렉시티를 계산하는 함수를 구현해 보겠습니다:

```python
import math

def calculate_perplexity(test_corpus, bigram_probs):
    tokens = nltk.word_tokenize(test_corpus.lower())
    n = len(tokens)
    log_probability = 0

    for i in range(n - 1):
        bigram = (tokens[i], tokens[i+1])
        if bigram in bigram_probs:
            log_probability += math.log2(bigram_probs[bigram])
        else:
            log_probability += math.log2(1e-10)  # 미등장 바이그램에 대한 작은 확률

    perplexity = 2 ** (-log_probability / n)
    return perplexity

# 사용 예
test_corpus = "고양이가 바닥에 앉았다."
perplexity = calculate_perplexity(test_corpus, bigram_probs)
print(f"퍼플렉시티: {perplexity:.2f}")
```

## 5. N-gram 모델 시각화

N-gram 모델을 더 잘 이해하기 위해 바이그램 전이를 시각화해 보겠습니다:

```python
import matplotlib.pyplot as plt
import networkx as nx

def visualize_bigrams(bigram_probs, top_n=5):
    G = nx.DiGraph()

    for (w1, w2), prob in bigram_probs.items():
        G.add_edge(w1, w2, weight=prob)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=3000, font_size=10, font_weight='bold')

    edge_labels = {(w1, w2): f"{prob:.2f}" for (w1, w2), prob in bigram_probs.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("바이그램 전이 확률")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_bigrams(bigram_probs)
```

이 시각화는 우리의 바이그램 모델에서 단어 간 전이를 보여주며, 간선의 가중치는 전이 확률을 나타냅니다.

## 결론

이 세션에서는 언어 모델의 개념을 소개하고 N-gram 모델, 특히 바이그램에 초점을 맞추어 살펴보았습니다. 우리가 다룬 내용은 다음과 같습니다:

1. 언어 모델의 정의와 응용
2. N-gram 모델과 마르코프 가정
3. N-gram 확률 계산
4. 미등장 N-gram 처리를 위한 스무딩 기법
5. 퍼플렉시티를 사용한 언어 모델 평가
6. N-gram 모델 시각화

다음 세션에서는 통계적 언어 모델링의 더 고급 주제를 탐구하고 언어 모델링에 대한 신경망 접근법을 살펴보기 시작할 것입니다.
