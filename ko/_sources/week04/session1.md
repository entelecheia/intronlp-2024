# 4주차 세션 1 - 단어 임베딩과 Word2Vec 소개

## 1. 단어 임베딩 소개

단어 임베딩은 연속적인 벡터 공간에서 단어를 밀집 벡터로 표현한 것입니다. 이는 현대 자연어 처리(NLP)의 기본 개념으로, 전통적인 단어 표현 방법에 비해 여러 가지 장점을 제공합니다.

### 1.1 전통적인 단어 표현의 한계

원-핫 인코딩과 같은 전통적인 방법은 단어를 희소 벡터로 표현합니다:

```python
vocab = ["고양이", "개", "쥐"]
one_hot_고양이 = [1, 0, 0]
one_hot_개 = [0, 1, 0]
one_hot_쥐 = [0, 0, 1]
```

이 접근 방식의 문제점:

- 어휘 크기에 따라 차원이 증가함
- 의미론적 정보를 포착하지 못함
- 모든 단어가 서로 동일한 거리에 있음

### 1.2 단어 임베딩의 아이디어

단어 임베딩은 분포 가설에 기반합니다: 비슷한 맥락에서 발생하는 단어들은 비슷한 의미를 가지는 경향이 있습니다.

주요 특성:

- 밀집 벡터 표현 (예: 100-300 차원)
- 의미론적, 구문론적 정보를 포착
- 유사한 단어들이 벡터 공간에서 가깝게 위치함

이 개념을 시각화해 봅시다:

```python
import matplotlib.pyplot as plt

def plot_vectors(vectors, labels):
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(vectors):
        plt.scatter(x, y)
        plt.annotate(labels[i], (x, y))
    plt.title("2D 공간의 단어 벡터")
    plt.xlabel("차원 1")
    plt.ylabel("차원 2")
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.show()

# 2D 단어 벡터 예시
vectors = [(0.2, 0.3), (0.3, 0.2), (0.5, 0.7), (-0.3, -0.2), (-0.2, -0.3)]
labels = ["고양이", "개", "동물", "자동차", "트럭"]

plot_vectors(vectors, labels)
```

이 시각화는 의미론적으로 유사한 단어들("고양이", "개", "동물")이 벡터 공간에서 관련 없는 단어들("자동차", "트럭")에 비해 서로 더 가깝게 위치하는 것을 보여줍니다.

## 2. Word2Vec

Word2Vec은 2013년 Mikolov 등이 소개한 가장 인기 있는 단어 임베딩 모델 중 하나입니다. Continuous Bag of Words (CBOW)와 Skip-gram 두 가지 방식이 있습니다.

### 2.1 CBOW 아키텍처

CBOW는 주변 단어들을 통해 목표 단어를 예측합니다.

```{mermaid}
graph TD
    A[입력 주변 단어들] --> B[입력 레이어]
    B --> C[은닉 레이어]
    C --> D[출력 레이어]
    D --> E[목표 단어]
```

### 2.2 Skip-gram 아키텍처

Skip-gram은 목표 단어를 통해 주변 단어들을 예측합니다.

```{mermaid}
graph TD
    A[입력 목표 단어] --> B[입력 레이어]
    B --> C[은닉 레이어]
    C --> D[출력 레이어]
    D --> E[주변 단어들]
```

### 2.3 학습 과정

Word2Vec은 신경망을 사용하여 단어 임베딩을 학습합니다:

1. 단어 벡터를 무작위로 초기화
2. 텍스트 코퍼스 위로 윈도우를 슬라이드
3. 각 윈도우에 대해:
   - CBOW: 주변 단어들을 사용하여 중심 단어 예측
   - Skip-gram: 중심 단어를 사용하여 주변 단어들 예측
4. 예측 오류를 기반으로 단어 벡터 업데이트
5. 수렴할 때까지 반복

### 2.4 Gensim을 사용한 Word2Vec 구현

Gensim을 사용하여 간단한 Word2Vec 모델을 구현해 봅시다:

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# 예시 코퍼스
corpus = [
    "빠른 갈색 여우가 게으른 개를 뛰어넘습니다",
    "고양이와 개는 천적입니다",
    "개가 고양이를 나무 위로 쫓아갑니다"
]

# 코퍼스 토큰화
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Word2Vec 모델 학습
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# 유사한 단어 찾기
similar_words = model.wv.most_similar("개", topn=3)
print("'개'와 가장 유사한 단어들:", similar_words)

# 단어 유추 수행
result = model.wv.most_similar(positive=['개', '나무'], negative=['고양이'], topn=1)
print("개 - 고양이 + 나무 =", result[0][0])

# 단어 벡터 얻기
dog_vector = model.wv['개']
print("'개'의 벡터:", dog_vector[:5])  # 처음 5차원만 표시
```

### 2.5 단어 임베딩 시각화

t-SNE와 같은 차원 축소 기법을 사용하여 단어 임베딩을 시각화할 수 있습니다:

```python
from sklearn.manifold import TSNE
import numpy as np

def plot_embeddings(model, words):
    word_vectors = [model.wv[word] for word in words]
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        x, y = embeddings_2d[i, :]
        plt.scatter(x, y)
        plt.annotate(word, (x, y))
    plt.title("단어 임베딩 시각화")
    plt.show()

# 선택된 단어들의 임베딩 시각화
words_to_plot = ["개", "고양이", "여우", "나무", "빠른", "게으른", "뛰어넘다", "쫓아가다"]
plot_embeddings(model, words_to_plot)
```

## 3. Word2Vec의 장점

1. 의미론적 관계를 포착함
2. 대규모 코퍼스에 대해 효율적으로 학습 가능
3. 어휘 외 단어 처리 가능 (하위 단어 정보 사용)
4. 다양한 하위 NLP 작업에 유용함

## 4. 한계 및 고려사항

1. 대량의 학습 데이터 필요
2. 다의어 (여러 의미를 가진 단어) 처리 불가
3. 정적 임베딩 (문맥을 고려하지 않음)
4. 학습 데이터의 편향이 임베딩에 반영될 수 있음

## 결론

Word2Vec을 비롯한 단어 임베딩은 단어의 풍부하고 밀집된 표현을 제공함으로써 많은 NLP 작업을 혁신했습니다. 다음 세션에서는 GloVe와 FastText와 같은 다른 단어 임베딩 기법들을 살펴보고, 더 고급 응용에 대해 논의할 것입니다.

## 연습 문제

1. 더 큰 코퍼스(예: 뉴스 기사 모음)에 대해 Word2Vec 모델을 학습시킵니다.
2. 다양한 하이퍼파라미터(vector_size, window, min_count)를 실험해보고 결과 임베딩에 미치는 영향을 관찰합니다.
3. 학습된 모델을 사용하여 간단한 단어 유추 작업을 구현합니다.
4. 특정 도메인(예: 스포츠, 기술)과 관련된 단어 집합에 대한 임베딩을 시각화하고 모델이 포착한 관계를 분석합니다.

```python
# 여기에 코드를 작성하세요
```

이 연습을 통해 Word2Vec에 대한 실제적인 경험을 쌓고 단어 임베딩에 대한 이해를 심화할 수 있을 것입니다.
