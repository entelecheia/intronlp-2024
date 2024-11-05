# 3주차 세션 2: 고급 통계적 언어 모델

## 1. 고급 N-gram 기법

### 1.1 보간법과 백오프

보간법과 백오프는 다른 차수의 N-gram 모델을 결합하여 성능을 향상시키는 기법입니다.

#### 선형 보간법

선형 보간법은 다른 차수의 N-gram에서 확률을 결합합니다:

P(wᵢ|wᵢ₋₂wᵢ₋₁) = λ₃P(wᵢ|wᵢ₋₂wᵢ₋₁) + λ₂P(wᵢ|wᵢ₋₁) + λ₁P(wᵢ)

여기서 λ₁ + λ₂ + λ₃ = 1

```python
def interpolate(trigram_prob, bigram_prob, unigram_prob, lambda1, lambda2, lambda3):
    return lambda3 * trigram_prob + lambda2 * bigram_prob + lambda1 * unigram_prob

# 사용 예
trigram_prob = 0.002
bigram_prob = 0.01
unigram_prob = 0.1
lambda1, lambda2, lambda3 = 0.1, 0.3, 0.6

interpolated_prob = interpolate(trigram_prob, bigram_prob, unigram_prob, lambda1, lambda2, lambda3)
print(f"보간된 확률: {interpolated_prob:.4f}")
```

#### Katz 백오프

Katz 백오프는 가능한 경우 고차 N-gram을 사용하지만, 미등장 시퀀스에 대해서는 저차 N-gram으로 "백오프"합니다:

```python
def katz_backoff(trigram, bigram, unigram, trigram_counts, bigram_counts, unigram_counts, k=0.5):
    if trigram in trigram_counts and trigram_counts[trigram] > k:
        return trigram_counts[trigram] / bigram_counts[trigram[:2]]
    elif bigram in bigram_counts:
        alpha = (1 - k * len([t for t in trigram_counts if t[:2] == bigram])) / (1 - k * len([b for b in bigram_counts if b[0] == bigram[0]]))
        return alpha * (bigram_counts[bigram] / unigram_counts[bigram[0]])
    else:
        return unigram_counts[unigram] / sum(unigram_counts.values())

# 사용 예 (단순화)
trigram_counts = {('the', 'cat', 'sat'): 2, ('cat', 'sat', 'on'): 3}
bigram_counts = {('the', 'cat'): 5, ('cat', 'sat'): 4, ('sat', 'on'): 3}
unigram_counts = {'the': 10, 'cat': 8, 'sat': 6, 'on': 5}

prob = katz_backoff(('the', 'cat', 'sat'), ('cat', 'sat'), 'sat', trigram_counts, bigram_counts, unigram_counts)
print(f"Katz 백오프 확률: {prob:.4f}")
```

### 1.2 스킵-그램 모델

스킵-그램 모델은 N-gram 시퀀스에 갭을 허용하여 더 긴 범위의 의존성을 포착합니다:

```python
from collections import defaultdict
import nltk

nltk.download('punkt') #nltk 다운로드 추가

def create_skipgram_model(text, n=3, k=1):
    tokens = nltk.word_tokenize(text.lower())
    skipgram_counts = defaultdict(int)
    context_counts = defaultdict(int)

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        skipgram_counts[ngram] += 1

        for j in range(1, k+1):
            if i+n+j <= len(tokens):
                skipgram = tuple(tokens[i:i+n-1] + [tokens[i+n+j-1]])
                skipgram_counts[skipgram] += 1

        context = tuple(tokens[i:i+n-1])
        context_counts[context] += 1

    skipgram_probs = {gram: count / context_counts[gram[:-1]]
                      for gram, count in skipgram_counts.items()}

    return skipgram_probs

# 사용 예
text = "빠른 갈색 여우가 게으른 개를 뛰어넘었다"
skipgram_probs = create_skipgram_model(text, n=3, k=1)

print("일부 스킵-그램 확률:")
for gram, prob in list(skipgram_probs.items())[:5]:
    print(f"P({gram[-1]}|{' '.join(gram[:-1])}) = {prob:.2f}")
```

## 2. 클래스 기반 언어 모델

클래스 기반 모델은 단어를 클래스로 그룹화하여 매개변수 수를 줄이고 데이터 희소성 문제를 해결합니다:

```python
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

def create_class_based_model(sentences, num_classes=10):
    # Word2Vec 모델 학습
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # 단어 벡터 클러스터링
    word_vectors = [model.wv[word] for word in model.wv.key_to_index]
    kmeans = KMeans(n_clusters=num_classes)
    kmeans.fit(word_vectors)

    # 단어를 클래스에 할당
    word_classes = {word: kmeans.predict([model.wv[word]])[0] for word in model.wv.key_to_index}

    # 클래스 전이 카운트
    class_transitions = defaultdict(int)
    class_counts = defaultdict(int)

    for sentence in sentences:
        for i in range(len(sentence) - 1):
            c1, c2 = word_classes[sentence[i]], word_classes[sentence[i+1]]
            class_transitions[(c1, c2)] += 1
            class_counts[c1] += 1

    # 전이 확률 계산
    class_transition_probs = {(c1, c2): count / class_counts[c1]
                              for (c1, c2), count in class_transitions.items()}

    return word_classes, class_transition_probs

# 사용 예
sentences = [
    ['빠른', '갈색', '여우가', '게으른', '개를', '뛰어넘었다'],
    ['한', '빠른', '갈색', '개가', '게으른', '고양이에게', '짖었다']
]

word_classes, class_transition_probs = create_class_based_model(sentences)

print("단어 클래스:")
for word, class_ in list(word_classes.items())[:5]:
    print(f"{word}: 클래스 {class_}")

print("\n일부 클래스 전이 확률:")
for (c1, c2), prob in list(class_transition_probs.items())[:5]:
    print(f"P(클래스 {c2}|클래스 {c1}) = {prob:.2f}")
```

## 3. 최대 엔트로피 언어 모델

최대 엔트로피(MaxEnt) 모델, 또는 로그선형 모델이라고도 불리는 이 모델은 다양한 특성을 언어 모델에 통합할 수 있게 합니다:

```python
import numpy as np
from scipy.optimize import minimize


def maxent_model(features, labels):
    def neg_log_likelihood(weights):
        # 점수 계산 (로짓)
        scores = np.dot(features, weights)
        # 시그모이드 확률 계산
        probs = 1 / (1 + np.exp(-scores))
        # 음의 로그 우도 계산 (이진 분류)
        return -np.sum(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

    # 초기 가중치 설정
    initial_weights = np.zeros(features.shape[1])
    # 최적화 함수 사용
    result = minimize(neg_log_likelihood, initial_weights, method='L-BFGS-B')
    return result.x

# 사용 예
features = np.array([
    [1, 0, 1],  # "고양이"에 대한 특성 벡터
    [1, 1, 0],  # "개"에 대한 특성 벡터
    [0, 1, 1]   # "한 마리의 개"에 대한 특성 벡터
])
labels = np.array([0, 1, 1])  # 0: "고양이", 1: "개"

weights = maxent_model(features, labels)
print("MaxEnt 모델 가중치:", weights)
```

## 4. 신경망 언어 모델 소개

신경망 언어 모델은 신경망을 사용하여 단어의 분산 표현을 학습하고 확률을 예측합니다.

### 4.1 단어 임베딩

단어 임베딩은 의미적 관계를 포착하는 단어의 밀집 벡터 표현입니다:

```python
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.font_manager as fm
import sys #한국어 폰트 깨짐 부분 해결

# Google Colab 환경에서 실행 중인지 확인
if 'google.colab' in sys.modules:
    # debconf를 Noninteractive 모드로 설정
    !echo 'debconf debconf/frontend select Noninteractive' | \
    debconf-set-selections

    # fonts-nanum 패키지를 설치
    !sudo apt-get -qq -y install fonts-nanum

    # Matplotlib의 폰트 매니저 가져오기
    import matplotlib.font_manager as fm

    # 나눔 폰트의 시스템 경로 찾기
    font_files = fm.findSystemFonts(fontpaths=['/usr/share/fonts/truetype/nanum'])

    # 찾은 각 나눔 폰트를 Matplotlib 폰트 매니저에 추가
    for fpath in font_files:
        fm.fontManager.addfont(fpath)

# 한글 폰트 설정
def set_korean_font():
    # 나눔 고딕 폰트를 사용할 경우
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 경로는 폰트 설치 위치에 따라 다를 수 있음
    fontprop = fm.FontProperties(fname=font_path, size=10)
    plt.rc('font', family=fontprop.get_name())
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호가 깨지는 것을 방지

def train_word2vec(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def visualize_embeddings(model, words):
    word_vectors = []
    valid_words = []
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])
            valid_words.append(word)
        else:
            print(f"'{word}'가 어휘에 없습니다.")

    if word_vectors:
        # 리스트를 numpy 배열로 변환
        word_vectors = np.array(word_vectors)

        # perplexity를 샘플 수보다 작게 설정
        perplexity_value = min(len(word_vectors) - 1, 5)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
        embeddings_2d = tsne.fit_transform(word_vectors)

        plt.figure(figsize=(10, 8))
        for i, word in enumerate(valid_words):
            x, y = embeddings_2d[i]
            plt.scatter(x, y)
            plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points')
        plt.title("단어 임베딩 시각화")
        plt.xlabel("t-SNE 특성 0")
        plt.ylabel("t-SNE 특성 1")
        plt.show()
    else:
        print("시각화할 유효한 단어가 없습니다.")

# 한글 폰트 설정 함수 호출
set_korean_font()

# 사용 예
sentences = [
    ['빠른', '갈색', '여우가', '게으른', '개를', '뛰어넘었다'],
    ['한', '빠른', '갈색', '개가', '게으른', '고양이에게', '짖었다']
]

model = train_word2vec(sentences)
words_to_plot = ['빠른', '갈색', '여우', '개', '게으른', '고양이']
visualize_embeddings(model, words_to_plot)
```

### 4.2 간단한 피드포워드 신경망 언어 모델

TensorFlow를 사용한 기본적인 피드포워드 신경망 언어 모델:

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Input
from tensorflow.keras.models import Model

def create_ffnn_lm(vocab_size, embedding_dim, context_size):
    inputs = Input(shape=(context_size,))
    embedding = Embedding(vocab_size, embedding_dim)(inputs)
    flattened = tf.keras.layers.Flatten()(embedding)
    hidden = Dense(128, activation='relu')(flattened)
    output = Dense(vocab_size, activation='softmax')(hidden)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

# 사용 예 (단순화)
vocab_size = 1000
embedding_dim = 50
context_size = 3

model = create_ffnn_lm(vocab_size, embedding_dim, context_size)
model.summary()
```

## 5. 통계적 및 신경망 언어 모델 비교

트라이그램 모델과 간단한 신경망 언어 모델의 퍼플렉시티를 비교해 보겠습니다:

```python
import numpy as np
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras import layers, models

# 트라이그램 퍼플렉시티 계산 함수
def calculate_trigram_perplexity(test_data, trigram_probs):
    log_prob = 0
    n = 0
    for sentence in test_data:
        for i in range(2, len(sentence)):
            trigram = tuple(sentence[i-2:i+1])
            if trigram in trigram_probs:
                log_prob += np.log2(trigram_probs[trigram])
            else:
                log_prob += np.log2(1e-10)  # 미등장 트라이그램에 대한 스무딩
            n += 1
    perplexity = 2 ** (-log_prob / n)
    return perplexity

# 신경망 퍼플렉시티 계산 함수
def calculate_neural_perplexity(test_data, model, word_to_id):
    log_prob = 0
    n = 0
    for sentence in test_data:
        for i in range(2, len(sentence)):
            context = [word_to_id.get(w, 0) for w in sentence[i-2:i]]
            target = word_to_id.get(sentence[i], 0)
            probs = model.predict(np.array([context]))[0]
            log_prob += np.log2(probs[target])
            n += 1
    perplexity = 2 ** (-log_prob / n)
    return perplexity

# 신경망 모델 생성 함수
def create_ffnn_lm(vocab_size, embedding_dim, context_size):
    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=context_size))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(vocab_size, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

# 사용 예 (단순화된 예시)
train_data = [
    ['빠른', '갈색', '여우가', '게으른', '개를', '뛰어넘었다'],
    ['한', '빠른', '갈색', '개가', '게으른', '고양이에게', '짖었다']
]

test_data = [
    ['갈색', '여우가', '뛰어넘었다', '한', '게으른', '고양이를'],
    ['빠른', '개가', '짖었다', '갈색', '여우에게']
]

# 트라이그램 확률 계산
trigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)

for sentence in train_data:
    for i in range(len(sentence) - 2):
        trigram = tuple(sentence[i:i+3])
        bigram = tuple(sentence[i:i+2])
        trigram_counts[trigram] += 1
        bigram_counts[bigram] += 1

trigram_probs = {trigram: count / bigram_counts[trigram[:2]] for trigram, count in trigram_counts.items()}

# 신경망 모델 학습 데이터 준비
vocab = list(set(word for sentence in train_data + test_data for word in sentence))
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

X = []
y = []
for sentence in train_data:
    for i in range(2, len(sentence)):
        X.append([word_to_id[w] for w in sentence[i-2:i]])
        y.append(word_to_id[sentence[i]])

X = np.array(X)
y = np.array(y)

# 신경망 모델 생성 및 학습
model = create_ffnn_lm(len(vocab), 50, 2)
model.fit(X, y, epochs=50, verbose=0)

# 퍼플렉시티 계산 및 비교
trigram_perplexity = calculate_trigram_perplexity(test_data, trigram_probs)
neural_perplexity = calculate_neural_perplexity(test_data, model, word_to_id)

print(f"트라이그램 모델 퍼플렉시티: {trigram_perplexity:.2f}")
print(f"신경망 모델 퍼플렉시티: {neural_perplexity:.2f}")
```

## 결론

이번 세션에서는 통계적 언어 모델링의 고급 기법을 탐구하고 언어 모델링에 대한 신경망 접근법을 소개했습니다. 우리가 다룬 내용은 다음과 같습니다:

1. 보간법과 백오프 방법을 포함한 고급 N-gram 기법
2. 더 긴 범위의 의존성을 포착하기 위한 스킵-그램 모델
3. 데이터 희소성 문제를 해결하기 위한 클래스 기반 언어 모델
4. 다양한 특성을 통합하기 위한 최대 엔트로피 언어 모델
5. 단어 임베딩과 간단한 피드포워드 신경망 모델을 포함한 신경망 언어 모델 소개
6. 퍼플렉시티를 사용한 통계적 및 신경망 언어 모델 비교

이러한 고급 기법들은 이전 세션에서 논의한 기본 N-gram 모델의 일부 한계를 해결합니다. 이들은 언어 모델링에서 향상된 성능과 유연성을 제공합니다.

## 주요 요점

1. **보간법과 백오프** 기법은 다른 차수의 N-gram을 결합하여 모델의 견고성을 향상시킵니다.
2. **스킵-그램 모델**은 N-gram 시퀀스에 갭을 허용하여 더 긴 범위의 의존성을 포착합니다.
3. **클래스 기반 모델**은 단어를 클래스로 그룹화하여 매개변수 수를 줄이고 데이터 희소성 문제를 해결합니다.
4. **최대 엔트로피 모델**은 다양한 언어학적 특성을 통합할 수 있는 유연한 프레임워크를 제공합니다.
5. **신경망 언어 모델**은 단어의 분산 표현을 학습하고 언어의 복잡한 패턴을 포착할 수 있습니다.
6. **단어 임베딩**은 단어를 밀집 벡터로 표현하여 단어 간의 의미적 관계를 포착합니다.

## 향후 방향

NLP와 언어 모델에 대한 우리의 학습을 계속하면서, 더 고급 신경망 아키텍처를 탐구할 것입니다:

1. **순환 신경망 (RNN)**: 이 모델들은 순차 데이터를 처리하도록 설계되었으며 텍스트의 장거리 의존성을 포착할 수 있습니다.

2. **장단기 메모리 (LSTM) 네트워크**: 소실 기울기 문제를 해결하는 RNN의 한 유형으로, 장기 의존성을 더 잘 모델링할 수 있습니다.

3. **트랜스포머 모델**: 이 주의 기반 모델들은 NLP를 혁신하여 다양한 언어 작업에서 최첨단 성능을 달성했습니다.

4. **사전 학습된 언어 모델**: BERT, GPT 및 그 변형과 같은 모델들로, 대규모 코퍼스에 대해 사전 학습되고 특정 작업에 대해 미세 조정될 수 있습니다.

## 실용적 고려사항

실제 응용 프로그램에서 언어 모델을 사용할 때 다음 사항을 고려하세요:

1. **모델 복잡성 vs 데이터 크기**: 더 복잡한 모델은 효과적으로 학습하기 위해 더 많은 데이터가 필요합니다. 데이터셋 크기에 적합한 모델을 선택하세요.

2. **도메인별 적응**: 사전 학습된 모델은 특정 도메인이나 작업에 대해 미세 조정하거나 적응시켜야 할 수 있습니다.

3. **계산 리소스**: 신경망 모델, 특히 대규모 사전 학습 모델은 학습과 사용에 계산 비용이 많이 들 수 있습니다. 모델을 선택할 때 사용 가능한 리소스를 고려하세요.

4. **해석 가능성**: 신경망 모델은 종종 더 나은 성능을 보이지만 통계적 모델보다 해석하기 어려울 수 있습니다. 특정 응용 프로그램에 대한 성능과 해석 가능성 사이의 균형을 고려하세요.

5. **윤리적 고려사항**: 특히 사전 학습된 모델을 사용하거나 민감한 응용 프로그램에서 작업할 때 언어 모델의 잠재적 편향성에 주의하세요.

## 연습

이 세션에서 다룬 개념에 대한 이해를 강화하기 위해:

1. 제공된 코드 스니펫을 시작점으로 사용하여 선형 보간법을 적용한 트라이그램 모델을 구현하세요.
2. 작은 코퍼스(예: 뉴스 기사 모음이나 책의 장)에 대해 모델을 학습시키세요.
3. 보간된 모델의 퍼플렉시티를 보간 없는 단순 트라이그램 모델과 비교하세요.
4. 다른 보간 가중치를 실험하고 모델 성능에 미치는 영향을 관찰하세요.
5. (선택사항) 신경망에 익숙하다면, 간단한 순환 신경망(RNN) 언어 모델을 구현하고 그 성능을 N-gram 모델과 비교해 보세요.

```python
# 여기에 코드를 작성하세요
```

이 연습을 완료함으로써, 여러분은 고급 통계적 언어 모델링 기법에 대한 실습 경험을 얻고 전통적인 방법에서 신경망 접근법으로의 전환을 이해하기 시작할 것입니다.
