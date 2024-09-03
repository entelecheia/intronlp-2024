# 세션 3 - 한국어 텍스트 전처리 및 토큰화

## 소개

이번 강의에서는 한국어 텍스트의 전처리와 토큰화에 관련된 독특한 과제와 기술을 살펴보겠습니다. 교착어인 한국어는 영어와 같은 언어들과 비교했을 때 뚜렷한 차이점을 가지고 있으며, 이러한 차이점을 이해하는 것은 효과적인 한국어 NLP를 위해 매우 중요합니다.

먼저 필요한 라이브러리를 가져오겠습니다:

```python
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from konlpy.tag import Okt, Kkma, Hannanum
from gensim.models import Word2Vec
import networkx as nx

# 한국어 형태소 분석기 초기화
okt = Okt()
kkma = Kkma()
hannanum = Hannanum()
```

## 1. 한국어의 특성

한국어는 교착어로, 단어가 형태소의 결합으로 형성됩니다. 이러한 특성으로 인해 한국어 텍스트 전처리는 영어와 같은 고립어에 비해 더욱 복잡합니다.

교착어와 고립어의 차이를 시각화해 보겠습니다:

```python
def plot_language_comparison():
    G = nx.DiGraph()
    G.add_edge("나", "는")
    G.add_edge("나", "를")
    G.add_edge("나", "에게")
    G.add_edge("I", "Subject")
    G.add_edge("I", "Object")
    G.add_edge("I", "To me")

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=3000, font_size=10, font_weight='bold',
            arrows=True, edge_color='gray')

    plt.title("교착어(한국어)와 고립어(영어)의 비교")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_language_comparison()
```

이 시각화에서 볼 수 있듯이, 한국어는 "나"(I)라는 어근에 조사를 붙여 문법적 기능을 변경하지만, 영어에서는 "I"라는 단어가 변하지 않고 단어 순서나 추가 단어로 그 기능이 결정됩니다.

## 2. 한국어 텍스트 전처리의 과제

한국어 텍스트 전처리는 몇 가지 독특한 과제에 직면합니다:

1. 명확한 단어 경계의 부재
2. 복잡한 형태론적 구조
3. 조사와 접사의 빈번한 사용
4. 높은 동음이의어 비율
5. 복합 명사

이러한 과제들을 예시와 함께 살펴보겠습니다:

```python
def demonstrate_korean_challenges():
    examples = [
        "나는학교에갔다",  # 띄어쓰기 없음
        "먹었습니다",  # 복잡한 형태론
        "사과를 먹었다",  # 조사 사용
        "감기에 걸렸다",  # 동음이의어 (감기는 "cold" 또는 "winding"을 의미할 수 있음)
        "한국어자연어처리",  # 복합 명사
    ]

    for i, example in enumerate(examples, 1):
        print(f"예시 {i}: {example}")
        print("형태소:", okt.morphs(example))
        print("명사:", okt.nouns(example))
        print("품사:", okt.pos(example))
        print()

demonstrate_korean_challenges()
```

## 3. 한국어 문장 토큰화

한국어 문장 토큰화는 다양한 맥락에서 사용되는 문장부호로 인해 까다로울 수 있습니다. 간단한 한국어 문장 토큰화기를 구현해 보겠습니다:

```python
def korean_sent_tokenize(text):
    # 간단한 규칙 기반 문장 토큰화기
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences

# 사용 예시
sample_text = "안녕하세요. 오늘은 날씨가 좋습니다! 산책 가실래요?"
sentences = korean_sent_tokenize(sample_text)
print("토큰화된 문장:", sentences)
```

## 4. 한국어 형태소 분석

형태소 분석은 한국어 NLP에서 매우 중요합니다. 이는 단어를 가장 작은 의미 단위(형태소)로 분해하는 과정을 포함합니다. 다양한 한국어 형태소 분석기를 비교해 보겠습니다:

```python
def compare_morphological_analyzers(text):
    analyzers = [okt, kkma, hannanum]
    analyzer_names = ['Okt', 'Kkma', 'Hannanum']

    fig, axes = plt.subplots(len(analyzers), 1, figsize=(12, 4*len(analyzers)))
    fig.suptitle("한국어 형태소 분석기 비교")

    for ax, analyzer, name in zip(axes, analyzers, analyzer_names):
        morphs = analyzer.morphs(text)
        ax.bar(range(len(morphs)), [1]*len(morphs), align='center')
        ax.set_xticks(range(len(morphs)))
        ax.set_xticklabels(morphs, rotation=45, ha='right')
        ax.set_title(f"{name} 분석기")
        ax.set_ylabel("형태소")

    plt.tight_layout()
    plt.show()

# 사용 예시
sample_text = "나는 맛있는 한국 음식을 좋아합니다."
compare_morphological_analyzers(sample_text)
```

## 5. 한국어 품사 태깅

한국어의 교착어적 특성으로 인해 품사 태깅이 더욱 복잡합니다. 다양한 분석기를 사용한 품사 태깅을 살펴보겠습니다:

```python
def compare_pos_tagging(text):
    analyzers = [okt, kkma, hannanum]
    analyzer_names = ['Okt', 'Kkma', 'Hannanum']

    for name, analyzer in zip(analyzer_names, analyzers):
        print(f"{name} 품사 태깅:")
        print(analyzer.pos(text))
        print()

# 사용 예시
sample_text = "나는 학교에서 한국어를 공부합니다."
compare_pos_tagging(sample_text)
```

품사 태그를 시각화해 보겠습니다:

```python
def visualize_pos_tags(text, analyzer):
    pos_tags = analyzer.pos(text)
    words, tags = zip(*pos_tags)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(words), y=[1]*len(words), hue=list(tags), dodge=False)
    plt.title(f"{analyzer.__class__.__name__}을 사용한 품사 태그")
    plt.xlabel("단어")
    plt.ylabel("")
    plt.legend(title="품사 태그", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

visualize_pos_tags(sample_text, okt)
```

## 6. 한국어 단어 임베딩

한국어 단어 임베딩은 언어의 형태론적 복잡성을 고려해야 합니다. 한국어 텍스트에 대해 간단한 Word2Vec 모델을 훈련시켜 보겠습니다:

```python
def train_korean_word2vec(sentences, vector_size=100, window=5, min_count=1):
    tokenized_sentences = [okt.morphs(sent) for sent in sentences]
    model = Word2Vec(tokenized_sentences, vector_size=vector_size, window=window, min_count=min_count)
    return model

# 사용 예시
korean_sentences = [
    "나는 학교에 갑니다.",
    "그는 공부를 열심히 합니다.",
    "우리는 한국어를 배웁니다.",
    "그녀는 책을 좋아합니다."
]

w2v_model = train_korean_word2vec(korean_sentences)

# 단어 임베딩 시각화
def plot_korean_word_embeddings(model, words):
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(12, 8))
    for i, word in enumerate(words):
        if word in model.wv:
            x, y = embedded[i]
            plt.scatter(x, y)
            plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.title("한국어 단어 임베딩 시각화")
    plt.xlabel("t-SNE 특성 0")
    plt.ylabel("t-SNE 특성 1")
    plt.tight_layout()
    plt.show()

plot_korean_word_embeddings(w2v_model, okt.morphs("나 학교 공부 한국어 책 좋아하다"))
```

## 7. 결론 및 모범 사례

이번 강의에서는 한국어 텍스트의 전처리와 토큰화에 관련된 독특한 과제와 기술을 살펴보았습니다. 다음은 명심해야 할 몇 가지 모범 사례입니다:

1. 토큰화와 형태소 분석을 위해 항상 전문화된 한국어 NLP 도구(예: KoNLPy)를 사용하세요.
2. 다양한 한국어 형태소 분석기 간의 차이를 인지하고 작업에 가장 적합한 것을 선택하세요.
3. 한국어 텍스트를 처리할 때 조사와 접사의 영향을 고려하세요.
4. 단어 임베딩을 사용할 때는 한국어의 형태론적 뉘앙스를 포착하기 위해 적절히 토큰화된 한국어 텍스트로 훈련하세요.
5. 한국어 텍스트를 해석할 때 동음이의어와 문맥을 주의 깊게 고려하세요.

효과적인 한국어 NLP는 언어의 구조와 특성에 대한 깊은 이해를 필요로 한다는 점을 기억하세요. 실제 한국어 텍스트 데이터로 지속적인 연습과 실험을 하면 한국어 처리의 독특한 과제를 다루는 데 도움이 될 것입니다.

## 연습

1. 한국어 텍스트 데이터셋을 선택하세요 (예: 뉴스 기사, 소셜 미디어 게시물, 제품 리뷰).
2. 다음을 포함하는 완전한 한국어 텍스트 전처리 파이프라인을 구현하세요:
   - 문장 토큰화
   - 형태소 분석
   - 품사 태깅
   - 불용어 제거 (한국어 불용어 목록을 만들어야 합니다)
3. 전처리된 한국어 텍스트로 Word2Vec 모델을 훈련시키세요.
4. 결과로 나온 단어 임베딩을 분석하고 관찰된 흥미로운 의미적 관계에 대해 논의하세요.

```python
# 여기에 코드를 작성하세요
```

이 연습은 한국어 텍스트 전처리 파이프라인 전체에 대한 실제 경험을 제공하고, NLP 작업에서 한국어 데이터를 다룰 때의 뉘앙스를 이해하는 데 도움이 될 것입니다.
