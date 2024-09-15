# 2주차 세션 2 - 고급 텍스트 전처리 및 표현

## 소개

이번 강의에서는 고급 텍스트 전처리 기술을 살펴보고 기계 학습 알고리즘에 적합한 형식으로 텍스트 데이터를 표현하는 방법을 깊이 있게 다룰 것입니다. 1회차에서 다룬 기본 개념을 바탕으로 텍스트 데이터를 분석하기 위한 더 정교한 접근 방식을 소개할 것입니다.

먼저 필요한 라이브러리를 가져오겠습니다:

```python
import spacy
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import matplotlib.font_manager as fm
import matplotlib
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from konlpy.tag import Okt

# 필요한 NLTK 데이터 다운로드
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# spaCy 모델 로드
nlp = spacy.load("en_core_web_sm")
```

## 1. 고급 텍스트 정제 기술

1회차의 기본적인 텍스트 정제를 바탕으로, 몇 가지 고급 기술을 살펴보겠습니다:

### 이모지와 이모티콘 처리

이모지와 이모티콘은 감정 정보를 담고 있을 수 있습니다. 이들을 제거하거나 텍스트 설명으로 대체할 수 있습니다.

```python
!pip install emoji
import emoji

def handle_emojis(text):
    # 이모지를 텍스트 설명으로 대체
    return emoji.demojize(text)

# 사용 예시
sample_text = "이 영화 정말 좋아요! 😍👍"
processed_text = handle_emojis(sample_text)
print("원본 텍스트:", sample_text)
print("처리된 텍스트:", processed_text)
```

### 비 ASCII 문자 처리

비 ASCII 문자를 포함할 수 있는 텍스트 데이터의 경우, 이를 제거하거나 정규화할 수 있습니다:

```python
# 유니코드 관련 모듈 임포트
import unicodedata

def normalize_unicode(text):
    # 유니코드 문자 정규화: NFKC 형태로 정규화 (한글을 유지)
    return unicodedata.normalize('NFKC', text)

# 사용 예시
sample_text = "카페 오 레는 맛있는 음료입니다"
normalized_text = normalize_unicode(sample_text)
print("원본 텍스트:", sample_text)
print("정규화된 텍스트:", normalized_text)
```

## 2. 축약어와 특수 경우 처리

축약어(예: "don't", "I'm")는 일부 NLP 작업에서 문제가 될 수 있습니다. 축약어를 확장하는 함수를 만들어 보겠습니다:

```python
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "don't": "do not",
    "I'm": "I am",
    "isn't": "is not",
    "it's": "it is",
    "won't": "will not",
    # 필요에 따라 더 많은 축약어 추가
}

def expand_contractions(text):
    for contraction, expansion in CONTRACTION_MAP.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
    return text

# 사용 예시
sample_text = "I can't believe it's not butter!"
expanded_text = expand_contractions(sample_text)
print("원본 텍스트:", sample_text)
print("확장된 텍스트:", expanded_text)
```

## 3. 개체명 인식 (NER)

개체명 인식은 텍스트에서 명명된 개체(예: 인명, 조직, 장소)를 식별하고 분류하는 과정입니다. spaCy를 사용하여 NER을 수행해 보겠습니다:

```python
# en_core_web_sm 모델 다운로드 (한 번만 실행하면 됩니다)
!python -m spacy download en_core_web_sm

# NER 모델 로드 (영어 모델 사용)
nlp = spacy.load("en_core_web_sm")

def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# 사용 예시
sample_text = "Apple was founded by Steve Jobs in Cupertino, California."
entities = perform_ner(sample_text)
print("명명된 개체:", entities)

# NER 결과 시각화
from spacy import displacy
displacy.render(nlp(sample_text), style="ent", jupyter=True)
```

## 4. 품사 태깅 (POS Tagging)

품사 태깅은 단어에 문법적 범주(예: 명사, 동사, 형용사)를 라벨링하는 과정입니다. 이 작업에 NLTK를 사용해 보겠습니다:

```python
# 1. 한글 폰트 설치
!apt-get install -y fonts-nanum

# 2. Colab에서 그래프를 바로 표시하기 위한 매직 명령어 추가
%matplotlib inline

# 필요한 모듈 임포트
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from konlpy.tag import Okt

# 나눔고딕 폰트 경로 설정
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_prop = fm.FontProperties(fname=font_path, size=10)

# 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# KoNLPy의 Okt 형태소 분석기 초기화
okt = Okt()

def pos_tag_text(text):
    return okt.pos(text)

# 테스트 텍스트
sample_text = "빠른 갈색 여우가 게으른 개를 뛰어넘습니다."
pos_tags = pos_tag_text(sample_text)
print("품사 태그:", pos_tags)

# 품사 태그 시각화
def plot_pos_tags(pos_tags):
    words, tags = zip(*pos_tags)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(words), y=[1]*len(words), hue=list(tags), dodge=False)
    plt.title("품사 태그", fontproperties=font_prop)
    plt.xlabel("단어", fontproperties=font_prop)
    plt.ylabel("")
    plt.legend(title="품사 태그", bbox_to_anchor=(1.05, 1), loc='upper left', prop=font_prop)
    plt.xticks(rotation=45, ha='right')
    for label in plt.gca().get_xticklabels():
        label.set_fontproperties(font_prop)
    plt.tight_layout()
    plt.show() 

# 그래프 출력
plot_pos_tags(pos_tags)
```

## 5. 텍스트 표현 방법

이제 고급 전처리 기술을 다뤘으니, 기계 학습 알고리즘에 적합한 형식으로 텍스트 데이터를 표현하는 방법을 살펴보겠습니다.

### 단어 주머니 (Bag of Words, BoW)

단어 주머니 모델은 텍스트를 단어 빈도의 벡터로 표현하며, 문법과 단어 순서를 무시합니다.

```python
def create_bow(corpus):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(corpus)
    return bow_matrix, vectorizer.get_feature_names_out()

# 사용 예시
corpus = [
    "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
    "게으른 개는 하루 종일 잠을 잡니다.",
    "빠른 갈색 여우는 빠릅니다."
]

bow_matrix, feature_names = create_bow(corpus)
print("BoW 특성 이름:", feature_names)
print("BoW 행렬 형태:", bow_matrix.shape)
print("BoW 행렬:")
print(bow_matrix.toarray())

# BoW 시각화
plt.figure(figsize=(12, 6))
sns.heatmap(bow_matrix.toarray(), annot=True, fmt='d', cmap='YlGnBu', xticklabels=feature_names)
plt.title("단어 주머니 표현")
plt.ylabel("문서")
plt.xlabel("단어")
plt.tight_layout()
plt.show()
```

### TF-IDF (단어 빈도-역문서 빈도)

TF-IDF는 문서 집합에 대한 문서 내 단어의 중요도를 나타냅니다.

```python
!apt-get install -y fonts-nanum

# 2. 폰트 캐시를 삭제하여 matplotlib가 다시 생성하도록 처리
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# 폰트 경로 설정
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
fm.fontManager.addfont(font_path)
plt.rc('font', family='NanumGothic')

# 캐시 파일 삭제 (matplotlib가 캐시를 다시 생성하도록 강제)
font_cache_dir = os.path.expanduser('~/.cache/matplotlib')
if os.path.exists(font_cache_dir):
    import shutil
    shutil.rmtree(font_cache_dir)

# 마이너스 기호 깨짐 방지 설정
plt.rcParams['axes.unicode_minus'] = False

# TF-IDF 생성 함수
def create_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer.get_feature_names_out()

# 사용 예시
corpus = [
    "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
    "종일 잠을 잡니다.",
    "빠른 여우는 게으른 개를 뛰어넘습니다."
]

tfidf_matrix, feature_names = create_tfidf(corpus)
print("TF-IDF 특성 이름:", feature_names)
print("TF-IDF 행렬 형태:", tfidf_matrix.shape)
print("TF-IDF 행렬:")
print(tfidf_matrix.toarray())

# TF-IDF 시각화
plt.figure(figsize=(12, 6))
sns.heatmap(tfidf_matrix.toarray(), annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=feature_names)
plt.title("TF-IDF 표현")
plt.ylabel("문서")
plt.xlabel("단어")
plt.tight_layout()
plt.show()
```

## 6. 단어 임베딩

단어 임베딩은 의미적 관계를 포착하는 단어의 밀집 벡터 표현입니다. 인기 있는 단어 임베딩 기술인 Word2Vec을 살펴보겠습니다.

```python
!pip install konlpy
import numpy as np
from konlpy.tag import Okt
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Word2Vec 학습 함수
def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
    return model

# 형태소 분석기 (Okt) 초기화
okt = Okt()

# 한국어 문장을 토큰화하는 함수
def tokenize_korean(corpus):
    return [okt.morphs(sentence) for sentence in corpus]

# 사용 예시
corpus = ["빠른 갈색 여우가 게으른 개를 뛰어넘습니다.", "종일 잠을 잡니다.", "빠른 여우는 게으른 개를 뛰어넘습니다."]
tokenized_corpus = tokenize_korean(corpus)
w2v_model = train_word2vec(tokenized_corpus)

# 모델에 존재하는 단어만 추출하는 함수
def get_model_words(model, words):
    return [word for word in words if word in model.wv]

# t-SNE를 사용하여 단어 임베딩 시각화
def plot_word_embeddings(model, words, perplexity=2):
    words_in_model = get_model_words(model, words)

    if len(words_in_model) < 2:
        print("시각화할 단어가 충분하지 않습니다.")
        return

    # Word vectors를 numpy 배열로 변환
    word_vectors = np.array([model.wv[word] for word in words_in_model])

    # perplexity 값을 단어 수보다 작게 설정
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(word_vectors) - 1))
    embedded = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(12, 8))
    for i, word in enumerate(words_in_model):
        x, y = embedded[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.title("단어 임베딩 시각화")
    plt.xlabel("t-SNE 특성 0")
    plt.ylabel("t-SNE 특성 1")
    plt.tight_layout()
    plt.show()

# 모델에 존재하는 단어로 시각화
words_to_visualize = ['빠른', '갈색', '여우', '게으른', '개', '뛰어넘습니다', '잠을']
plot_word_embeddings(w2v_model, words_to_visualize, perplexity=2)

# 학습된 모델에서 단어들을 확인해봅니다
print("모델에 포함된 단어들:", w2v_model.wv.index_to_key)

# 모델에 존재하는 단어들만 추출한 목록 확인
words_in_model = get_model_words(w2v_model, words_to_visualize)
print("시각화할 수 있는 단어들:", words_in_model)

# 시각화할 단어가 없는 경우, 이 부분을 통해 원인을 알 수 있습니다.
if len(words_in_model) < 2:
    print("시각화할 단어가 충분하지 않습니다. 모델에 단어가 학습되지 않았을 수 있습니다.")
else:
    plot_word_embeddings(w2v_model, words_to_visualize, perplexity=2)
```

## 7. 결론 및 모범 사례

이번 강의에서는 고급 텍스트 전처리 기술과 텍스트 데이터를 표현하는 다양한 방법을 다뤘습니다. 다음은 명심해야 할 몇 가지 모범 사례입니다:

1. 특정 작업과 데이터셋에 기반하여 전처리 기술을 선택하세요.
2. 다양한 텍스트 표현 방법 간의 트레이드오프를 고려하세요:
   - BoW와 TF-IDF는 간단하지만 의미적 정보를 잃을 수 있습니다.
   - 단어 임베딩은 의미적 관계를 포착하지만 효과적으로 훈련하려면 더 많은 데이터가 필요합니다.
3. 단어 임베딩을 사용할 때는 특히 작은 데이터셋에서 더 나은 성능을 위해 사전 훈련된 모델 사용을 고려하세요.
4. 특히 대규모 데이터셋을 다룰 때 다양한 표현 방법에 필요한 계산 리소스를 고려하세요.
5. 전처리 및 표현 선택이 모델 성능에 미치는 영향을 정기적으로 평가하세요.

텍스트 전처리와 표현은 NLP 파이프라인에서 중요한 단계이며, 여기서 내리는 선택이 모델의 성능에 큰 영향을 미칠 수 있음을 기억하세요.

## 연습

1. 선택한 데이터셋(예: 뉴스 기사 모음, 트윗, 제품 리뷰)을 선택하세요.
2. 우리가 배운 고급 전처리 기술(예: 이모지 처리, 축약어 확장, NER)을 적용하세요.
3. 전처리된 텍스트의 BoW와 TF-IDF 표현을 모두 만드세요.
4. 데이터셋에서 Word2Vec 모델을 훈련하고 몇 가지 흥미로운 단어의 임베딩을 시각화하세요.
5. 이러한 표현 방법들 간의 차이점과 이들이 후속 NLP 작업에 어떤 영향을 미칠 수 있는지 생각해 보세요.

```python
# 여기에 코드를 작성하세요
```

이 연습은 고급 전처리 기술과 다양한 텍스트 표현 방법에 대한 실제적인 경험을 제공할 것입니다. 이를 통해 실제 NLP 작업에서 이들의 실질적인 영향을 이해하는 데 도움이 될 것입니다.
