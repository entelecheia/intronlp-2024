# 2주차 세션 1 - 텍스트 전처리 기초

## 소개

이번 강의에서는 모든 자연어 처리(NLP) 파이프라인에서 중요한 단계인 텍스트 전처리의 기본 사항을 자세히 살펴보겠습니다. 텍스트 전처리는 원시 텍스트 데이터를 기계 학습 알고리즘이 쉽게 분석할 수 있는 깨끗하고 표준화된 형식으로 변환하는 과정입니다.

일반적인 전처리 단계는 다음과 같습니다:

1. 텍스트 정제
2. 소문자 변환
3. 토큰화
4. 불용어 제거
5. 어간 추출 또는 표제어 추출
6. 텍스트 표현

```{mermaid}
:align: center
graph TD
    A[원시 텍스트] --> B[텍스트 정제]
    B --> C[소문자 변환]
    C --> D[토큰화]
    D --> E[불용어 제거]
    E --> F[어간 추출/표제어 추출]
    F --> G[텍스트 표현]
    G --> H[처리된 텍스트]
```

먼저 필요한 라이브러리를 가져오겠습니다:

```python
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# 필요한 NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 1. 텍스트 전처리의 중요성

텍스트 전처리는 NLP에서 여러 가지 이유로 중요합니다:

1. 텍스트에서 노이즈와 관련 없는 정보를 제거합니다.
2. 텍스트를 표준화하여 알고리즘이 처리하기 쉽게 만듭니다.
3. 후속 NLP 작업의 성능을 크게 향상시킬 수 있습니다.

일반적인 텍스트 전처리 파이프라인을 시각화해 보겠습니다:

```python
import networkx as nx

def create_preprocessing_pipeline_graph():
    G = nx.DiGraph()
    steps = ["원시 텍스트", "텍스트 정제", "소문자 변환", "토큰화",
             "불용어 제거", "어간 추출/표제어 추출", "처리된 텍스트"]
    G.add_edges_from(zip(steps[:-1], steps[1:]))

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=3000, font_size=10, font_weight='bold',
            arrows=True, edge_color='gray')

    plt.title("텍스트 전처리 파이프라인")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

create_preprocessing_pipeline_graph()
```

이 파이프라인은 텍스트 전처리의 일반적인 단계를 보여줍니다. 이제 각 단계를 자세히 살펴보겠습니다.

## 2. 텍스트 정제

텍스트 정제는 분석과 관련이 없는 텍스트의 요소를 제거하거나 대체하는 과정입니다. 이 단계는 특히 웹 스크래핑 데이터나 소셜 미디어 콘텐츠를 다룰 때 중요합니다.

일반적인 텍스트 정제 작업은 다음과 같습니다:

- HTML 태그 제거
- 특수 문자 처리
- URL 및 이메일 주소 제거 또는 대체
- 숫자와 날짜 처리

기본적인 텍스트 정제 함수를 구현해 보겠습니다:

```python
def clean_text(text):
    # HTML 태그 제거
    text = re.sub('<.*?>', '', text)
    # URL 제거
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 이메일 주소 제거
    text = re.sub(r'\S+@\S+', '', text)
    # 특수 문자와 숫자 제거
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# 사용 예시
sample_text = "<p>https://example.com 에서 우리 웹사이트를 확인하거나 info@example.com으로 이메일을 보내주세요. 특별 할인: 50% 할인!</p>"
cleaned_text = clean_text(sample_text)
print("원본 텍스트:", sample_text)
print("정제된 텍스트:", cleaned_text)
```

## 3. 소문자 변환

텍스트를 소문자로 변환하면 텍스트를 표준화하고 어휘 크기를 줄일 수 있습니다. 그러나 대소문자 정보가 관련될 수 있는 경우(예: 명명된 개체 인식)를 고려하는 것이 중요합니다.

```python
def to_lowercase(text):
    return text.lower()

# 사용 예시
sample_text = "빠른 갈색 여우가 게으른 개를 뛰어넘습니다"
lowercased_text = to_lowercase(sample_text)
print("원본 텍스트:", sample_text)
print("소문자 변환 텍스트:", lowercased_text)
```

소문자 변환이 단어 빈도에 미치는 영향을 시각화해 보겠습니다:

```python
def plot_word_frequency(text, title):
    words = word_tokenize(text)
    word_freq = nltk.FreqDist(words)
    plt.figure(figsize=(12, 6))
    word_freq.plot(30, cumulative=False)
    plt.title(title)
    plt.show()

plot_word_frequency(sample_text, "단어 빈도 (원본)")
plot_word_frequency(lowercased_text, "단어 빈도 (소문자 변환)")
```

보시다시피, 소문자 변환은 고유 단어의 수를 줄이며, 이는 많은 NLP 작업에 유용할 수 있습니다.

## 4. 토큰화

토큰화는 텍스트를 더 작은 단위, 일반적으로 단어나 문장으로 나누는 과정입니다. 이는 많은 NLP 작업의 기본적인 단계입니다.

```python
def tokenize_text(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return words, sentences

# 사용 예시
sample_text = "이것은 첫 번째 문장입니다. 여기 또 다른 문장이 있습니다. 질문은 어떨까요?"
words, sentences = tokenize_text(sample_text)
print("단어:", words)
print("문장:", sentences)
```

토큰화 과정을 시각화해 보겠습니다:

```python
def visualize_tokenization(text):
    words, sentences = tokenize_text(text)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 단어 토큰화
    ax1.bar(range(len(words)), [1]*len(words), align='center')
    ax1.set_xticks(range(len(words)))
    ax1.set_xticklabels(words, rotation=45, ha='right')
    ax1.set_title('단어 토큰화')
    ax1.set_ylabel('토큰')

    # 문장 토큰화
    ax2.bar(range(len(sentences)), [1]*len(sentences), align='center')
    ax2.set_xticks(range(len(sentences)))
    ax2.set_xticklabels(sentences, rotation=0, ha='center', wrap=True)
    ax2.set_title('문장 토큰화')
    ax2.set_ylabel('문장')

    plt.tight_layout()
    plt.show()

visualize_tokenization(sample_text)
```

이 시각화는 텍스트가 개별 단어와 문장으로 어떻게 나뉘는지 이해하는 데 도움이 됩니다.

## 5. 불용어 제거

불용어는 보통 텍스트의 의미에 많은 기여를 하지 않는 일반적인 단어들입니다. 이를 제거하면 데이터의 노이즈를 줄이는 데 도움이 될 수 있습니다.

```python
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# 사용 예시
sample_text = "이것은 매우 일반적인 단어가 포함된 샘플 문장입니다."
filtered_text = remove_stopwords(sample_text)
print("원본 텍스트:", sample_text)
print("불용어 제거 텍스트:", filtered_text)
```

불용어 제거의 효과를 시각화해 보겠습니다:

```python
def plot_word_cloud(text, title):
    from wordcloud import WordCloud

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

plot_word_cloud(sample_text, "단어 구름 (불용어 포함)")
plot_word_cloud(filtered_text, "단어 구름 (불용어 제거)")
```

보시다시피, 불용어를 제거하면 더 의미 있는 내용 단어에 집중할 수 있습니다.

## 6. 어간 추출과 표제어 추출

어간 추출과 표제어 추출은 단어를 어근 형태로 축소하는 기술로, 어휘 크기를 줄이고 유사한 단어를 그룹화하는 데 도움이 될 수 있습니다.

### 어간 추출

```python
def stem_words(text):
    ps = PorterStemmer()
    words = word_tokenize(text)
    stemmed_words = [ps.stem(word) for word in words]
    return ' '.join(stemmed_words)

# 사용 예시
sample_text = "달리는 사람이 달리기 트랙을 빠르게 달립니다"
stemmed_text = stem_words(sample_text)
print("원본 텍스트:", sample_text)
print("어간 추출 텍스트:", stemmed_text)
```

### 표제어 추출

```python
def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# 사용 예시
sample_text = "아이들이 장난감을 가지고 놀고 있습니다"
lemmatized_text = lemmatize_words(sample_text)
print("원본 텍스트:", sample_text)
print("표제어 추출 텍스트:", lemmatized_text)
```

어간 추출과 표제어 추출을 비교해 보겠습니다:

```python
def compare_stem_lemma(text):
    stemmed = stem_words(text)
    lemmatized = lemmatize_words(text)

    words = word_tokenize(text)
    stem_words = word_tokenize(stemmed)
    lemma_words = word_tokenize(lemmatized)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(words))
    ax.plot(x, [1]*len(words), 'bo', label='원본')
    ax.plot(x, [0.66]*len(stem_words), 'ro', label='어간 추출')
    ax.plot(x, [0.33]*len(lemma_words), 'go', label='표제어 추출')

    for i, (orig, stem, lemma) in enumerate(zip(words, stem_words, lemma_words)):
        ax.text(i, 1, orig, ha='center', va='bottom')
        ax.text(i, 0.66, stem, ha='center', va='bottom', color='red')
        ax.text(i, 0.33, lemma, ha='center', va='bottom', color='green')

    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend()
    ax.set_title('원본, 어간 추출, 표제어 추출 단어 비교')

    plt.tight_layout()
    plt.show()

compare_stem_lemma("달리는 사람이 달리기 트랙을 빠르게 달립니다")
```

이 시각화는 어간 추출과 표제어 추출의 차이를 이해하는 데 도움이 됩니다.

네, 계속해서 한국어로 번역하겠습니다.

## 7. 모든 과정 통합하기

이제 모든 개별 단계를 다뤘으니, 종합적인 텍스트 전처리 함수를 만들어 보겠습니다:

```python
def preprocess_text(text, remove_stopwords=True, stem=False, lemmatize=True):
    # 텍스트 정제
    text = clean_text(text)

    # 소문자로 변환
    text = to_lowercase(text)

    # 토큰화
    words = word_tokenize(text)

    # 지정된 경우 불용어 제거
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

    # 어간 추출 또는 표제어 추출
    if stem:
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words]
    elif lemmatize:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# 사용 예시
sample_text = """<p>빠른 갈색 여우가 게으른 개를 뛰어넘습니다.
                  이 문장은 종종 타이포그래피에서 글꼴을 보여주는 데 사용됩니다.
                  자세한 정보는 https://example.com을 방문하세요.</p>"""

preprocessed_text = preprocess_text(sample_text)
print("원본 텍스트:", sample_text)
print("전처리된 텍스트:", preprocessed_text)
```

## 8. 결론 및 모범 사례

이번 강의에서는 NLP 작업을 위한 기본적인 텍스트 전처리 기술을 다뤘습니다. 다음은 명심해야 할 몇 가지 모범 사례입니다:

1. 어떤 전처리 단계를 적용할지 결정할 때는 항상 특정 작업과 데이터셋을 고려하세요.
2. 훈련 데이터와 테스트 데이터 전반에 걸쳐 전처리를 일관성 있게 유지하세요.
3. 너무 많은 정보를 제거하지 않도록 주의하세요 (예: 숫자가 일부 작업에 중요할 수 있습니다).
4. 전처리 단계를 명확하게 문서화하세요. 이는 결과에 큰 영향을 미칠 수 있습니다.
5. 다양한 기술 간의 트레이드오프를 고려하세요 (예: 어간 추출 vs 표제어 추출).
6. 다국어 데이터를 다룰 때는 전처리 기술이 해당 언어에 적합한지 확인하세요.

전처리는 단순한 기술적 단계가 아니라 연구 목표와 데이터의 특성을 신중히 고려해야 하는 분석적 단계임을 기억하세요.

다음 세션에서는 더 고급 전처리 기술을 살펴보고 텍스트 표현 방법에 대해 알아볼 것입니다. 기대해 주세요!

## 연습

여러분이 선택한 문단에 `preprocess_text` 함수를 적용해 보세요. 다양한 매개변수 조합(예: 불용어 제거 유무, 어간 추출 vs 표제어 추출 사용)으로 실험해 보고 이들이 출력에 어떤 영향을 미치는지 관찰하세요. 어떤 조합이 다양한 유형의 NLP 작업에 가장 적합할지 생각해 보세요.

```python
# 여기에 코드를 작성하세요
```

이 연습은 각 전처리 선택이 미치는 영향을 이해하고 각 기술을 언제 사용해야 할지에 대한 직관을 개발하는 데 도움이 될 것입니다.
