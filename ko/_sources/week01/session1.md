# 1주차 세션 1 - 자연어처리의 기초와 발전

## 1. 자연어처리(NLP) 소개

### 1.1 NLP의 정의

자연어처리(NLP)는 언어학, 컴퓨터 과학, 인공지능을 결합한 학제간 분야로, 컴퓨터가 인간의 언어를 이해, 해석, 생성할 수 있게 합니다. NLP의 주요 목표는 인간 의사소통과 컴퓨터 이해 사이의 간극을 좁히는 것입니다.

```{mermaid}
graph TD
    A[자연어처리] --> B[언어학]
    A --> C[컴퓨터 과학]
    A --> D[인공지능]
    B --> E[구문론]
    B --> F[의미론]
    B --> G[화용론]
    C --> H[알고리즘]
    C --> I[데이터 구조]
    D --> J[기계학습]
    D --> K[딥러닝]
```

### 1.2 기본 개념

NLP의 주요 개념:

1. **토큰화**: 텍스트를 개별 단어나 하위 단어로 분리
2. **구문 분석**: 문장의 문법 구조 분석
3. **의미 분석**: 단어와 문장의 의미 해석
4. **개체명 인식(NER)**: 텍스트에서 명명된 개체 식별 및 분류
5. **품사 태깅(POS Tagging)**: 단어에 문법적 범주 할당
6. **감성 분석**: 텍스트의 감정적 톤 결정

파이썬의 Natural Language Toolkit (NLTK)를 사용한 간단한 예제:

```python
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import names
from nltk.chunk import ne_chunk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# 예문
sentence = "존은 뉴욕의 구글에서 일합니다."

# 토큰화
tokens = word_tokenize(sentence)
print("토큰:", tokens)

# 품사 태깅
pos_tags = pos_tag(tokens)
print("품사 태그:", pos_tags)

# 개체명 인식
ner_tree = ne_chunk(pos_tags)
print("개체명:")
for chunk in ner_tree:
    if hasattr(chunk, 'label'):
        print(chunk.label(), ' '.join(c[0] for c in chunk))
```

출력:

```
토큰: ['존은', '뉴욕의', '구글에서', '일합니다', '.']
품사 태그: [('존은', 'NNP'), ('뉴욕의', 'NNP'), ('구글에서', 'NNP'), ('일합니다', 'VBP'), ('.', '.')]
개체명:
PERSON 존은
GPE 뉴욕의
ORGANIZATION 구글에서
```

## 2. NLP의 역사적 관점

```{mermaid}
timeline
    title NLP의 진화
    1950년대 : 규칙 기반 시스템
    1960년대 : 초기 기계 번역
    1970년대 : 개념적 온톨로지
    1980년대 : 통계적 NLP 시작
    1990년대 : 기계학습 접근법
    2000년대 : 통계적 기계 번역 & 웹 스케일 데이터
    2010년대 : 딥러닝 & 신경망
    2020년대 : 대규모 언어 모델
```

### 2.1 초기 접근법 (1950년대-1980년대)

초기 NLP 시스템은 주로 규칙 기반으로, 수작업으로 만든 규칙과 전문가 지식에 의존했습니다. 이러한 접근법은 노암 촘스키의 형식 언어 이론의 영향을 받았으며, 이는 언어가 문법 규칙 집합으로 설명될 수 있다고 제안했습니다.

주요 발전:

1. **ELIZA (1966)**: 패턴 매칭과 대체 규칙을 사용하여 심리치료사의 응답을 시뮬레이션하는 초기 챗봇 중 하나.
2. **SHRDLU (1970)**: 단순화된 블록 세계에서 명령을 해석하고 응답할 수 있는 자연어 이해 프로그램.
3. **개념 의존성 이론 (1970년대)**: Roger Schank이 제안한 이론으로, 문장의 의미를 언어 독립적 형식으로 표현하는 것을 목표로 함.

ELIZA와 유사한 패턴 매칭 예제:

```python
import re

patterns = [
    (r'나는 (.*)', "당신이 {}라고 말씀하시는 이유가 무엇인가요?"),
    (r'나는 당신을 (.*)', "당신이 저를 {}하는 이유가 무엇인가요?"),
    (r'(.*) 죄송합니다 (.*)', "사과할 필요가 없습니다."),
    (r'안녕(.*)', "안녕하세요! 오늘 어떤 도움이 필요하신가요?"),
    (r'(.*)', "그것에 대해 좀 더 자세히 설명해 주시겠어요?")
]

def eliza_response(input_text):
    for pattern, response in patterns:
        match = re.match(pattern, input_text.rstrip(".!"))
        if match:
            return response.format(*match.groups())
    return "잘 이해하지 못했습니다. 다르게 표현해 주시겠어요?"

# 사용 예
while True:
    user_input = input("당신: ")
    if user_input.lower() == '종료':
        break
    print("ELIZA:", eliza_response(user_input))
```

초기 접근법의 한계:

- 복잡하거나 모호한 언어 처리 불가
- 모든 가능한 언어적 변형을 포함하도록 확장하기 어려움
- 새로운 도메인이나 언어에 적응하기 어려움

### 2.2 통계적 혁명 (1980년대-2000년대)

1980년대에는 NLP에서 통계적 방법으로의 전환이 일어났습니다. 이는 다음과 같은 요인에 의해 주도되었습니다:

1. 디지털 텍스트 코퍼스의 가용성 증가
2. 컴퓨팅 파워의 성장
3. 기계학습 기술의 발전

주요 발전:

1. **은닉 마르코프 모델(HMMs)**: 품사 태깅 및 음성 인식에 사용
2. **확률적 문맥 자유 문법(PCFGs)**: 구문 분석 작업에 적용
3. **IBM 모델**: 통계적 기계 번역의 선구자
4. **최대 엔트로피 모델**: NLP의 다양한 분류 작업에 사용

감성 분석을 위한 간단한 나이브 베이즈 분류기 예제:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 샘플 데이터
texts = [
    "이 영화는 정말 좋았어요", "훌륭한 영화, 강력 추천합니다",
    "최악의 영화, 시간 낭비였어요", "이 영화가 싫어요",
    "이 영화에 대해 중립적인 의견이에요", "그저 그랬어요, 특별한 건 없었어요"
]
labels = [1, 1, 0, 0, 2, 2]  # 1: 긍정, 0: 부정, 2: 중립

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 텍스트 벡터화
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 나이브 베이즈 분류기 학습
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# 예측 및 평가
y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=['부정', '긍정', '중립']))
```

이 시기에는 또한 실제 세계의 대규모 텍스트 데이터 컬렉션을 통한 언어 연구를 강조하는 말뭉치 언어학이 등장했습니다.

## 3. 현대 NLP와 딥러닝 (2010년대-현재)

현재 NLP 시대는 특히 트랜스포머 기반 모델을 중심으로 한 딥러닝 접근법의 우세로 특징지어집니다.

```{mermaid}
graph TD
    A[현대 NLP] --> B[단어 임베딩]
    A --> C[심층 신경망]
    A --> D[트랜스포머 아키텍처]
    B --> E[Word2Vec]
    B --> F[GloVe]
    C --> G[RNN/LSTM]
    C --> H[NLP를 위한 CNN]
    D --> I[BERT]
    D --> J[GPT]
    D --> K[T5]
```

주요 발전:

1. **단어 임베딩**: 단어의 밀집 벡터 표현 (예: Word2Vec, GloVe)
2. **순환 신경망(RNN)**: 특히 시퀀스 모델링을 위한 LSTM(Long Short-Term Memory) 네트워크
3. **트랜스포머 아키텍처**: 다양한 NLP 작업에서 성능을 혁신한 주의 기반 모델

감성 분석을 위한 사전 학습된 BERT 모델 사용 예제:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 사전 학습된 모델과 토크나이저 로드
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_score = torch.argmax(probabilities).item() + 1  # 1부터 5까지의 점수
    return sentiment_score

# 사용 예
texts = [
    "이 영화는 정말 좋았어요! 환상적이었습니다.",
    "영화는 그저 그랬어요, 특별한 건 없었어요.",
    "최악의 영화였어요. 연기도 형편없고 줄거리도 엉망이었습니다."
]

for text in texts:
    sentiment = analyze_sentiment(text)
    print(f"텍스트: {text}")
    print(f"감성 점수 (1-5): {sentiment}\n")
```

## 4. 전통적인 NLP 파이프라인

전통적인 NLP 파이프라인은 일반적으로 여러 단계로 구성됩니다:

```{mermaid}
graph LR
    A[텍스트 입력] --> B[텍스트 전처리]
    B --> C[특징 추출]
    C --> D[모델 학습]
    D --> E[평가]
    E --> F[응용]
```

### 4.1 텍스트 전처리

텍스트 전처리는 원시 텍스트 데이터를 정제하고 표준화하는 데 중요합니다. 일반적인 단계는 다음과 같습니다:

1. 토큰화: 텍스트를 단어나 하위 단어로 분리
2. 소문자화: 모든 텍스트를 소문자로 변환하여 차원을 줄임
3. 노이즈 제거: 관련 없는 문자나 형식 제거
4. 어간 추출 및 표제어 추출: 단어를 어근 형태로 축소

NLTK를 사용한 전처리 파이프라인 예제:

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # 토큰화 및 소문자화
    tokens = word_tokenize(text.lower())

    # 불용어 제거 및 알파벳이 아닌 토큰 제거
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

    # 어간 추출
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # 표제어 추출
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return {
        '원본': tokens,
        '어간 추출': stemmed_tokens,
        '표제어 추출': lemmatized_tokens
    }

# 사용 예
text = "고양이들이 숲을 통해 빠르게 달리고 있습니다."
preprocessed = preprocess_text(text)
print("원본 토큰:", preprocessed['원본'])
print("어간 추출 토큰:", preprocessed['어간 추출'])
print("표제어 추출 토큰:", preprocessed['표제어 추출'])
```

### 4.2 특징 추출

특징 추출은 텍스트를 기계 학습 모델이 처리할 수 있는 수치 표현으로 변환하는 과정입니다. 일반적인 기법으로는:

1. 단어 주머니 모델: 단어 빈도를 벡터로 표현
2. TF-IDF (Term Frequency-Inverse Document Frequency): 문서와 말뭉치에서의 중요도에 따라 단어에 가중치 부여
3. N-그램: N개의 인접한 단어 시퀀스 캡처

scikit-learn을 사용하여 TF-IDF 특징을 생성하는 예제:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 샘플 문서
documents = [
    "고양이가 매트 위에 앉았다",
    "개가 고양이를 쫓았다",
    "매트는 새것이었다"
]

# TF-IDF 특징 생성
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 특징 이름 가져오기
feature_names = vectorizer.get_feature_names_out()

# 각 문서의 TF-IDF 점수 출력
for i, doc in enumerate(documents):
    print(f"문서 {i + 1}:")
    for j, feature in enumerate(feature_names):
        score = tfidf_matrix[i, j]
        if score > 0:
            print(f"  {feature}: {score:.4f}")
    print()
```

### 4.3 모델 학습 및 평가

특징이 추출되면 특정 NLP 작업을 위해 다양한 기계 학습 알고리즘을 적용하여 모델을 학습할 수 있습니다. 일반적인 알고리즘으로는:

1. 나이브 베이즈
2. 서포트 벡터 머신 (SVM)
3. 의사결정 트리 및 랜덤 포레스트
4. 로지스틱 회귀

간단한 텍스트 분류 모델 학습 및 평가 예제:

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 샘플 데이터
texts = [
    "이 영화는 정말 좋아요", "훌륭한 영화, 강력 추천합니다",
    "최악의 영화, 시간 낭비였어요", "이 영화가 싫어요",
    "이 영화에 대해 중립적인 의견이에요", "그저 그랬어요, 특별한 건 없었어요"
]
labels = [1, 1, 0, 0, 2, 2]  # 1: 긍정, 0: 부정, 2: 중립

# 데이터를 학습 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# TF-IDF 특징 생성
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 나이브 베이즈 분류기 학습
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# 테스트 세트에 대한 예측
y_pred = clf.predict(X_test_tfidf)

# 모델 평가
print(classification_report(y_test, y_pred, target_names=['부정', '긍정', '중립']))
```

## 5. 전통적인 NLP의 도전 과제

1. **언어 모호성 처리** (계속):

   예: "나는 망원경으로 언덕 위의 사람을 보았다"

   - 그 사람이 망원경을 들고 있나요?
   - 화자가 망원경으로 그 사람을 보고 있나요?
   - 망원경이 언덕 위에 있나요?

   이 문장은 어휘적, 구문적 모호성을 모두 보여주며, 추가적인 맥락이나 복잡한 추론 없이는 전통적인 NLP 시스템이 해결하기 어려운 과제입니다.

2. **맥락과 의미론 다루기**:
   전통적인 NLP 모델은 다음을 포착하는 데 어려움을 겪었습니다:

   - 텍스트의 장거리 의존성
   - 맥락적 뉘앙스와 함축된 의미
   - 화용론 및 담화 수준의 이해

   예: 풍자나 아이러니를 이해하려면 문자 그대로의 단어 의미를 넘어선 맥락 파악이 필요합니다.

   다음 대화를 고려해보세요:
   A: "오늘 어땠어요?"
   B: "아, 정말 좋았죠. 차가 고장 나고, 셔츠에 커피를 쏟고, 중요한 회의에 늦었어요."

   전통적인 NLP 시스템은 "좋았죠"라는 단어 때문에 B의 응답을 긍정적으로 해석할 수 있지만, 이어지는 부정적인 사건들로 인해 이 응답이 실제로는 풍자임을 인식하지 못할 수 있습니다.

3. **희귀 단어 및 어휘 외 용어 처리**:
   전통적인 모델은 다음과 같은 경우에 어려움을 겪었습니다:

   - 학습 중에 보지 못한 단어 (어휘 외 단어)
   - 고유 명사 및 전문 용어
   - 신조어 및 진화하는 언어

   예: 일반 텍스트로 학습된 모델은 의학 또는 법률 문서의 특정 도메인 용어를 처리하는 데 어려움을 겪을 수 있습니다.

4. **공지시 해결**:
   텍스트 내에서 서로 다른 단어나 구가 동일한 개체를 가리키는 경우를 식별하는 것.

   예: "존이 상점에 갔습니다. 그는 우유를 샀습니다."
   시스템은 "그"가 "존"을 가리킨다는 것을 이해해야 합니다.

5. **계산 복잡성**:
   어휘와 데이터셋이 커짐에 따라 전통적인 NLP 방법은 확장성 문제에 직면했습니다:

   - 단어 주머니 모델의 고차원 특징 공간
   - 복잡한 문장 구문 분석의 계산 비용
   - 대규모 언어 모델 저장을 위한 메모리 요구 사항

6. **일반화 부족**:
   전통적인 모델은 학습 데이터와 다른 도메인이나 텍스트 스타일에 적용될 때 성능이 저하되는 경우가 많았습니다.

   예: 영화 리뷰로 학습된 감성 분석 모델은 제품 리뷰나 소셜 미디어 게시물에 적용될 때 성능이 떨어질 수 있습니다.

7. **의미적 유사성 포착의 어려움**:
   단어 주머니나 TF-IDF와 같은 전통적인 방법은 단어 간의 의미적 관계를 포착하는 데 어려움을 겪습니다.

   예: "개가 고양이를 쫓았다"와 "그 동물이 다른 동물을 추격했다"는 문장은 매우 다른 단어 주머니 표현을 가지지만 유사한 의미를 가집니다.

8. **다국어 및 교차 언어 작업 처리**:
   전통적인 NLP 시스템은 종종 각 언어에 대해 별도의 모델을 필요로 했으며, 이로 인해 다국어 및 교차 언어 작업이 어려웠습니다.

이러한 도전 과제들을 설명하기 위해 전통적인 NLP 기법을 사용한 간단한 예제를 살펴보겠습니다:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 샘플 문장
sentences = [
    "개가 고양이를 쫓았다.",
    "그 동물이 다른 동물을 추격했다.",
    "나는 개와 고양이를 좋아한다.",
    "금융 기관은 강 옆에 있다."
]

# 단어 주머니 표현 생성
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(sentences)

# 문장 간 코사인 유사도 계산
similarity_matrix = cosine_similarity(bow_matrix)

print("유사도 행렬:")
print(similarity_matrix)

print("\n어휘:")
print(vectorizer.get_feature_names_out())
```

출력:

```
유사도 행렬:
[[1.         0.         0.40824829 0.        ]
 [0.         1.         0.         0.        ]
 [0.40824829 0.         1.         0.        ]
 [0.         0.         0.         1.        ]]

어휘:
['개가' '강' '고양이를' '그' '금융' '기관은' '나는' '다른' '동물이' '동물을'
 '뿐에' '쫓았다' '좋아한다' '있다' '추격했다']
```

이 예제는 전통적인 NLP 접근 방식의 여러 한계를 보여줍니다:

1. **의미적 유사성**: 문장 1과 2는 매우 유사한 의미를 가지지만, 단어 주머니 표현이 완전히 다르기 때문에 유사도가 0입니다.

2. **단어 순서**: 단어 주머니 모델은 단어 순서에 대한 모든 정보를 잃어버리며, 이는 의미 이해에 중요할 수 있습니다.

3. **모호성**: 마지막 문장의 "강"이라는 단어가 강(river)을 의미하는지 아니면 형용사로 사용되었는지 모델은 구분할 수 없습니다.

4. **어휘 불일치**: "개"와 "동물"은 관련된 개념임에도 불구하고 완전히 다른 용어로 취급됩니다.

5. **확장성**: 어휘가 증가함에 따라 특징 공간이 점점 더 희소해지고 고차원화되어 계산적 문제가 발생합니다.

이러한 도전 과제들은 분산 의미론, 단어 임베딩, 그리고 결국에는 딥 러닝 모델을 포함한 더 발전된 NLP 기술의 개발을 촉진했습니다. 다음 세션에서는 이러한 현대적 NLP 접근 방식이 이러한 한계를 어떻게 해결하고 자연어 이해 및 생성의 새로운 가능성을 열었는지 살펴볼 것입니다.

## 결론

이 세션에서는 자연어처리의 기본 사항을 다루었으며, 초기 규칙 기반 시스템부터 통계적 접근 방식에 이르기까지 그 진화를 추적했습니다. 우리는 기본 개념, 전통적인 NLP 파이프라인, 그리고 이러한 기존 방법이 직면한 중요한 도전 과제들을 살펴보았습니다.

주요 내용 요약:

1. NLP는 언어학, 컴퓨터 과학, AI를 결합하여 컴퓨터가 인간의 언어를 처리하고 이해할 수 있게 하는 학제간 분야입니다.
2. 이 분야는 규칙 기반 시스템에서 통계적 방법을 거쳐 현대의 딥 러닝 접근 방식으로 발전해 왔습니다.
3. 전통적인 NLP 파이프라인은 텍스트 전처리, 특징 추출, 모델 학습/평가를 포함합니다.
4. 전통적인 NLP 방법은 언어 모호성 처리, 맥락, 희귀 단어, 의미적 유사성 처리에 있어 상당한 도전에 직면했습니다.

다음 세션에서는 현대적인 NLP 기술을 더 깊이 살펴보며, 딥 러닝과 트랜스포머 모델이 이러한 많은 도전 과제들을 어떻게 해결하고 자연어 이해 및 생성의 경계를 넓히는지 탐구할 것입니다.
