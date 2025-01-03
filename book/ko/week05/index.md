# 5주차: 트랜스포머

## 소개

안녕하세요, 5주차 수업에 오신 것을 환영합니다! 이번 주에는 자연어 처리 분야를 크게 발전시킨 혁명적인 모델인 **트랜스포머 아키텍처**를 살펴볼 예정입니다. 트랜스포머는 어텐션 메커니즘의 개념을 도입하여 기존의 순환 신경망보다 더 효과적으로 데이터의 관계를 포착할 수 있게 합니다.

## 학습 목표

이번 주가 끝날 때까지 여러분은 다음을 할 수 있어야 합니다:

- 트랜스포머 아키텍처의 기본 개념을 이해한다.
- 트랜스포머 내에서 어텐션 메커니즘이 어떻게 작동하는지 설명한다.
- 트랜스포머 모델의 구조와 구성 요소를 분석한다.
- 현대 NLP 응용 프로그램에 대한 트랜스포머의 영향을 이해한다.

## 주요 학습 내용

### 어텐션 메커니즘

- **셀프 어텐션**: 모델이 입력 시퀀스의 다른 부분에 어떻게 집중하여 표현을 생성하는지 배웁니다.
- **멀티헤드 어텐션**: 다양한 관계를 포착하기 위해 여러 어텐션 메커니즘이 병렬로 작동하는 방식을 이해합니다.

### 트랜스포머 구조

- **인코더와 디코더 모듈**: 입력을 처리하고 출력을 생성하는 데 있어 인코더와 디코더의 역할을 탐구합니다.
- **위치 인코딩**: 트랜스포머가 순환 레이어 없이 순차적 데이터를 어떻게 처리하는지 배웁니다.
- **피드포워드 네트워크**: 어텐션 출력을 처리하는 완전 연결 레이어를 학습합니다.

## 강의 세부 사항

- **형식**: 강의 및 트랜스포머 모델 구조에 대한 심층 분석.
- **다룰 주제**:
  - 전통적인 RNN의 한계와 어텐션 메커니즘의 필요성.
  - "Attention is All You Need" 논문에 대한 상세한 설명.
  - 트랜스포머 구성 요소의 시각화 및 분석.
  - 이전 아키텍처에 비해 트랜스포머의 장점에 대한 토론.

## 실습 활동

- **모델 분석**: 트랜스포머 아키텍처를 구성 요소로 분해하고 각 부분의 기능을 이해합니다.
- **시각화 연습**: 도구를 사용하여 어텐션 점수를 시각화하고 모델이 다른 입력 요소에 어떻게 집중하는지 봅니다.
- **선택적 코딩 과제**: PyTorch나 TensorFlow 같은 Python 라이브러리를 사용하여 간단한 어텐션 메커니즘이나 소형 트랜스포머 모델을 구현합니다.

## 자료

- **필수 읽기 자료**:
  - Vaswani 외, "Attention is All You Need" ([논문 링크](https://arxiv.org/abs/1706.03762))
- **보충 자료**:
  - Jay Alammar의 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  - [Transformers from Scratch](https://e2eml.school/transformers.html)
- **동영상**:
  - [DeepLearningAI의 트랜스포머 아키텍처](https://www.youtube.com/watch?v=4Bdc55j80l8)

## 과제

- **주간 실습 과제 (6주차까지 제출)**:
  - **과제**: 트랜스포머 모델을 분석하고 각 구성 요소와 그들이 모델의 언어 처리 능력에 어떻게 기여하는지 설명하세요.
  - **제출**: 다이어그램과 설명을 포함한 보고서 (2-3페이지)
  - **평가 기준**:
    - 설명의 명확성
    - 분석의 깊이
    - 보조 시각 자료의 사용

## 참고 사항

- 이해를 최대화하기 위해 강의 전에 필수 읽기 자료를 완료하세요.
- 대화형 토론 세션을 위해 질문을 준비해 오세요.

## 다음 주 예고

다음 주에는 **LLM API**를 다룰 예정입니다. 트랜스포머와 어텐션 메커니즘에 대해 배운 개념을 실제로 적용해 볼 것입니다. 다음 주제의 기초가 되므로 이번 주 내용을 충분히 이해하고 오시기 바랍니다.

```{tableofcontents}

```
