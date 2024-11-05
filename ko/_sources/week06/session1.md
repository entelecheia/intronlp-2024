# 6주차 세션 1: LLM API 소개 및 OpenAI API 사용법

## 1. 대규모 언어 모델 API 소개

대규모 언어 모델(LLM) API는 개발자들이 복잡한 모델을 직접 훈련하거나 호스팅할 필요 없이 강력한 자연어 처리 기능에 접근할 수 있게 해줍니다. 이러한 API는 다음과 같은 다양한 기능을 제공합니다:

- 텍스트 생성
- 텍스트 완성
- 질문 답변
- 요약
- 감정 분석
- 언어 번역

가장 두드러진 LLM API 중 하나는 OpenAI에서 제공하는 것으로, 이번 세션에서 중점적으로 다룰 예정입니다.

## 2. OpenAI API 개요

OpenAI API를 통해 개발자들은 GPT-3.5와 GPT-4를 포함한 다양한 모델에 접근하여 여러 자연어 처리 작업을 수행할 수 있습니다. 주요 특징은 다음과 같습니다:

- **다양한 모델**: 여러 작업과 성능 수준에 최적화된 다양한 모델
- **사용자 정의 가능한 매개변수**: 온도와 최대 토큰 수 같은 생성 매개변수 제어
- **미세 조정 기능**: 특정 사용 사례에 맞춰 모델을 사용자 정의할 수 있는 능력
- **중재 도구**: 안전과 규정 준수를 위한 내장 콘텐츠 필터링

## 3. OpenAI API 설정하기

OpenAI API를 사용하려면 다음 단계를 따르세요:

- a) https://platform.openai.com/에서 OpenAI 계정에 가입합니다
- b) API 키 섹션으로 이동하여 새로운 비밀 키를 생성합니다
- c) API 키를 안전하게 보관합니다 (절대 공개적으로 공유하거나 노출하지 마세요)
- d) OpenAI Python 라이브러리를 설치합니다:

```bash
pip install openai
```

e) API 키를 환경 변수로 설정합니다:

```bash
export OPENAI_API_KEY='여기에-당신의-api-키를-넣으세요'
```

## 4. 첫 번째 API 호출하기

OpenAI API를 사용하여 텍스트를 생성하는 간단한 Python 스크립트를 만들어 봅시다:

```python
import openai

# OpenAI 클라이언트 초기화
client = openai.OpenAI()

# API 호출하기
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 도움이 되는 어시스턴트입니다."},
        {"role": "user", "content": "인공지능이란 무엇인가요?"}
    ]
)

# 생성된 응답 출력하기
print(response.choices[0].message.content)
```

이 스크립트는 다음을 수행합니다:

1. OpenAI 라이브러리를 임포트합니다
2. OpenAI 클라이언트를 초기화합니다
3. 채팅 완성 엔드포인트에 API 호출을 합니다
4. 생성된 응답을 출력합니다

## 5. API 응답 이해하기

API 응답은 다양한 필드를 포함하는 JSON 객체입니다. 주요 필드는 다음과 같습니다:

- `id`: 응답의 고유 식별자
- `object`: 반환된 객체의 유형 (예: "chat.completion")
- `created`: 응답이 생성된 시간의 Unix 타임스탬프
- `model`: 사용된 모델의 이름
- `choices`: 생성된 응답을 포함하는 배열
- `usage`: 요청에 대한 토큰 사용 정보

## 6. API 매개변수

API 호출을 할 때 생성 프로세스의 다양한 측면을 제어할 수 있습니다. 중요한 매개변수는 다음과 같습니다:

- `model`: 사용할 모델을 지정합니다 (예: "gpt-4o-mini", "gpt-4o")
- `messages`: 대화 기록을 나타내는 메시지 객체의 배열
- `max_tokens`: 생성할 최대 토큰 수
- `temperature`: 무작위성을 제어합니다 (0에서 2 사이, 낮을수록 더 결정적)
- `top_p`: 온도의 대안으로, 핵 샘플링을 사용합니다
- `n`: 각 프롬프트에 대해 생성할 완성 수
- `stop`: API가 더 이상 토큰을 생성하지 않을 시퀀스

추가 매개변수를 사용한 예시:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 도움이 되는 어시스턴트입니다."},
        {"role": "user", "content": "AI에 대한 짧은 시를 써주세요."}
    ],
    max_tokens=50,
    temperature=0.7,
    n=2,
    stop=["\n\n"]
)

for choice in response.choices:
    print(choice.message.content)
    print("---")
```

이 예시는 AI에 대한 두 개의 짧은 시를 생성하며, 각각 50 토큰으로 제한되고, 0.7의 온도를 사용하며, 이중 개행에서 생성을 중지합니다.

## 7. 모범 사례와 비율 제한

OpenAI API를 사용할 때 다음 사항을 명심하세요:

- **비율 제한**: OpenAI는 API 호출에 대한 비율 제한을 적용합니다. 사용량을 모니터링하고 적절한 오류 처리를 구현하세요.
- **비용**: API 사용은 처리된 토큰 수에 따라 청구됩니다. 비용을 추정하고 예상치 못한 요금을 피하기 위해 사용량 제한을 설정하세요.
- **프롬프트 엔지니어링**: 모델에서 최상의 결과를 얻기 위해 효과적인 프롬프트를 작성하세요.
- **오류 처리**: API 오류와 네트워크 문제를 관리하기 위해 강력한 오류 처리를 구현하세요.

## 8. 토큰화 기초

LLM API를 사용할 때 토큰화를 이해하는 것이 중요합니다:

- 토큰은 모델이 처리하는 기본 단위로, 대략 단어의 일부에 해당합니다.
- 영어 텍스트는 일반적으로 토큰당 약 4자로 분해됩니다.
- 입력(프롬프트)과 출력(완성) 모두 토큰을 소비합니다.
- 다른 모델은 서로 다른 토큰 제한을 가집니다 (예: gpt-4o-mini의 경우 4096).

OpenAI 토크나이저를 사용하여 토큰을 계산할 수 있습니다:

```python
from openai import OpenAI

client = OpenAI()

def num_tokens_from_string(string: str, model_name: str) -> int:
    """텍스트 문자열의 토큰 수를 반환합니다."""
    encoding = client.tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

text = "안녕하세요, 세상! 오늘 기분이 어떠신가요?"
model = "gpt-4o-mini"
token_count = num_tokens_from_string(text, model)
print(f"이 텍스트는 {model} 모델에 대해 {token_count}개의 토큰을 포함합니다")
```

이 함수는 `tiktoken` 라이브러리(OpenAI 패키지에 포함됨)를 사용하여 특정 모델에 대한 토큰을 정확하게 계산합니다.

## 결론

이 세션에서는 OpenAI API에 초점을 맞춰 LLM API 작업의 기초를 소개했습니다. API 설정, 기본 호출 만들기, 응답 이해하기, 그리고 토큰화와 같은 중요한 개념을 다뤘습니다. 다음 세션에서는 고급 사용 패턴을 더 깊이 살펴보고 텍스트 생성을 제어하기 위한 다양한 샘플링 방법을 탐구할 것입니다.

## 참고 자료

- [1] https://blog.allenai.org/a-guide-to-language-model-sampling-in-allennlp-3b1239274bc3?gi=95ecfb79a8b8
- [2] https://huyenchip.com/2024/01/16/sampling.html
- [3] https://leftasexercise.com/2023/05/10/mastering-large-language-models-part-vi-sampling/
- [4] https://platform.openai.com/docs/quickstart
