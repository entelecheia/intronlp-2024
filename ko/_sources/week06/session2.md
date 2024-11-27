# 6주차 세션 2: 샘플링 방법과 텍스트 생성

## 개요

이번 세션에서는 대규모 언어 모델(LLM) API의 고급 사용법을 더 깊이 살펴보고, 생성된 텍스트의 무작위성과 창의성을 제어하는 샘플링 방법에 초점을 맞출 것입니다. `temperature`, `top_p`, `top_k` 등의 매개변수가 모델의 출력에 어떤 영향을 미치는지 살펴볼 것입니다. 실습을 통해 다양한 자연어 처리 작업에서 원하는 결과를 얻기 위해 이러한 매개변수를 미세 조정하는 방법을 배울 것입니다.

---

## 1. 이전 세션 복습

지난 세션에서 우리는:

- **OpenAI API 설정**: 인증 방법과 기본 API 호출 방법을 배웠습니다.
- **토큰화 탐구**: 텍스트가 어떻게 토큰으로 분해되는지와 API 사용에 미치는 영향을 이해했습니다.
- **초기 API 호출 수행**: 프롬프트를 기반으로 간단한 텍스트를 생성했습니다.
- **모범 사례 논의**: 오류 처리, 비용 관리, 비율 제한에 대해 다뤘습니다.

---

## 2. 샘플링 방법 이해하기

샘플링 방법은 텍스트 생성 중 언어 모델이 다음 단어(토큰)를 선택하는 방식을 결정합니다. 이러한 매개변수를 조정하면 출력의 창의성과 무작위성을 제어할 수 있습니다.

### **2.1 온도 샘플링**

- **정의**: 소프트맥스 함수를 적용하여 확률을 생성하기 전에 로짓(원시 출력)을 조정하는 매개변수(`temperature`)입니다.
- **범위**: 일반적으로 0.0에서 1.0 사이이지만, 더 높을 수 있습니다.
  - **낮은 온도 (<0.5)**: 모델을 더 결정적이고 집중적으로 만듭니다.
  - **높은 온도 (>0.7)**: 무작위성과 창의성을 증가시킵니다.

#### **수학적 설명**

각 토큰 $ i $의 확률 $ p_i $는 다음과 같이 계산됩니다:

$$
p_i = \frac{\exp{\left( \frac{z_i}{T} \right)}}{\sum_j \exp{\left( \frac{z_j}{T} \right)}}
$$

- $ z_i $는 토큰 $ i $의 로짓입니다.
- $ T $는 온도입니다.

$ T $가 증가할수록 확률 분포가 더 균일해집니다.

#### **다이어그램: 확률 분포에 대한 온도의 영향**

```mermaid
graph LR
    A[로짓] --> B[온도로 나누기]
    B --> C[소프트맥스 함수]
    C --> D[확률 분포]
```

### **2.2 Top-p (핵) 샘플링**

- **정의**: 모든 가능한 토큰을 고려하는 대신, 누적 확률이 임계값 `p`를 초과하는 가장 작은 토큰 집합만을 고려합니다.
- **매개변수**: `top_p`는 0.0에서 1.0 사이의 값을 가집니다.
  - **낮은 Top-p (<0.5)**: 가장 가능성이 높은 토큰들만 고려됩니다.
  - **높은 Top-p (~1.0)**: 더 넓은 범위의 토큰들을 고려합니다.

#### **시각적 설명**

- **누적 확률 곡선**: 확률로 정렬된 토큰들을 플롯팅합니다.
- **Top-p 임계값**: 누적 확률 `p`에서의 수직선입니다.

### **2.3 Top-k 샘플링**

- **정의**: 모델이 확률이 가장 높은 상위 `k`개의 토큰만을 고려합니다.
- **매개변수**: `top_k`는 정수입니다.
  - **낮은 Top-k**: 가장 가능성이 높은 토큰들로 선택을 제한합니다.
  - **높은 Top-k**: 더 많은 다양성을 허용합니다.

### **2.4 빈도 및 존재 페널티**

- **빈도 페널티**: 생성된 텍스트에서의 빈도에 기반하여 토큰이 다시 선택될 가능성을 줄입니다.
- **존재 페널티**: 이미 등장한 토큰에 불이익을 주어 모델이 새로운 주제를 도입하도록 장려합니다.

---

## 3. 코드를 통한 실제 예시

이제 이러한 샘플링 방법을 실제로 적용해 보겠습니다.

### **3.1 환경 설정**

OpenAI 라이브러리가 설치되어 있고 API 키가 설정되어 있는지 확인하세요.

```bash
export OPENAI_API_KEY='여기에-당신의-api-키를-넣으세요'
```

```python
import openai

client = openai.OpenAI()

def generate_text(prompt, **kwargs):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 도움이 되는 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        **kwargs
    )
    return response.choices[0].message.content
```

### **3.2 온도 실험**

온도 변화가 출력에 어떤 영향을 미치는지 살펴보겠습니다.

#### **프롬프트**

```python
prompt = "먼 옛날 먼 나라에 살았던"
```

#### **코드 예시: 온도 변경**

```python
temperatures = [0.2, 0.7, 1.0]

for temp in temperatures:
    text = generate_text(
        prompt=prompt,
        max_tokens=50,
        temperature=temp,
        n=1,
        stop=None
    )
    print(f"온도 {temp}:\n{text}\n{'-'*40}")
```

#### **예상 출력**

- **온도 0.2**: 더 예측 가능하고, 덜 창의적입니다.

  ```
  온도 0.2:
  용감하고 친절한 젊은 왕자가 있었습니다. 그는 숲을 탐험하고 새로운 생물들을 만나는 것을 좋아했습니다. 어느 날, 그는 아름다운 꽃들과 노래하는 새들로 가득한 마법의 정원으로 이어지는 숨겨진 길을 발견했습니다.
  ----------------------------------------
  ```

- **온도 0.7**: 균형 잡힌 창의성을 보입니다.

  ```
  온도 0.7:
  지혜로 유명한 신비로운 늙은 마법사가 있었습니다. 사람들은 그의 조언을 구하기 위해 사방에서 찾아왔지만, 그는 좀처럼 자신의 비밀을 드러내지 않았습니다. 어느 날, 한 어린 소녀가 그의 관심을 끄는 질문을 가지고 그를 찾아왔습니다.
  ----------------------------------------
  ```

- **온도 1.0**: 매우 창의적이지만, 가능한 덜 일관성 있습니다.
  ```
  온도 1.0:
  금 대신 도자기 찻잔을 수집하는 용이 있었습니다. 마을 사람들은 이를 기이하게 여겼지만, 용은 신경 쓰지 않았습니다. 그는 매일 오후 별들에 대한 고대 두루마리를 읽으며 차를 마셨습니다.
  ----------------------------------------
  ```

### **3.3 Top-p 샘플링 탐구**

`top_p`를 조정하면 출력에 어떤 영향을 미치는지 살펴보겠습니다.

#### **코드 예시: Top-p 변경**

```python
top_p_values = [0.3, 0.7, 1.0]

for p in top_p_values:
    text = generate_text(
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
        top_p=p,
        n=1,
        stop=None
    )
    print(f"Top-p {p}:\n{text}\n{'-'*40}")
```

#### **예상 출력**

- **Top-p 0.3**: 가장 가능성 높은 토큰들로 제한됩니다.

  ```
  Top-p 0.3:
  용감하고 친절한 젊은 왕자가 있었습니다. 그는 왕국을 탐험하고 백성들을 돕는 것을 좋아했습니다. 어느 날, 그는 자신의 땅에 번영을 가져올 수 있는 전설적인 보물을 찾기 위한 여정을 떠났습니다.
  ----------------------------------------
  ```

- **Top-p 0.7**: 더 다양한 단어 선택을 보입니다.

  ```
  Top-p 0.7:
  꿈으로 가득 찬 마음을 지닌 호기심 많은 모험가가 있었습니다. 그녀는 미지의 것을 탐험하고 숨겨진 경이로움을 발견하기를 갈망했습니다. 지도를 손에 들고, 그녀는 자신의 인생을 영원히 바꿀 여정을 시작했습니다.
  ----------------------------------------
  ```

- **Top-p 1.0**: 가능한 모든 토큰의 전체 범위를 고려합니다 (top-p를 사용하지 않는 것과 유사).
  ```
  Top-p 1.0:
  귀를 기울이는 이들에게 비밀을 속삭이는 말하는 나무가 있었습니다. 여행자들은 먼 곳에서부터 그 나무의 고대의 시간과 잊혀진 영역에 대한 이야기를 듣기 위해 찾아왔고, 각 이야기는 전보다 더 매혹적이었습니다.
  ----------------------------------------
  ```

### **3.4 온도와 Top-p 결합하기**

두 매개변수를 함께 사용하여 출력을 미세 조정할 수 있습니다.

#### **코드 예시**

```python
text = generate_text(
    prompt=prompt,
    max_tokens=50,
    temperature=0.9,
    top_p=0.6,
    n=1,
    stop=None
)
print(f"결합된 매개변수:\n{text}")
```

#### **설명**

- **온도 0.9**: 높은 창의성
- **Top-p 0.6**: 높은 온도에도 불구하고 가장 가능성 있는 토큰들로 선택을 제한합니다.

---

## 4. OpenAI API의 고급 매개변수

샘플링 방법 외에도 모델의 출력을 제어할 수 있는 다른 매개변수들이 있습니다.

### **4.1 출력 길이 제어하기**

- **`max_tokens`**: 생성된 응답의 최대 토큰 수를 설정합니다.
- **제한사항**: 총 토큰 수(프롬프트 + 완성)는 모델의 컨텍스트 길이 내에 있어야 합니다(예: `text-davinci-003`의 경우 4,096 토큰).

#### **예시**

```python
text = generate_text(
    prompt="양자 컴퓨팅을 간단한 용어로 설명해주세요.",
    max_tokens=50
)
```

### **4.2 중지 시퀀스 사용하기**

- **목적**: 모델에게 텍스트 생성을 언제 중지할지 알려줍니다.
- **매개변수**: `stop`은 문자열 또는 문자열 리스트를 받습니다.

#### **예시**

```python
text = generate_text(
    prompt="운동의 세 가지 이점을 나열하세요:\n1.",
    max_tokens=50,
    stop=["\n"]
)
text = response.choices[0].text.strip()
print(f"생성된 목록:\n1.{text}")
```

#### **예상 출력**

```
생성된 목록:
1. 심혈관 건강을 개선합니다.
```

### **4.3 출력 형식 지정하기**

- **지시적 프롬프트**: 모델에게 특정 방식으로 출력을 형식화하도록 요청합니다.
- **JSON 출력**: 구조화된 데이터 추출에 유용합니다.

#### **예시: JSON 생성하기**

```python
prompt = """
다음 정보를 추출하여 JSON 형식으로 제공하세요:
- 이름
- 나이
- 직업

텍스트: "홍길동은 서울에 사는 29세의 소프트웨어 엔지니어입니다."
"""

text = generate_text(
    prompt=prompt,
    max_tokens=100,
    temperature=0,
    stop=None
)
print(text)
```

#### **예상 출력**

```json
{
  "이름": "홍길동",
  "나이": 29,
  "직업": "소프트웨어 엔지니어"
}
```

---

## 5. 실습 연습

**목표**: 샘플링 매개변수를 조정하여 창의적인 이야기 프롬프트를 생성하는 스크립트를 작성합니다.

### **지시사항**

1. **기본 프롬프트 선택**: 예를 들어, "감정을 가진 로봇들의 세계에서, 한 로봇이 시작합니다..."
2. **다양한 값으로 실험**:
   - `temperature`를 0.5에서 1.2 사이로 변경합니다.
   - `top_p`를 0.5에서 1.0 사이로 조정합니다.
3. **출력 관찰**:
   - 변경사항이 창의성과 일관성에 어떤 영향을 미치는지 주목합니다.
4. **결과 문서화**:
   - 매개변수 값과 해당 출력을 기록합니다.

### **샘플 코드**

```python
prompt = "감정을 가진 로봇들의 세계에서, 한 로봇이 시작합니다"

temperatures = [0.5, 0.9, 1.2]
top_p_values = [0.5, 0.8, 1.0]

for temp in temperatures:
    for p in top_p_values:
        text = generate_text(
            prompt=prompt,
            max_tokens=60,
            temperature=temp,
            top_p=p,
            n=1,
            stop="."
        )
        print(f"온도: {temp}, Top-p: {p}\n{text}\n{'-'*50}")
```

---

## 6. 일반적인 문제와 문제 해결

### **반복적이거나 무의미한 출력**

- **원인**: 너무 낮거나 높은 온도.
- **해결책**: `temperature`와 `top_p`를 조정하여 균형을 찾습니다.

### **API 오류**

- **InvalidRequestError**: 매개변수가 허용 범위 내에 있는지 확인합니다.
- **RateLimitError**: 지수 백오프를 사용한 재시도를 구현합니다.

### **형식 문제**

- **원치 않는 텍스트**: `stop` 시퀀스를 사용하여 모델이 원하는 지점 이상으로 생성하지 않도록 합니다.
- **잘못된 출력 형식**: 프롬프트에 명확한 지침과 예시를 제공합니다.

---

## 7. 요약 및 주요 takeaways

- **샘플링 방법**: `temperature`, `top_p`, `top_k`의 이해는 텍스트 생성 제어에 중요합니다.
- **매개변수 조정**: 이러한 매개변수를 조정하면 창의성과 일관성의 균형을 맞출 수 있습니다.
- **고급 제어**: `max_tokens`, `stop` 시퀀스, 형식 지정 지침을 사용하여 출력을 정제합니다.
- **실습**: 다양한 설정으로 실험해보는 것이 그 효과를 배우는 가장 좋은 방법입니다.

---

## 8. 다음 주 과제

1. **샘플링 방법 보고서**:

   - 선택한 프롬프트를 사용하여 `temperature`와 `top_p`의 다양한 조합을 실험합니다.
   - 코드 스니펫과 생성된 출력을 포함하여 발견한 내용을 요약한 짧은 보고서를 준비합니다.

2. **제어된 출력 만들기**:

   - OpenAI API를 사용하여 최소 다섯 개의 항목을 생성하는 스크립트를 작성합니다 (예: "2024년 상위 5개 프로그래밍 언어").
   - 출력이 올바르게 형식화되고 다섯 개 항목을 나열한 후 중지되도록 합니다.

3. **구조화된 데이터 추출**:
   - API를 사용하여 텍스트 블록에서 특정 정보를 추출하고 JSON 형식으로 출력합니다.
   - 프롬프트, 코드, 결과를 공유합니다.

---

## 결론

샘플링 방법을 이해하고 효과적으로 사용하는 것은 LLM API의 잠재력을 최대한 활용하는 데 필수적입니다. 이러한 매개변수를 마스터함으로써 창의성, 일관성, 형식에 대한 특정 요구사항을 충족하는 텍스트를 생성할 수 있습니다. 계속해서 실험하고 이 강력한 도구들이 제공하는 가능성을 탐구하세요.