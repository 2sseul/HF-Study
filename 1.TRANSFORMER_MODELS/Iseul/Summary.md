## NLP란?

- NLP는 인간 언어와 관련된 모든 것을 이해하는데 중점을 둔 언어학 및 머신러닝 분야
- NLP의 목표는 개별 단어를 개별적으로 이해하는 것 + 그 단어의 맥락을 이해하는 것

## Transformer 모델

- 자연어 처리(NLP), 컴퓨터 비전, 오디오 처리 등 다양한 방식으로 모든 종류의 작업을 해결하는데 사용된다

### pipeline()

- 이 함수는 모델(Model) 과 그 모델이 필요로 하는 `전처리(Preprocessing)`, `후처리(Postprocessing)` 단계를 자동으로 연결
- 사용자는 복잡한 과정(`토크나이징, 모델 추론, 결과 해석`)을 몰라도, 텍스트를 입력하면 바로 이해할 수 있는 결과 도출 가능

  ```
  from transformers import pipeline

  classifier = pipeline("sentiment-analysis")
  classifier("I can't wait to go to Singapore")
  ```

  ```
  [{'label': 'POSITIVE', 'score': 0.9962900876998901}]
  ```

- 위처럼 pipeline은 `텍스트 → 토큰화 → 모델 추론 → 라벨 변환` 과정을 자동으로 처리하기 때문에 사용자는 `classifier('문장')`만 호출하면 된다.

- pipeline() 함수에 텍스트를 입력하면 내부적으로는 다음 세 단계를 거치게 된다.

  1. `전처리`: 텍스트를 모델이 이해할 수 있는 형태로 변환
  2. `모델 처리`: 전처리된 데이터가 모델에 전달되어 예측을 수행
  3. `후처리`: 결과를 사람이 이해할 수 있는 형태로 요약

  ### 지원되는 파이프라인 종류

  텍스트 파이프라인

  - [`text-generation`: 텍스트 생성](#text-generation-텍스트-생성)
  - `text-classification`: 텍스트 분류
  - `summarization`: 요약
  - `translation`: 번역
  - [`zero-shot-classification`: 제로샷 분류](#zero-shot-classification-제로샷-분류)
  - `feature-extraction`: 벡터 표현 추출

  이미지 파이프라인

  - `image-to-text`: 이미지에 대한 설명문 텍스트로 생성
  - `image-classification`: 이미지 내 대상을 식별하고 분류
  - `object-detection`: 이미지 내 특정 객체의 위치와 종류 검출

  오디오 파이프라인

  - `automatic-speech-recognition`: 음성 → 텍스트 변환
  - `audio-classification`: 오디오 분류
  - `text-to-speech`: 텍스트 → 음성 변환

  멀티모달(Multimodal)

  - `image-text-to-text`: 텍스트 프롬프트와 이미지를 함께 입력받아 텍스트로 응답

### 제로샷 분류(zero-shot-classification)

제로샷 분류(Zero-shot Classification)는 단순히 기존 데이터와의 유사성을 비교하는 방식이 아니다. 이 기법은 주로 자연어 추론(NLI) 모델을 활용해, 입력 문장을 전제로 두고 `“이 문장은 [라벨]과 관련 있다”`와 같은 가설을 만들어 참(entailment)인지 여부를 판별한다. 이를 통해 훈련에서 보지 못한 새로운 클래스도 자연어 라벨만 정의하면 분류할 수 있다. 핵심은 제로샷 분류는 유사도 기반 접근이 아니라 `추론 기반 접근`이라는 점.

```
classifier = pipeline("zero-shot-classification")

classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
```

- 추가로 학습 & 파인튜닝시키지 않고 사전 학습(pretrained) + NLI 파인튜닝된 모델을 그대로 활용

### 텍스트 생성(Text-generation)

프롬프트(prompt)를 제공하면 모델이 이어지는 문장을 생성해 준다.

```
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

```
[{'generated_text': 'In this course, we will teach you how to create and use custom webpages and plugins with CSS. ... Visual Studio is now available for Windows and Mac. We will also'}]
```

- `num_return_sequences`, `max_length` 옵션으로 출력 개수나 길이 조절 가능

## How do Transformers work?

- `GPT, BERT, T5`와 같은 Transformer 모델들은 모두 언어모델로 학습된 모델들이다.
- 대량의 원시 텍스트 데이터를 기반으로 학습되었고, 학습 과정은 [자기 지도 학습(self-supervised learning)](#자기지도-학습) 방식으로 진행된다.

  #### 자기지도 학습

  > 자기 지도 학습(Self-supervised learning)이란**레이블이 없는 원시 데이터(raw data)로부터 학습 목표를 자동으로 생성해내는 방식**을 말한다. 사람이 직접 데이터를 일일이 라벨링할 필요가 없기 때문에, 대규모 데이터셋을 활용할 수 있다는 장점이 있다. 모델은 입력 데이터의 일부분을 숨기거나 변형한 뒤, 그것을 다시 복원하도록 학습하면서 스스로 언어적・구조적 규칙을 익혀 나간다.

  ```
  ex_1) 이미지에서 일부를 가리고 모델이 가려진 부분을 예측하게 하는 것
  ex_2) 문장의 특정 단어를 [MASK] 토큰으로 가리고 모델이 그 단어를 예측하게 만드는 것
  ```

### 전이학습(Transfer Learning)

한 작업에 대해 훈련된 모델을 사용해 유사한 작업을 수행하는 모델의 기초로 활용하는 딥러닝 접근법. 사전 학습된 모델이 습득한 지식은 "전이"되기 때문에, 전이 학습(transfer learning) 이라는 용어가 사용된다.

1. `사전 학습(Pretraining)`: 모델을 처음부터 학습하는 것
2. `파인튜닝(Fine-tuning)`: 사전 학습된 모델을 가져와 특정 작업에 필요한 데이터로 추가 학습. 파인튜닝은 시간, 비용, 데이터, 환경적 부담 모두를 줄이고 더 나은 성능을 낼 수 있는 합리적인 방법

즉, 사전 학습된 모델(당면 과제와 최대한 가까운 모델)을 활용하고 파인튜닝하는 것이 좋다!

## General Transformer Architecture

![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_blocks-dark.svg)

Transformer는 크게 인코더(Encoder)와 디코더(Decoder) 두 블록으로 이루어져 있다.

- `인코더`: 입력 문장을 받아 의미 표현(피처)을 생성 → `이해`에 최적화
  - `인코더 전용 모델`: 입력 이해가 중요한 작업 (문장 분류, 개체명 인식)
    - 사전 학습은 일반적으로 주어진 문장을 어떻게든 손상시키는 작업을 주로 함
    - ex) 문장에서 무작위 단어를 마스킹하는 것
- `디코더`: 인코더의 표현 + 이전 출력들을 바탕으로 목표 시퀀스를 생성 → `생성`에 최적화
  - `디코더 전용 모델`: 생성 중심 작업 (텍스트 생성, GPT 계열)
    - 주로 다음 단어를 예측하는데 중점
    - 텍스트 생성과 관련된 작업에 적합
- `인코더-디코더 모델(or 시퀀스-투-시퀀스 모델)`: 입력 기반 생성 작업 (번역, 요약, T5 같은 모델)
  - 텍스트 `요약`에 용이

## Attention Layer

Transformer의 핵심은 어텐션(attention) 레이어라고 할 수 있다. 문장에서 각 단어의 의미를 이해할 때, 어떤 단어에 더 집중해야 할지를 학습한다.

```
ex) 영어 문장 "You like this course" → 프랑스어로 번역 시,
"like"은 주어 "You"에 의존하여 올바른 동사 형태를 결정
"this"는 "course"의 성별(남성/여성)에 따라 번역이 달라진다.
```

즉, 어텐션은 `문맥 속 멀리 떨어진 단어도 고려할 수 있도록` 도와준다.

## The original architecture

![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers-dark.svg)

- `인코더(encoder, 양방향)`
  - 입력 문장 전체를 한꺼번에 받기 떄문에 모든 단어(앞, 뒤 구분 없음)를 자유롭게 참고할 수 있다. 그래서 문장 속에서 멀리 떨어진 단어 간의 관계도 잘 잡을 수 있다.
    - ex) “The book that you gave me yesterday was amazing”
    - → "book"을 이해할 때 "was amazing"까지 다 참고할 수 있음
- `디코더(decoder, 단방향, 미래는 마스크 처리)`
  - 출력을 한 단어씩 순차적으로 만들어 내기 때문에 아직 생성하지 않은 미래 단어는 참고할 수 없다. 대신, *과거에 생성한 단어들 + 인코더 출력 전체*를 참고한다.
  - 이때 미래 단어를 못 보도록 Masked Multi-Head Attention을 사용한다. (답을 보고 미리 학습해서 실제 생성할 때 제대로 동작하지 못하는 현상 방지하기 위해)
    - ex) 한국어 번역 중 “나는 사과를 …”까지 만들었다면, 다음 단어 “먹는다”를 예측할 때는 “나는 사과를”까지만 참고 가능.

> 그래서 번역 같은 작업할 때 인코더는 원문 문맥을 다 이해하고, 디코더는 그 이해를 활용해 결과를 순차적으로 생성하는 구조가 된다.

## 언어모델이 동작하는 방식

언어 모델은 주어진 문맥에서 특정 단어가 나올 확률을 예측하도록 훈련된다.

1. `Masked Language Modeling (MLM)`

- 대표 모델: BERT (인코더 기반 모델) - 문맥 이해에 강함
- 방법: 입력 문장에서 일부 단어(토큰)를 무작위로 가려놓고 [MASK] 토큰으로 대체
- 모델의 목표: 주변 단어들을 참고하여 가려진 단어를 맞히는 것.
- 특징: 단어의 앞뒤 양방향 문맥을 모두 학습할 수 있다.

```
입력: 나는 맥도날드에서 [MASK]를 먹었다.
출력: 햄버거
```

2. `Causal Language Modeling (CLM)`

- 대표 모델: GPT (디코더 기반 모델) - 다음 단어 예측에 용이
- 방법: 문장의 앞부분을 보고 다음 단어를 예측
- 모델의 목표: 이전에 생성된 단어들만 참고하여 순서대로 새로운 단어를 생성
- 특징: 과거 문맥만 활용할 수 있고 미래 단어는 볼 수 없습니다.

```
입력: 나는 사과를
출력: 먹었다
```

## 편향과 한계(Bias and limitations)

대형 언어 모델은 방대한 양의 텍스트 데이터로 사전 학습되고 보통 연구자들은 보통 인터넷에 공개된 텍스트를 크롤링(scraping)해서 사용한다.
이 과정에서 좋은 데이터뿐만 아니라 왜곡된 데이터, 차별적인 발언, 혐오 표현 등도 함께 들어가게 되기 때문에, 모델이 학습한 지식 속에는 사회적 편견과 차별적 요소가 그대로 반영될 수 있다.

```
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")

print(unmasker("This man works as a [MASK]."))
# ['lawyer', 'carpenter', 'doctor', 'waiter', 'mechanic']

print(unmasker("This woman works as a [MASK]."))
# ['nurse', 'waitress', 'teacher', 'maid', 'prostitute']

```

- `man`의 경우: 변호사, 목수, 의사, 웨이터, 정비공 등 비교적 다양한 직업이 나옴
- `woman`의 경우: 간호사, 웨이트리스, 교사, 가정부, 심지어 매춘부까지 포함됨
- 즉, 동일한 문맥인데도 성별에 따라 직업을 특정 성별에 묶는 편향적 결과가 나타난다.

> BERT는 사실 인터넷 전체를 긁어 모은 데이터가 아니라, 상대적으로 중립적이라고 여겨지는 `Wikipedia + BookCorpus`로 학습된 모델이다. 그럼에도 불구하고 이런 성차별적 편향이 드러난다는 건, 편향이 얼마나 쉽게 내재화될 수 있는지 보여준다.

- 사전학습 모델을 그대로 사용하거나, 파인튜닝을 하더라도 내재된 편향은 완전히 사라지지 않는다.
- 따라서 실제 서비스에 도입할 때는, 모델이 성차별적, 인종차별적, 동성애 혐오적인 출력을 할 수 있다는 가능성을 항상 염두에 둬야 한다.
- 편향 완화를 위해 추가적인 필터링, 데이터 정제, 안전장치 설계가 반드시 필요하다.
