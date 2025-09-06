# 1. Transformer Models

---

## 트랜스포머로 무엇을 할 수 있나요?

- **The pipeline function**
    
    ```python
    from transformers import pipeline
    
    nlp = pipeline(task='작업명', model='모델명')
    result = nlp("입력 텍스트")
    print(result)
    ```
    
    - `Transformers` 라이브러리의 가장 상위레벨의 API
    - Pre-Processing ⇒ Model ⇒ Post-Processing
- ‘작업명’
    - `feature-extraction`
    - `fill-mask`
    - `ner`
    - `question-analysis`
    - `summarization`
    - `text-generation`
    - `translation`
    - `zero-shot-classification`

## 트랜스포머는 어떻게 동작하나요?

- 트랜스포머 기반 모델(GPT, BERT, BART, T5)들은 언어 모델로, 대규모 텍스트 데이터를 활용해 자가 지도 학습 방식(self-supervised learning)으로 학습된 모델이다.
- 자가 지도 학습 (self-supervised learning)
    - 지도학습 (Supervised Learning) : 입력과 출력(정답)
    - 비지도 학습 (Unsupervised Learning) : 입력 ⇒ 데이터의 숨겨준 구조, 군집, 패턴
    - 자가 지도 학습 (Self-supervised Learning) : 데이터 자체에서 학습 목표(레이블)을 만들어 학습
        
        ```python
        “나는 오늘 [MASK]를 먹었다” → [MASK] 예측
        ```
        
        1. 정답 문장이 있음
        2. 일부 단어를 가림
        3. 모델이 [MASK]를 예측
        4. 오차 계산
        5. 학습
- “아래 그래프에서 볼 수 있듯이 이는 환경 오염 문제로 이어지기도 합니다.”
    
    → 모델을 처음부터 학습시키는 사전 학습보다 이미 학습된 대규모 언어모델을 이용하여 가중치를 쌓아올리는 전이학습이 환경오염에 부정적인 영향을 줄일 수 있는 방법이다.
    

## 트랜스포머의 구조

![image.png](attachment:0d234caa-2a1f-4219-b4fe-28ef4dac4a8c:image.png)

- 인코더 : “입력”, 입력 데이터를 분석, 이해
- 디코더 : “출력”, 인코더의 (특징) 표현과 또 다른 입력을 활용해 타겟 시퀀스를 생성
- GPT-계열 (*Auto-regressive* 트랜스포머 모델로도 불림)
    
    → 디코더만 사용, 단어 예측, 생성
    
- BERT-계열 (*Auto-encoding* 트랜스포머 모델로도 불림)
    
    → 인코더만 사용, 문장 분류 등
    
- BART/T5-계열 (*Sequence-to-sequence* 트랜스포머 모델로도 불림)
    
    → 인코더-디코더를 모두 사용함. 주로 요약, 번역 등
    

## 어텐션 (Attention)

- 이 레이어가 단어의 표현을 다룰 때 입력으로 넣어준 문장의 특정 단어에 어텐션(주의)을 기울이고 특정 단어는 무시하도록 알려준다는 사실을 기억하셔야 합니다.
    - 이로써 문맥(context)를 파악할 수 있다.
    - **Query, Key, Value** 벡터로 문맥을 파악
        
        Q는 현재 단어의 정보, K는 문장 속 모든 단어의 정보
