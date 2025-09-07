## Behind the pipeline

#### Tokenizer

- 정의
    - 텍스트(비정형 데이터)를 직접 처리할 수 없으므로, 모델이 이해할 수 있는 숫자로 변환하는 과정
- 역할
    - 토큰화: 입력 텍스트를 토큰(서브 워드) 또는 기호로 나눔
    - 정수 매핑: 각 토큰을 숫자 형태로 매핑
    - 추가 입력: 모델에 유용한 추가적인 입력을 더함
- 출력
    - `input_ids`, `token_type_ids`, `attention_mask`라는 세 개의 키를 포함하는 딕셔너리
        - `input_ids`는 각 문장의 고유한 토큰 식별자인 정수들로 이루어진 행
            - 행의 수는 문장의 수에 의존함
            - decode를 통해 원래 문장으로 되돌릴 수 있음
                - e.g., tokenizer.decode(encoding[”input_ids”])
        - `token_type_ids`는 문장 간 구분을 해줌
            - e.g., `tokenizer(["How are you?", "I'm fine, thank you!"])`은 id가 0으로 채워짐
                - 여러 문장을 독립적으로 처리
                - 각 문장이 독립적이기 때문에 id를 구분할 필요가 없음
                - e.g., 감성 분석(각 문장이 긍정적인지 부정적인지), 텍스트 분류(각 문장을 특정 카테고리로 분류)
            - e.g., `tokenizer("How are you?", "I'm fine, thank you!”)`은 id가 0, 1로 채워짐
                - 여러 문장을 연관성있게 처리
                - 인과 관계를 위해 id를 구분해야 함
                - e.g., 질문-답변, 자연어 추론(한 문장에서 다른 문장을 의미하는지 판단)
        - `attention_mask`문장 내 활용할 토큰을 설정
            - 활용할 토큰은 1, 패딩 토큰은 0
            - Zero-Padding된 패딩 토큰은 무시
- 부가
    - 이러한 전처리 과정은 모델을 사전 훈련할 때와 동일한 방식으로 이루어져야 함
        - 사전 학습된 모델을 활용할 경우 Model Hub에서 별도로 로드를 해야 함
        - Transformers는 Tensor 데이터를 입력으로 받으므로 `return_tensors = True`인자를 입력해야 함
        - 입력 문장들의 차원이 다를 경우 `padding = True`를 통해 가장 긴 문장을 기준으로 자동 Zero-Padding을 함
            - `padding = “longest”`: 최대 길이를 기준으로 패딩
            - `padding = “max_length”`: 모델의 최대 길이를 기준으로 패딩
            - `padding = “max_length”, max_length = 8`: 지정된 최대 길이를 기준으로 패딩
        - 모델이 처리할 수 있는 토큰 수를 넘었을 때, `truncation = True`를 통해 입력 데이터를 자를 수 있음
            - `truncation=True`: 모델의 최대 길이보다 긴 시퀀스를 잘라냄
            - `max_length = 8, truncation = True`: 지정된 최대 길이보다 긴 시퀀스를 잘라냄
        - 반환 타입
            - `return_tensors = “pt”`: Pytorch 텐서를 반환
            - `return_tensors = “np”`: Numpy 배열을 반환
    - `AutoTokenizer`과 `from_pretrained()` 메서드에 모델 체크포인트 이름을 인자로 주어 해당 모델의 토크나이저 정보를 가져올 수 있음
        - 캐시에 저장되므로 이후 별도의 로드가 필요 X
    - [CLS], [SEP]와 같은 특수 토큰을 통해 문장 간 경계를 더 잘 표현할 수 있음
        - `tokenizer.tokenize`를 사용할 경우, 수동으로 특수 토큰을 추가해줘야 함
        - `tokenize`를 사용할 경우, 자동으로 특수 토큰을 처리
    - `tokenizer`은 자동으로 배치 처리를 하는 반면, `tokenizer.tokenize`로 수동 토크나이징을 할 경우, Batch를 고려해야 함
        - e.g., `tokenizer.tokenize(”I’m junho”)` # error occur when data input model because doen’t process batch dimension
            - `tokens = tokenizer.tokenize(”I’m junho”)` → `ids = tokenizer.convert_tokens_to_ids(tokens)` → `torch.tensor([ids])` # not work when torch.tensor(ids)
        - 여러 배치 데이터를 처리할 경우, `[ids, ids]`와 같이 배치 데이터를 묶어줄 수 있음
            - 이 때, 입력 데이터의 길이가 서로 다를 경우 패딩 처리를 해줘야 함

#### Post Processing

- 정의
    - Vanilla Transformer의 출력인 고차원 특징 벡터(Hidden-State라 불림)에서 Head를 추가하여 작업에 맞게 활용
        - LAB에서 기본 모델은 기본적인 아키텍처를 사용하기 때문에 Hidden State를 출력
            - 출력은 Named Tuple이나 Dictionary 형태
            - Hiddden State의 차원
                - 배치 크기: 한 번에 처리되는 시퀀스(문장)의 개수
                - 시퀀스 길이: 시퀀스를 숫자로 표현한 것의 길이
                - 히든 사이즈: 각 모델 입력의 벡터 차원
        - 반면 Head 모델은 하나 혹은 소수의 선형 레이어를 통과하여 다른 차원으로 투영됨
- Head의 종류
    - Model: Hidden State를 출력
    - ForCausalLM: 텍스트를 생성하여 출력
    - ForMaskedLM: 중간에 마스킹된 텍스트를 예측하여 출력
    - ForMultipleChoise: 주어진 여러 선택지 중 가장 올바른 답을 출력
    - ForQuestionAnswering: 주어진 질문에 대한 답을 출력
    - ForSequenceClassification: 전체 문장을 하나의 카테고리 레이블로 출력
    - ForTokenClassification: 각 개별 토큰에 대해 레이블을 출력
    - etc.

## Models

#### Loading & Saving

- 모델 로드
    - 자동 로드
        - e.g., `AutoModel.from_pretrained`
    - 모델 타입으로 로드
        - e.g., `BertModel.from_pretrained`
- 모델 저장
    - config.json, pytorch_model.bin 파일이 저장됨
        - config.json: 메타 데이터와 체크포인트 등이 저장
        - pytorch_model.bin: 모델 파라미터가 저장
    - 로컬
        - 저장
            - `model.save_pretrained(”directory”)`
        - 불러오기
            - `AutoModel.from_pretrained(”directory”)`
    - Huggingface Hub
        - Hub 로그인
            - `from huggingface_hub import notebook_login` → `notebook_login()`
        - 저장
            - `model.push_to_hub(”My Model”)`
        - 불러오기
            - `AutoModel.from_pretrained(”username/My Model”)`

## Optimized Inference Deployment


#### TGL(Text Generation Inference)

- 프로덕션 환경에서 안정적이고 예측 가능하도록 설계되었으며, 고정된 시퀀스 길이를 사용하여 메모리 사용량을 일관되게 유지
- Flash Attention2
    - 트랜스포머 모델의 어텐션 메커니즘을 최적화하는 기술로, 메모리 대역폭 병목 현상을 해결
        - 기존 어텐션 메커니즘은 시퀀스 길이가 길어질수록 계산 복잡도와 메모리 사용량이 이차적으로 증가
        - 데이터 전송을 반복하여 GPU가 유휴 상태에 빠지는 병목 현상을 초래
        - 데이터를 한 번만 SRAM에 로드하고 모든 계산을 SRAM에서 수행함으로써, 메모리 전송을 최소화
- Continuous Batching

#### vLLM

- 모델의 메모리를 더 작은 블록으로 나누어, 다양한 크기의 요청을 유연하게 처리하고 메모리 공간 낭비를 줄임
- 여러 요청 간 메모리를 공유하고 메모리 단편화(Fragmentation)을 줄임
- PageAttention
    - KV 캐시 메모리 관리를 통해 병목 현상을 해결
    - 기존 모델은 텍스트 생성 중 중복 계산을 줄이기 위해 생성된 각 토큰에 대한 어텐션 키와 값(KV 캐시)을 저장하여 긴 시퀀스나 여러 동시 요청이 있을 때 메모리 병목 현상이 일어남
    - 캐시 메모리 관리 방법
        - 메모리 페이징: KV 캐시를 하나의 큰 블록으로 다루는 대신, 운영 체제의 가상 메모리와 유사하게 고정된 크기의 페이지로 분할
        - 비연속적 저장: 페이지는 GPU 메모리에 연속적으로 저장할 필요가 없어서 더 유연하게 메모리 할당이 가능
        - 페이지 테이블 관리: 페이지 테이블이 어떤 페이지에 어떤 시퀀스가 속하는지 추적하여 효율적으로 조회 및 접근이 가능
        - 메모리 공유: 병렬 샘플링 시 프롬프트의 KV 캐시를 저장하는 페이지를 여러 시퀀스에서 공유

#### llama.cpp

- 일반 소비자용 하드웨어에서 LLaMA 모델을 실행하기 위해 설계된 최적화된 C/C++ 구현체
- 양자화
    - 모델의 가중치를 32비트나 16비트 부동 소수점에서 8비트 정수, 4비트 또는 그 이하의 낮은 정밀도 형식으로 줄이는 것
    - 품질 손실을 최소화하면서 메모리 사용량을 크게 줄이고 추론 속도를 향상
    - 특징
        - 다양한 양자화 수준: 8비트, 4비트, 3비트, 2비트 등 다양한 양자화를 지원
        - GGML/GGUF 방식: 양자화된 추론에 최적화된 맞춤 텐서 형식을 사용
        - 혼합 정밀도: 모델의 각기 다른 부분에 다른 양자화 수준을 적용
        - 하드웨어별 최적화: 다양한 CPU 아키텍처에 대한 최적화된 코드 경로를 포함

#### 비교

- TGI
    - 엔터프라이즈 수준의 배포에 탁월
    - 쿠버네티스 지원이 내장
    - 프로메테우스 및 그라파나를 통한 모니터링, 자동 스케일링, 포괄적인 안전 기능과 같이 프로덕션 실행에 필요한 모든 것을 포함
- vLLM
    - 개발자 친화적인 배포 방식
    - 핵심이 파이썬으로 구축되어 요구사항에 맞게 커스터마이징이 가능
    - 클러스터 관리를 위한 Ray와 함께 잘 작동하여, 고성능과 적응성이 필요할 때 좋음
    - llama.cpp
        - 단순성과 이식성을 우선시
        - 서버 구현이 경량이며, 광범위한 하드웨어에서 실행할 수 있음
        - 최소한의 종속성과 간단한 C/C++ 코어 덕분에 파이썬 프레임워크 설치가 어려운 환경에서도 쉽게 배포