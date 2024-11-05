# 10주차 세션 2: 벡터 데이터베이스와 임베딩

## 개요

이번 세션에서는 LLM 기반 Q&A 시스템 구축에 있어서 벡터 데이터베이스와 임베딩이 갖는 중요한 역할에 대해 알아보겠습니다. 텍스트 데이터를 의미론적 의미를 포착하는 수치 벡터로 변환하는 임베딩과, 이를 통해 효율적인 유사도 검색이 가능해지는 과정을 살펴볼 것입니다. 인기 있는 벡터 데이터베이스 솔루션들과 임베딩의 생성 및 활용 방법, 그리고 이러한 구성 요소들이 Q&A 시스템의 성능과 확장성을 어떻게 향상시키는지 학습하게 될 것입니다.

## 소개

텍스트 데이터가 기하급수적으로 증가함에 따라, 관련 정보를 효율적으로 저장하고 검색하는 것이 중요한 과제가 되었습니다. 임베딩은 텍스트를 고차원 벡터 공간에서 표현하여 단어, 문장, 또는 문서 간의 의미론적 관계를 포착할 수 있게 해줍니다. 벡터 데이터베이스는 이러한 고차원 벡터를 저장하고 쿼리하는데 최적화된 특수 시스템입니다. 정보를 빠르게 검색하고 처리할 수 있는 고급 확장형 Q&A 시스템을 구축하고자 하는 AI 엔지니어에게 임베딩과 벡터 데이터베이스의 이해는 필수적입니다.

## 벡터 데이터베이스와 그 중요성

### 목적과 장점

- **정의**: 벡터 데이터베이스는 고차원 벡터 데이터를 효율적으로 처리하도록 설계된 특수 저장 시스템입니다.
- **주요 특징**:
  - **효율적인 유사도 검색**: 고차원 데이터에 최적화된 알고리즘을 사용하여 빠른 유사도 검색을 수행합니다.
  - **확장성**: 수백만 또는 수십억 개의 벡터를 처리할 수 있습니다.
  - **실시간 검색**: 응답성 높은 애플리케이션을 위한 빠른 쿼리를 지원합니다.
- **장점**:
  - **최적화된 저장**: 성능 저하 없이 대량의 벡터 데이터를 효율적으로 저장합니다.
  - **커스터마이즈 가능한 인덱싱**: 다양한 사용 사례에 맞는 인덱싱 방법을 제공합니다.
  - **유연한 통합**: 애플리케이션과의 원활한 통합을 위한 API와 커넥터를 제공합니다.

### 주요 솔루션

#### Pinecone

- **특징**:
  - 완전 관리형, 클라우드 네이티브 벡터 데이터베이스입니다.
  - 실시간 인덱싱과 쿼리를 지원합니다.
  - 높은 가용성과 보안을 제공합니다.
- **장점**:
  - 설정과 확장이 쉽습니다.
  - 인프라 관리가 필요 없습니다.
  - 인기 있는 ML 프레임워크와 통합됩니다.

#### Weaviate

- **특징**:
  - 오픈소스, 확장 가능한 벡터 검색 엔진입니다.
  - GraphQL과 RESTful API를 지원합니다.
  - 벡터 검색과 키워드 검색을 결합한 하이브리드 검색을 제공합니다.
- **장점**:
  - 플러그인으로 커스터마이즈가 가능합니다.
  - 커뮤니티 지원과 활발한 개발이 이루어지고 있습니다.
  - 완전한 제어를 위해 자체 호스팅이 가능합니다.

#### Milvus

- **특징**:
  - 확장성을 위해 구축된 오픈소스 벡터 데이터베이스입니다.
  - 다양한 인덱싱 알고리즘을 지원합니다.
  - 고성능 벡터 유사도 검색을 위해 설계되었습니다.
- **장점**:
  - 대규모 데이터셋을 효율적으로 처리합니다.
  - 유연한 배포 옵션을 제공합니다.
  - 데이터 과학 도구와 통합됩니다.

## 문서 임베딩

### 임베딩 모델과 기법

- **정의**: 임베딩은 단어나 구문 간의 의미론적 의미와 관계를 포착하는 텍스트의 수치 표현입니다.
- **주요 특징**:
  - **단어 임베딩**: 개별 단어를 표현합니다 (예: Word2Vec, GloVe).
  - **문장 임베딩**: 전체 문장이나 구문을 표현합니다 (예: Sentence-BERT).
  - **문서 임베딩**: 단락이나 문서와 같은 더 큰 텍스트 블록을 표현합니다.
- **모델**:
  - **Word2Vec**:
    - 목표 단어에서 문맥 단어를 예측하거나(Skip-gram) 그 반대(CBOW)를 수행합니다.
    - 의미론적 관계를 포착합니다 (예: king - man + woman ≈ queen).
  - **GloVe**:
    - 전역 단어-단어 동시 발생 통계를 사용합니다.
    - 단어 문맥을 기반으로 의미를 포착하는 임베딩을 생성합니다.
  - **Sentence-BERT (SBERT)**:
    - BERT를 확장하여 문장 임베딩을 생성합니다.
    - 의미론적 유사도 태스크를 위해 미세 조정되었습니다.

### 임베딩 생성

- **과정**:
  - **토큰화**: 텍스트를 토큰(단어, 하위 단어)으로 분할합니다.
  - **인코딩**: 임베딩 모델을 사용하여 토큰을 벡터로 변환합니다.
  - **집계**: 토큰 벡터를 단일 벡터로 결합합니다(문장이나 문서의 경우).
- **예시**:

  ```python
  from sentence_transformers import SentenceTransformer

  model = SentenceTransformer('all-MiniLM-L6-v2')
  sentence = "기계학습은 컴퓨터가 데이터로부터 학습할 수 있게 합니다."
  embedding = model.encode(sentence)
  ```

### 차원 축소 전략

- **목적**: 필수 정보를 유지하면서 계산 복잡성과 저장 요구사항을 줄입니다.
- **기법**:
  - **주성분 분석 (PCA)**:
    - 데이터를 저차원 공간으로 투영합니다.
    - 최대한의 분산을 보존합니다.
  - **t-분포 확률적 이웃 임베딩 (t-SNE)**:
    - 고차원 데이터를 2D 또는 3D로 시각화합니다.
    - 데이터의 로컬 구조를 보존합니다.
  - **오토인코더**:
    - 입력 데이터를 재구성하도록 훈련된 신경망입니다.
    - 병목층이 압축된 표현을 제공합니다.

### 유사도 검색 방법

- **거리 메트릭**:
  - **코사인 유사도**:
    - 두 벡터 간의 각도의 코사인을 측정합니다.
    - 값의 범위는 -1(반대)에서 1(동일)입니다.
  - **유클리드 거리**:
    - 공간상의 두 점 사이의 직선 거리를 측정합니다.
    - 벡터의 크기에 민감합니다.
  - **맨해튼 거리**:
    - 각 차원에서의 절대 차이의 합입니다.
- **검색 알고리즘**:
  - **전수 검색**:
    - 쿼리와 모든 벡터 간의 유사도를 계산합니다.
    - 정확하지만 확장성이 떨어집니다.
  - **근사 최근접 이웃 (ANN)**:
    - 정확도와 속도의 균형을 맞춥니다.
    - HNSW(Hierarchical Navigable Small World) 그래프 등이 있습니다.
  - **역색인**:
    - 키워드 검색과 벡터 검색을 결합한 하이브리드 검색에 사용됩니다.

### 요약

임베딩은 텍스트 데이터를 의미론적 의미를 포착하는 벡터로 변환하여 벡터 데이터베이스에서 유사도 검색을 가능하게 합니다. Q&A 시스템의 성능을 위해서는 적절한 임베딩 모델과 유사도 메트릭의 선택이 중요합니다.

## 벡터 데이터베이스 작업

### 문서 인덱싱

- **과정**:
  - **임베딩 생성**: 문서를 임베딩으로 변환합니다.
  - **메타데이터 연결**: 관련 정보(예: 문서 ID, 출처)를 첨부합니다.
  - **업서트**: 벡터 데이터베이스에 항목을 삽입하거나 업데이트합니다.
- **예시**:
  ```python
  # 임베딩과 메타데이터가 준비되었다고 가정
  items = [
      ('doc1', embedding1, {'title': 'AI 소개'}),
      ('doc2', embedding2, {'title': '딥러닝 기초'}),
      # 추가 항목...
  ]
  index.upsert(items=items)
  ```
- **모범 사례**:
  - 효율성을 위해 배치 작업을 수행합니다.
  - 인덱싱 성능을 모니터링합니다.
  - 삽입 후 데이터 무결성을 검증합니다.

### 쿼리 및 검색

- **과정**:
  - **쿼리 임베딩**: 사용자의 질문에 대한 임베딩을 생성합니다.
  - **유사도 검색**: 데이터베이스에서 상위 K개의 유사한 임베딩을 검색합니다.
  - **결과 처리**: 관련 정보를 추출하고 제시합니다.
- **예시**:
  ```python
  query = "강화학습의 개념을 설명해주세요."
  query_embedding = model.encode(query)
  results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
  for match in results['matches']:
      print(f"점수: {match['score']}, 제목: {match['metadata']['title']}")
  ```
- **고려사항**:
  - 관련성과 성능의 균형을 맞추어 적절한 `top_k` 값을 선택합니다.
  - 지원되는 경우 필터를 사용하여 결과를 좁힙니다.

### 업데이트 및 유지보수

- **벡터 업데이트**:
  - **재인덱싱**: 임베딩 모델이 업데이트된 경우 임베딩을 재생성합니다.
  - **삭제**: 쓸모없거나 관련 없는 항목을 제거합니다.
- **성능 최적화**:
  - **인덱스 새로고침**: 효율성 유지를 위해 주기적으로 인덱스를 재구축합니다.
  - **모니터링**: 쿼리 지연 시간과 인덱스 상태를 추적합니다.
- **확장 전략**:
  - **샤딩**: 여러 노드에 데이터를 분산합니다.
  - **복제**: 중복성과 부하 분산을 위해 데이터를 복제합니다.

### 요약

효과적인 벡터 데이터베이스 관리에는 적절한 인덱싱, 쿼리, 유지보수 사례가 포함됩니다. 이를 통해 데이터가 증가해도 Q&A 시스템이 응답성과 정확성을 유지할 수 있습니다.

## 실습 예제 및 연습

### 예제 1: 문서 임베딩 및 검색 파이프라인 구축

**목표**: Pinecone을 사용하여 문서 집합에 대한 임베딩을 생성하고 유사도 검색을 수행합니다.

**단계**:

1. **필요한 라이브러리 설치**:

   ```bash
   pip install sentence-transformers pinecone-client
   ```

2. **Pinecone 초기화**:

   ```python
   import pinecone

   pinecone.init(api_key='YOUR_API_KEY', environment='us-west1-gcp')
   index_name = 'document-embeddings'
   if index_name not in pinecone.list_indexes():
       pinecone.create_index(index_name, dimension=384)
   index = pinecone.Index(index_name)
   ```

3. **임베딩 모델 로드**:

   ```python
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('all-MiniLM-L6-v2')
   ```

4. **문서 준비 및 임베딩**:

   ```python
   documents = [
       {'id': 'doc1', 'text': '기계학습은 인공지능의 한 분야입니다...'},
       {'id': 'doc2', 'text': '신경망은 뇌에서 영감을 받은 컴퓨팅 시스템입니다...'},
       # 추가 문서
   ]
   embeddings = []
   for doc in documents:
       embedding = model.encode(doc['text']).tolist()
       embeddings.append((doc['id'], embedding, {'text': doc['text']}))
   ```

5. **Pinecone에 임베딩 업서트**:

   ```python
   index.upsert(items=embeddings)
   ```

6. **유사도 검색 수행**:
   ```python
   query = "신경망은 어떻게 작동하나요?"
   query_embedding = model.encode(query).tolist()
   results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
   for match in results['matches']:
       print(f"점수: {match['score']}")
       print(f"텍스트: {match['metadata']['text']}\n")
   ```

### 예제 2: Weaviate를 사용한 하이브리드 검색

**목표**: Weaviate를 사용하여 키워드 검색과 벡터 검색을 결합합니다.

**단계**:

1. **Weaviate 클라이언트 설치**:

   ```bash
   pip install weaviate-client
   ```

2. **Weaviate 클라이언트 초기화**:

   ```python
   import weaviate

   client = weaviate.Client("http://localhost:8080")
   ```

3. **스키마 정의 및 데이터 가져오기**:

   ```python
   schema = {
       'classes': [
           {
               'class': 'Document',
               'properties': [
                   {'name': 'text', 'dataType': ['text']},
                   {'name': 'title', 'dataType': ['string']},
               ]
           }
       ]
   }
   client.schema.create(schema)
   # 이전 예제와 유사하게 문서 가져오기
   ```

4. **하이브리드 검색 수행**:
   ```python
   query = "딥러닝 설명"
   response = client.query.get('Document', ['title', 'text']) \
       .with_hybrid(query, alpha=0.5) \
       .do()
   print(response)
   ```

### 실습 과제

**과제**: Milvus와 Sentence-BERT 임베딩을 사용하여 Q&A 검색 시스템을 구현하세요.

**지시사항**:

1. Milvus와 Python SDK를 설치하세요.
2. 문서 데이터셋을 준비하세요.
3. 임베딩을 생성하고 Milvus에 삽입하세요.
4. 사용자의 질문을 받아 가장 관련성 높은 문서를 반환하는 함수를 만드세요.

**기대 결과**:

- Milvus를 벡터 저장 및 검색에 활용하는 기능적인 스크립트

## 결론

### 요약

이번 세션에서는 벡터 데이터베이스와 임베딩이 LLM 기반 Q&A 시스템에서 얼마나 중요한지 살펴보았습니다. 다양한 임베딩 모델, 임베딩의 생성 및 조작 방법, 그리고 Pinecone, Weaviate, Milvus와 같은 벡터 데이터베이스가 이러한 임베딩을 효율적으로 저장하고 검색하는 방법에 대해 학습했습니다.

### 향후 방향

임베딩과 벡터 데이터베이스에 대한 견고한 이해를 바탕으로, 이제 이러한 구성 요소들을 완전한 Q&A 시스템에 통합할 수 있게 되었습니다. 다음 세션에서는 문서 처리, 임베딩, 벡터 데이터베이스, LLM을 결합하여 기능적인 Q&A 애플리케이션을 구축하는 데 중점을 둘 것입니다.

## 참고문헌 및 추가 자료

- **도서**:
  - _Deep Learning_ (Ian Goodfellow, Yoshua Bengio, Aaron Courville 저)
- **온라인 자료**:
  - [Pinecone 문서](https://docs.pinecone.io/)
  - [Weaviate 문서](https://weaviate.io/developers/weaviate)
  - [Milvus 문서](https://milvus.io/docs/)
  - [SentenceTransformers 문서](https://www.sbert.net/)
- **튜토리얼**:
  - [Pinecone과 Sentence Transformers를 사용한 시맨틱 검색 엔진 구축](https://towardsdatascience.com/building-a-semantic-search-engine-using-pinecone-and-sentence-transformers-3e2665ad72d8)
  - [Weaviate를 사용한 시맨틱 검색 구현](https://weaviate.io/developers/weaviate/quickstart)
  - [유사도 검색을 위한 Milvus 활용](https://milvus.io/docs/tutorials/quick_start)
