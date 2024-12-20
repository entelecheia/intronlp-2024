# 9차 프로젝트 연구일지

## 기본 정보

- **팀명**: ALL
- **프로젝트명**: AI Law helper (AI 법률 도우미)
- **주차**: 9차

## 팀 구성원 활동 요약

| 이름   | 역할                         | 주요 활동                                                                           | 다음 주 계획                                                     |
| ------ | ---------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 홍성관 | 팀장, NLP모델 및 시스템 설계 | • 프로젝트 요구사항 분석 <br> • 변호사 시뮬레이션 초안 작성 <br> • 팀원 교육 | • 백엔드 및 프론트엔드 코드 수정    |
| 박진우 | 백엔드 개발자               | • 백엔드 구조 설계 <br>                                                          | • UI 구체화                         |
| 김현서 | UI 디자이너                 | • 사용자 인터페이스 초안 작성 <br>                                               | • UI 구체화                                   |
| 김형규 | 데이터 엔지니어             | • 법률 및 판례 수집 <br> • 법률 용어 사전 구성 <br> • 데이터셋 초기 세팅           |  • 리눅스 운영 체제 학습 및 데이터셋 확보 |
| 부윤철 | 데이터 엔지니어              | • 법률 및 판례 수집 <br> • 법률 용어 사전 구성 <br> • 데이터셋 초기 세팅          | • 리눅스 운영 체제 학습 및 데이터셋 확보                               |

## 주간 목표 달성도

| 목표                                         | 상태   | 비고                                                                                           |
| -------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------- |
| 주제 선정 및 제안서 작성(AI 법정 시뮬레이터) | 완료   | 제안서 최종 승인                                                                               |
| 시뮬레이션 주제 선정                         | 완료   | 해킹 및 개인정보 유출, 불법적인 데이터 수집, 사이버 명예훼손                                   |
| 데이터셋 병합(3개 주제)                      | 완료   | • 현재 2개 주제 병합 <br> • json파일 형식으로 수정                                             |
| RAG 및 Langchain 관련 학습                   | 완료 | RAG 및 Langchain 관련 학습                                                                   |
| api key 연결                                 | 완료 | • api key 발급 및 연결                                                                         |
| 법률 데이터셋 조사                           | 진행중 | • 법률과 판례 조사 <br> • 데이터 조사가 완료된 인원들의 데이터 병합 (확대)               |
| UI 구체화                                  |진행중| • 민형법 상호작용 버튼 활성화                                                           |
| 데이터셋 전처리                                  |진행중| • 데이터 병합 및 전처리                                                           |
| 백엔드 코드 수정                                  |진행중| • 백엔드 코드 내 변경사항 수정                                                           |

## 주요 성과 및 결과물

1. RAG와 Langchain을 사용한 프로젝트를 통해 학습 (Open AI api를 사용하여 RAG를 통한 논문 내용 Q&A 시스템 구축)
2. 세 명의 데이터 병합본
3. git, github 사용법
4. 주제에 대한 문제점 파악으로 인해 주제 변경
5. 주제 변경에 의해 역할 수정

## 기술적 도전 및 해결 방안

1. **도전 1**: ubuntu 사용 미숙
   - 해결 방안: 프로젝트를 진행함으로써 학습
2. **도전 2**: git, github 사용 미숙
   - 해결 방안: 프로젝트 레포지터리 생성 및 git을 사용하여 본인 결과물 업로드

## 학습 내용

1.  RAG 시스템을 사용한 코드 분석 및 학습

    - 주요 내용: RAG 시스템이 어떻게 구동되고 사용되는지 학습
    - 적용 방안: 논문 내용 Q&A 시스템을 통해 적용 방안 실습

2.  Open AI API 사용법 숙지
    - 주요 내용: Open AI API를 사용하여 사용자의 질문에 대한 키워드 추출 및 검색 결과 도출 방법 학습
    - 적용 방안: 추출된 키워드를 기반으로 외부 검색 엔진이나 내부 데이터베이스와 연동하여 적절한 검색 결과를 도출

## 다음 주 계획

1. 데이터 확대
2. 홍성관: 백엔드 및 프론트엔드 코드 수정
3. 박진우: 대화 형식 UI 구체화
4. 김현서: UI 민 • 형법 카테고리 기능 추가
5. 김형규:  리눅스 운영체제 학습 및 모델 학습에 필요한 데이터 준비
6. 부윤철:  리눅스 운영체제 학습 및 모델 학습에 필요한 데이터 준비

## 기타 특이사항

- 데이터 부족으로 인해 데이터 수집 과정 연장
- github 및 ubuntu 사용법 미숙으로 인해 코드 수정 과정 연장
- 프로젝트의 관련된 개념 학습 

## 팀 미팅 요약

- **일시**: 2024년 10월 23일
- **참석자**: 홍성관, 박진우, 김현서, 부윤철, 김형규
- **주요 논의 사항**:
  1. 주제 방향성에 관해 논의
- **결정 사항**:
  1. 데이터 양 확대
  2. 대화형식의 UI 구현
  3. UI 민 • 형법 카테고리 기능 추가
  4. 리눅스 운영체제 학습 및 모델 학습에 필요한 데이터 준비

---

작성일: 2024-10-23
작성자: 홍성관, 박진우, 김현서, 부윤철, 김형규
