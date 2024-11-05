# 3주차: 언어 모델의 기초

## 개요

이번 주에는 많은 자연어 처리(NLP) 응용 프로그램에서 필수적인 구성 요소인 언어 모델의 기본 개념을 살펴봅니다. N-gram 모델과 통계적 언어 모델에 초점을 맞추어, 기계가 어떻게 인간과 유사한 텍스트를 예측하고 생성할 수 있는지에 대한 확실한 이해를 제공할 것입니다.

## 학습 목표

이번 주가 끝나면 다음을 할 수 있게 됩니다:

1. NLP에서 언어 모델의 개념과 중요성 이해하기
2. N-gram 모델의 이론과 응용을 설명하기
3. Python을 사용하여 간단한 N-gram 모델 구현하기
4. 통계적 언어 모델의 기초 이해하기
5. 퍼플렉시티를 사용하여 언어 모델의 성능 평가하기

## 주요 주제

### 1. 언어 모델 소개

- 언어 모델의 정의와 목적
- NLP 작업에서의 언어 모델 응용
- 언어 모델링 기술의 역사적 맥락과 진화

### 2. N-gram 모델

- N-gram의 개념 (unigram, bigram, trigram 등)
- N-gram 모델에서의 확률 계산
- N-gram 모델의 장단점
- 미등장 N-gram 처리를 위한 스무딩 기법

### 3. 통계적 언어 모델

- 언어 모델링에 대한 확률론적 접근
- 최대 우도 추정 (MLE)
- 언어 모델에서의 조건부 확률
- 통계적 언어 모델링의 과제 (데이터 희소성, 맥락 제한)

### 4. N-gram 모델 구현

- 텍스트 코퍼스로부터 N-gram 모델 구축하기
- N-gram 모델을 사용한 텍스트 생성
- 어휘 외 단어 처리하기

### 5. 언어 모델 평가

- 평가 지표로서의 퍼플렉시티 소개
- N-gram 모델의 퍼플렉시티 계산
- 퍼플렉시티 점수 해석하기

## 실습 구성요소

이번 주 실습 세션에서는 다음을 수행합니다:

- Python을 사용하여 간단한 N-gram 모델을 처음부터 구현하기
- NLTK를 사용하여 N-gram 모델 생성 및 실험하기
- 구현한 N-gram 모델을 사용하여 텍스트 생성하기
- 퍼플렉시티를 사용하여 모델의 성능 평가하기

## 과제

텍스트 코퍼스가 주어지면 다양한 차수의 N-gram 모델(unigram, bigram, trigram)을 구축해야 합니다. 모델을 구현하고, 각 모델을 사용하여 텍스트를 생성하며, 퍼플렉시티를 사용하여 성능을 비교해야 합니다. 또한 관찰 결과를 바탕으로 각 모델의 장단점을 논의하는 간단한 보고서를 작성해야 합니다.

## 앞으로의 전망

이번 주에 학습하는 언어 모델의 기본 개념은 NLP의 더 고급 주제로 나아가는 징검다리가 될 것입니다. 앞으로 몇 주 동안 신경망 기반 접근법과 최신 트랜스포머 모델을 포함한 더 정교한 언어 모델링 기술을 탐구할 것입니다.

## 추가 자료

- [Speech and Language Processing (3rd ed. draft) by Dan Jurafsky and James H. Martin - Chapter 3: N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/)
- [Natural Language Processing with Python – Analyzing Text with the Natural Language Toolkit by Steven Bird, Ewan Klein, and Edward Loper](http://www.nltk.org/book/)
- [A Gentle Introduction to Statistical Language Modeling and Neural Language Models](https://machinelearningmastery.com/statistical-language-modeling-and-neural-language-models/)

```{tableofcontents}

```
