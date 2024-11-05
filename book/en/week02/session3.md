# Week 2 Session 3: Korean Text Preprocessing and Tokenization

## Introduction

In this lecture, we'll explore the unique challenges and techniques involved in preprocessing and tokenizing Korean text. Korean, as an agglutinative language, presents distinct challenges compared to languages like English, and understanding these differences is crucial for effective NLP in Korean.

Let's start by importing the necessary libraries:

```python
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from konlpy.tag import Okt, Kkma, Hannanum
from gensim.models import Word2Vec
import networkx as nx

# Initialize Korean morphological analyzers
okt = Okt()
kkma = Kkma()
hannanum = Hannanum()
```

## 1. Characteristics of Korean Language

Korean is an agglutinative language, which means that words are formed by combining morphemes. This characteristic makes Korean text preprocessing more complex compared to isolating languages like English.

Let's visualize the difference between agglutinative and isolating languages:

```python
def plot_language_comparison():
    G = nx.DiGraph()
    G.add_edge("나", "는")
    G.add_edge("나", "를")
    G.add_edge("나", "에게")
    G.add_edge("I", "Subject")
    G.add_edge("I", "Object")
    G.add_edge("I", "To me")

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=3000, font_size=10, font_weight='bold',
            arrows=True, edge_color='gray')

    plt.title("Comparison of Agglutinative (Korean) vs Isolating (English) Languages")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_language_comparison()
```

In this visualization, we can see how Korean attaches particles to the root word "나" (I) to change its grammatical function, while in English, the word "I" remains unchanged and its function is determined by word order or additional words.

## 2. Challenges in Korean Text Preprocessing

Korean text preprocessing faces several unique challenges:

1. Lack of clear word boundaries
2. Complex morphological structure
3. Frequent use of particles and affixes
4. High degree of homonymy
5. Compound nouns

Let's examine some of these challenges with examples:

```python
def demonstrate_korean_challenges():
    examples = [
        "나는학교에갔다",  # No spaces
        "먹었습니다",  # Complex morphology
        "사과를 먹었다",  # Use of particles
        "감기에 걸렸다",  # Homonymy (감기 can mean "cold" or "winding")
        "한국어자연어처리",  # Compound noun
    ]

    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example}")
        print("Morphemes:", okt.morphs(example))
        print("Nouns:", okt.nouns(example))
        print("POS:", okt.pos(example))
        print()

demonstrate_korean_challenges()
```

## 3. Korean Sentence Tokenization

Sentence tokenization in Korean can be tricky due to the use of punctuation marks in various contexts. Let's implement a simple Korean sentence tokenizer:

```python
def korean_sent_tokenize(text):
    # Simple rule-based sentence tokenizer
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences

# Example usage
sample_text = "안녕하세요. 오늘은 날씨가 좋습니다! 산책 가실래요?"
sentences = korean_sent_tokenize(sample_text)
print("Tokenized sentences:", sentences)
```

## 4. Korean Morphological Analysis

Morphological analysis is crucial in Korean NLP. It involves breaking down words into their smallest meaningful units (morphemes). Let's compare different Korean morphological analyzers:

```python
def compare_morphological_analyzers(text):
    analyzers = [okt, kkma, hannanum]
    analyzer_names = ['Okt', 'Kkma', 'Hannanum']

    fig, axes = plt.subplots(len(analyzers), 1, figsize=(12, 4*len(analyzers)))
    fig.suptitle("Comparison of Korean Morphological Analyzers")

    for ax, analyzer, name in zip(axes, analyzers, analyzer_names):
        morphs = analyzer.morphs(text)
        ax.bar(range(len(morphs)), [1]*len(morphs), align='center')
        ax.set_xticks(range(len(morphs)))
        ax.set_xticklabels(morphs, rotation=45, ha='right')
        ax.set_title(f"{name} Analyzer")
        ax.set_ylabel("Morpheme")

    plt.tight_layout()
    plt.show()

# Example usage
sample_text = "나는 맛있는 한국 음식을 좋아합니다."
compare_morphological_analyzers(sample_text)
```

## 5. Part-of-Speech Tagging in Korean

POS tagging in Korean is more complex due to the language's agglutinative nature. Let's examine POS tagging using different analyzers:

```python
def compare_pos_tagging(text):
    analyzers = [okt, kkma, hannanum]
    analyzer_names = ['Okt', 'Kkma', 'Hannanum']

    for name, analyzer in zip(analyzer_names, analyzers):
        print(f"{name} POS tagging:")
        print(analyzer.pos(text))
        print()

# Example usage
sample_text = "나는 학교에서 한국어를 공부합니다."
compare_pos_tagging(sample_text)
```

Let's visualize the POS tags:

```python
def visualize_pos_tags(text, analyzer):
    pos_tags = analyzer.pos(text)
    words, tags = zip(*pos_tags)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(words), y=[1]*len(words), hue=list(tags), dodge=False)
    plt.title(f"POS Tags using {analyzer.__class__.__name__}")
    plt.xlabel("Words")
    plt.ylabel("")
    plt.legend(title="POS Tags", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

visualize_pos_tags(sample_text, okt)
```

## 6. Korean Word Embeddings

Word embeddings for Korean need to account for the language's morphological complexity. Let's train a simple Word2Vec model on Korean text:

```python
def train_korean_word2vec(sentences, vector_size=100, window=5, min_count=1):
    tokenized_sentences = [okt.morphs(sent) for sent in sentences]
    model = Word2Vec(tokenized_sentences, vector_size=vector_size, window=window, min_count=min_count)
    return model

# Example usage
korean_sentences = [
    "나는 학교에 갑니다.",
    "그는 공부를 열심히 합니다.",
    "우리는 한국어를 배웁니다.",
    "그녀는 책을 좋아합니다."
]

w2v_model = train_korean_word2vec(korean_sentences)

# Visualize word embeddings
def plot_korean_word_embeddings(model, words):
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(12, 8))
    for i, word in enumerate(words):
        if word in model.wv:
            x, y = embedded[i]
            plt.scatter(x, y)
            plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.title("Korean Word Embeddings Visualization")
    plt.xlabel("t-SNE feature 0")
    plt.ylabel("t-SNE feature 1")
    plt.tight_layout()
    plt.show()

plot_korean_word_embeddings(w2v_model, okt.morphs("나 학교 공부 한국어 책 좋아하다"))
```

## 7. Conclusion and Best Practices

In this lecture, we've explored the unique challenges and techniques involved in preprocessing and tokenizing Korean text. Here are some best practices to keep in mind:

1. Always use specialized Korean NLP tools (e.g., KoNLPy) for tokenization and morphological analysis.
2. Be aware of the differences between various Korean morphological analyzers and choose the one that best fits your task.
3. Consider the impact of particles and affixes when processing Korean text.
4. When using word embeddings, train on properly tokenized Korean text to capture the language's morphological nuances.
5. Be mindful of homonyms and context when interpreting Korean text.

Remember that effective Korean NLP requires a deep understanding of the language's structure and characteristics. Continual practice and experimentation with real Korean text data will help you develop intuition for handling the unique challenges of Korean language processing.

## Exercise

1. Choose a Korean text dataset (e.g., news articles, social media posts, or product reviews).
2. Implement a complete preprocessing pipeline for Korean text, including:
   - Sentence tokenization
   - Morphological analysis
   - POS tagging
   - Stop word removal (you'll need to create a Korean stop word list)
3. Train a Word2Vec model on your preprocessed Korean text.
4. Analyze the resulting word embeddings and discuss any interesting semantic relationships you observe.

```python
# Your code here
```

This exercise will give you hands-on experience with the entire Korean text preprocessing pipeline and help you understand the nuances of working with Korean language data in NLP tasks.
