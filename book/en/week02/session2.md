# Week 2 Session 2: Advanced Text Preprocessing and Representation

## Introduction

In this lecture, we'll explore advanced text preprocessing techniques and delve into methods for representing text data in a format suitable for machine learning algorithms. We'll build upon the foundational concepts covered in Session 1 and introduce more sophisticated approaches to prepare text data for analysis.

Let's start by importing the necessary libraries:

```python
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import spacy

# Download necessary NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
```

## 1. Advanced Text Cleaning Techniques

Building upon our basic text cleaning from Session 1, let's explore some advanced techniques:

### Handling Emoji and Emoticons

Emojis and emoticons can carry sentiment information. We can either remove them or replace them with textual descriptions.

```python
import emoji

def handle_emojis(text):
    # Replace emojis with their textual description
    return emoji.demojize(text)

# Example usage
sample_text = "I love this movie! 😍👍"
processed_text = handle_emojis(sample_text)
print("Original text:", sample_text)
print("Processed text:", processed_text)
```

### Handling Non-ASCII Characters

For text data that may contain non-ASCII characters, we can either remove them or normalize them:

```python
import unicodedata

def normalize_unicode(text):
    # Normalize Unicode characters
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

# Example usage
sample_text = "Café au lait is a délicious drink"
normalized_text = normalize_unicode(sample_text)
print("Original text:", sample_text)
print("Normalized text:", normalized_text)
```

## 2. Handling Contractions and Special Cases

Contractions (e.g., "don't", "I'm") can be problematic for some NLP tasks. Let's create a function to expand contractions:

```python
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "don't": "do not",
    "I'm": "I am",
    "isn't": "is not",
    "it's": "it is",
    "won't": "will not",
    # Add more contractions as needed
}

def expand_contractions(text):
    for contraction, expansion in CONTRACTION_MAP.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
    return text

# Example usage
sample_text = "I can't believe it's not butter!"
expanded_text = expand_contractions(sample_text)
print("Original text:", sample_text)
print("Expanded text:", expanded_text)
```

## 3. Named Entity Recognition (NER)

Named Entity Recognition is the process of identifying and classifying named entities (e.g., person names, organizations, locations) in text. Let's use spaCy for NER:

```python
def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Example usage
sample_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
entities = perform_ner(sample_text)
print("Named Entities:", entities)

# Visualize NER results
from spacy import displacy
displacy.render(nlp(sample_text), style="ent", jupyter=True)
```

## 4. Part-of-Speech (POS) Tagging

POS tagging involves labeling words with their grammatical categories (e.g., noun, verb, adjective). We'll use NLTK for this task:

```python
def pos_tag_text(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    return pos_tags

# Example usage
sample_text = "The quick brown fox jumps over the lazy dog."
pos_tags = pos_tag_text(sample_text)
print("POS Tags:", pos_tags)

# Visualize POS tags
def plot_pos_tags(pos_tags):
    words, tags = zip(*pos_tags)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(words), y=[1]*len(words), hue=list(tags), dodge=False)
    plt.title("Part-of-Speech Tags")
    plt.xlabel("Words")
    plt.ylabel("")
    plt.legend(title="POS Tags", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

plot_pos_tags(pos_tags)
```

## 5. Text Representation Methods

Now that we've covered advanced preprocessing techniques, let's explore methods for representing text data in a format suitable for machine learning algorithms.

### Bag of Words (BoW)

The Bag of Words model represents text as a vector of word frequencies, disregarding grammar and word order.

```python
def create_bow(corpus):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(corpus)
    return bow_matrix, vectorizer.get_feature_names_out()

# Example usage
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "The lazy dog sleeps all day.",
    "The quick brown fox is quick."
]

bow_matrix, feature_names = create_bow(corpus)
print("BoW Feature Names:", feature_names)
print("BoW Matrix Shape:", bow_matrix.shape)
print("BoW Matrix:")
print(bow_matrix.toarray())

# Visualize BoW
plt.figure(figsize=(12, 6))
sns.heatmap(bow_matrix.toarray(), annot=True, fmt='d', cmap='YlGnBu', xticklabels=feature_names)
plt.title("Bag of Words Representation")
plt.ylabel("Documents")
plt.xlabel("Words")
plt.tight_layout()
plt.show()
```

### TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF represents the importance of words in a document relative to a collection of documents.

```python
def create_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer.get_feature_names_out()

# Example usage
tfidf_matrix, feature_names = create_tfidf(corpus)
print("TF-IDF Feature Names:", feature_names)
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())

# Visualize TF-IDF
plt.figure(figsize=(12, 6))
sns.heatmap(tfidf_matrix.toarray(), annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=feature_names)
plt.title("TF-IDF Representation")
plt.ylabel("Documents")
plt.xlabel("Words")
plt.tight_layout()
plt.show()
```

## 6. Word Embeddings

Word embeddings are dense vector representations of words that capture semantic relationships. Let's explore Word2Vec, a popular word embedding technique.

```python
def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
    return model

# Example usage
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
w2v_model = train_word2vec(tokenized_corpus)

# Visualize word embeddings using t-SNE
from sklearn.manifold import TSNE

def plot_word_embeddings(model, words):
    word_vectors = [model.wv[word] for word in words]
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(12, 8))
    for i, word in enumerate(words):
        x, y = embedded[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.title("Word Embeddings Visualization")
    plt.xlabel("t-SNE feature 0")
    plt.ylabel("t-SNE feature 1")
    plt.tight_layout()
    plt.show()

plot_word_embeddings(w2v_model, ['quick', 'brown', 'fox', 'lazy', 'dog', 'jumps', 'sleeps'])
```

## 7. Conclusion and Best Practices

In this lecture, we've covered advanced text preprocessing techniques and various methods for representing text data. Here are some best practices to keep in mind:

1. Choose preprocessing techniques based on your specific task and dataset.
2. Consider the trade-offs between different text representation methods:
   - BoW and TF-IDF are simple but may lose semantic information.
   - Word embeddings capture semantic relationships but require more data to train effectively.
3. When using word embeddings, consider using pre-trained models for better performance, especially on small datasets.
4. Be mindful of the computational resources required for different representation methods, especially when dealing with large datasets.
5. Regularly evaluate the impact of your preprocessing and representation choices on your model's performance.

Remember that text preprocessing and representation are crucial steps in the NLP pipeline, and the choices you make here can significantly impact your model's performance.

## Exercise

1. Choose a dataset of your choice (e.g., a collection of news articles, tweets, or product reviews).
2. Apply the advanced preprocessing techniques we've learned (e.g., handling emojis, expanding contractions, NER).
3. Create both BoW and TF-IDF representations of your preprocessed text.
4. Train a Word2Vec model on your dataset and visualize the embeddings for some interesting words.
5. Reflect on the differences between these representation methods and how they might affect downstream NLP tasks.

```python
# Your code here
```

This exercise will give you hands-on experience with advanced preprocessing techniques and different text representation methods, helping you understand their practical implications in real-world NLP tasks.
