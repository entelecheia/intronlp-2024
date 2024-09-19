# Week 4 Session 1 - Introduction to Word Embeddings and Word2Vec

## 1. Introduction to Word Embeddings

Word embeddings are dense vector representations of words in a continuous vector space. They are a fundamental concept in modern Natural Language Processing (NLP), offering several advantages over traditional word representation methods.

### 1.1 Limitations of Traditional Word Representations

Traditional methods like one-hot encoding represent words as sparse vectors:

```python
vocab = ["cat", "dog", "mouse"]
one_hot_cat = [1, 0, 0]
one_hot_dog = [0, 1, 0]
one_hot_mouse = [0, 0, 1]
```

Problems with this approach:

- Dimensionality increases with vocabulary size
- No semantic information captured
- All words are equidistant from each other

### 1.2 The Idea Behind Word Embeddings

Word embeddings are based on the distributional hypothesis: words that occur in similar contexts tend to have similar meanings.

Key properties:

- Dense vector representations (e.g., 100-300 dimensions)
- Capture semantic and syntactic information
- Similar words are close in the vector space

Let's visualize this concept:

```python
import matplotlib.pyplot as plt

def plot_vectors(vectors, labels):
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(vectors):
        plt.scatter(x, y)
        plt.annotate(labels[i], (x, y))
    plt.title("Word Vectors in 2D Space")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.show()

# Example 2D word vectors
vectors = [(0.2, 0.3), (0.3, 0.2), (0.5, 0.7), (-0.3, -0.2), (-0.2, -0.3)]
labels = ["cat", "dog", "animal", "car", "truck"]

plot_vectors(vectors, labels)
```

This visualization shows how semantically similar words ("cat", "dog", "animal") are closer to each other in the vector space compared to unrelated words ("car", "truck").

## 2. Word2Vec

Word2Vec, introduced by Mikolov et al. in 2013, is one of the most popular word embedding models. It comes in two flavors: Continuous Bag of Words (CBOW) and Skip-gram.

### 2.1 CBOW Architecture

CBOW predicts a target word given its context words.

```{mermaid}
graph TD
    A[Input Context Words] --> B[Input Layer]
    B --> C[Hidden Layer]
    C --> D[Output Layer]
    D --> E[Target Word]
```

### 2.2 Skip-gram Architecture

Skip-gram predicts context words given a target word.

```{mermaid}
graph TD
    A[Input Target Word] --> B[Input Layer]
    B --> C[Hidden Layer]
    C --> D[Output Layer]
    D --> E[Context Words]
```

### 2.3 Training Process

Word2Vec uses a neural network to learn word embeddings:

1. Initialize word vectors randomly
2. Slide a window over the text corpus
3. For each window:
   - CBOW: Use context words to predict the center word
   - Skip-gram: Use center word to predict context words
4. Update word vectors based on prediction errors
5. Repeat until convergence

### 2.4 Implementing Word2Vec with Gensim

Let's implement a simple Word2Vec model using Gensim:

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Sample corpus
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The cat and the dog are natural enemies",
    "The dog chases the cat up the tree"
]

# Tokenize the corpus
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Find similar words
similar_words = model.wv.most_similar("dog", topn=3)
print("Words most similar to 'dog':", similar_words)

# Perform word analogy
result = model.wv.most_similar(positive=['dog', 'tree'], negative=['cat'], topn=1)
print("dog - cat + tree =", result[0][0])

# Get word vector
dog_vector = model.wv['dog']
print("Vector for 'dog':", dog_vector[:5])  # Showing first 5 dimensions
```

### 2.5 Visualizing Word Embeddings

To visualize our word embeddings, we can use dimensionality reduction techniques like t-SNE:

```python
from sklearn.manifold import TSNE
import numpy as np

def plot_embeddings(model, words):
    word_vectors = [model.wv[word] for word in words]
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        x, y = embeddings_2d[i, :]
        plt.scatter(x, y)
        plt.annotate(word, (x, y))
    plt.title("Word Embeddings Visualization")
    plt.show()

# Visualize embeddings for selected words
words_to_plot = ["dog", "cat", "fox", "tree", "quick", "lazy", "jumps", "chases"]
plot_embeddings(model, words_to_plot)
```

## 3. Advantages of Word2Vec

1. Captures semantic relationships
2. Efficient to train on large corpora
3. Can handle out-of-vocabulary words (with subword information)
4. Useful for various downstream NLP tasks

## 4. Limitations and Considerations

1. Requires large amounts of training data
2. Cannot handle polysemy (words with multiple meanings)
3. Static embeddings (don't account for context)
4. Bias in training data can be reflected in embeddings

## Conclusion

Word embeddings, particularly Word2Vec, have revolutionized many NLP tasks by providing rich, dense representations of words. In the next session, we'll explore other word embedding techniques like GloVe and FastText, and discuss more advanced applications.

## Exercise

1. Train a Word2Vec model on a larger corpus (e.g., a collection of news articles).
2. Experiment with different hyperparameters (vector_size, window, min_count) and observe their effects on the resulting embeddings.
3. Implement a simple word analogy task using your trained model.
4. Visualize the embeddings for a set of words related to a specific domain (e.g., sports, technology) and analyze the relationships captured by the model.

```python
# Your code here
```

This exercise will help you gain practical experience with Word2Vec and deepen your understanding of word embeddings.
