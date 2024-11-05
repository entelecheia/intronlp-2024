# Week 3 Session 1: Introduction to Language Models and N-grams

## 1. Introduction to Language Models

Language models are probabilistic models that assign probabilities to sequences of words. They play a crucial role in many Natural Language Processing (NLP) tasks, including:

- Speech recognition
- Machine translation
- Text generation
- Spelling correction
- Sentiment analysis

The main goal of a language model is to learn the joint probability distribution of sequences of words in a language.

### 1.1 Formal Definition

Given a sequence of words W = (w₁, w₂, ..., wₙ), a language model computes the probability P(W):

P(W) = P(w₁, w₂, ..., wₙ)

Using the chain rule of probability, we can decompose this into:

P(W) = P(w₁) _ P(w₂|w₁) _ P(w₃|w₁,w₂) _ ... _ P(wₙ|w₁,w₂,...,wₙ₋₁)

### 1.2 Applications of Language Models

1. **Predictive Text**: Suggesting the next word in a sentence (e.g., smartphone keyboards)
2. **Machine Translation**: Ensuring fluent and grammatical translations
3. **Speech Recognition**: Distinguishing between similar-sounding phrases
4. **Text Generation**: Creating human-like text for chatbots or content generation
5. **Information Retrieval**: Improving search results by understanding query intent

## 2. N-gram Models

N-gram models are a type of probabilistic language model based on the Markov assumption. They predict the probability of a word given the N-1 previous words.

### 2.1 Types of N-grams

- Unigram: P(wᵢ)
- Bigram: P(wᵢ|wᵢ₋₁)
- Trigram: P(wᵢ|wᵢ₋₂,wᵢ₋₁)
- 4-gram: P(wᵢ|wᵢ₋₃,wᵢ₋₂,wᵢ₋₁)

### 2.2 The Markov Assumption

N-gram models make the simplifying assumption that the probability of a word depends only on the N-1 previous words. This is known as the Markov assumption.

For a trigram model:

P(wᵢ|w₁,w₂,...,wᵢ₋₁) ≈ P(wᵢ|wᵢ₋₂,wᵢ₋₁)

### 2.3 Calculating N-gram Probabilities

N-gram probabilities are typically calculated using Maximum Likelihood Estimation (MLE):

P(wₙ|wₙ₋ᵢ,...,wₙ₋₁) = count(wₙ₋ᵢ,...,wₙ) / count(wₙ₋ᵢ,...,wₙ₋₁)

Let's implement a simple function to calculate bigram probabilities:

```python
from collections import defaultdict
import nltk
nltk.download('punkt')

def calculate_bigram_prob(corpus):
    # Tokenize the corpus
    tokens = nltk.word_tokenize(corpus.lower())

    # Count bigrams and unigrams
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        bigram_counts[bigram] += 1
        unigram_counts[tokens[i]] += 1
    unigram_counts[tokens[-1]] += 1

    # Calculate probabilities
    bigram_probs = {}
    for bigram, count in bigram_counts.items():
        w1, w2 = bigram
        bigram_probs[bigram] = count / unigram_counts[w1]

    return bigram_probs

# Example usage
corpus = "the cat sat on the mat . the dog sat on the floor ."
bigram_probs = calculate_bigram_prob(corpus)

print("Some bigram probabilities:")
for bigram, prob in list(bigram_probs.items())[:5]:
    print(f"P({bigram[1]}|{bigram[0]}) = {prob:.2f}")
```

### 2.4 Advantages and Limitations of N-gram Models

Advantages:

- Simple to understand and implement
- Computationally efficient
- Work well for many applications with sufficient data

Limitations:

- Limited context (only consider N-1 previous words)
- Data sparsity (many possible N-grams never occur in training data)
- No generalization to semantically similar contexts

## 3. Handling Unseen N-grams: Smoothing Techniques

One major issue with N-gram models is the zero-probability problem: N-grams that don't appear in the training data are assigned zero probability. Smoothing techniques address this issue.

### 3.1 Laplace (Add-1) Smoothing

Laplace smoothing adds 1 to all N-gram counts:

P(wₙ|wₙ₋ᵢ,...,wₙ₋₁) = (count(wₙ₋ᵢ,...,wₙ) + 1) / (count(wₙ₋ᵢ,...,wₙ₋₁) + V)

Where V is the vocabulary size.

Let's implement Laplace smoothing for bigrams:

```python
def laplace_smoothed_bigram_prob(corpus):
    tokens = nltk.word_tokenize(corpus.lower())
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        bigram_counts[bigram] += 1
        unigram_counts[tokens[i]] += 1
    unigram_counts[tokens[-1]] += 1

    V = len(set(tokens))  # Vocabulary size

    smoothed_probs = {}
    for w1 in unigram_counts:
        for w2 in unigram_counts:
            bigram = (w1, w2)
            smoothed_probs[bigram] = (bigram_counts[bigram] + 1) / (unigram_counts[w1] + V)

    return smoothed_probs

# Example usage
smoothed_probs = laplace_smoothed_bigram_prob(corpus)

print("Some Laplace-smoothed bigram probabilities:")
for bigram, prob in list(smoothed_probs.items())[:5]:
    print(f"P({bigram[1]}|{bigram[0]}) = {prob:.4f}")
```

### 3.2 Other Smoothing Techniques

- Add-k Smoothing: Similar to Laplace but adds k < 1
- Good-Turing Smoothing: Estimates probability for unseen N-grams based on the frequency of N-grams that appear once
- Kneser-Ney Smoothing: Uses the frequency of more general N-grams to estimate probabilities for specific N-grams

## 4. Evaluating Language Models: Perplexity

Perplexity is a common metric for evaluating language models. It measures how well a probability distribution predicts a sample. Lower perplexity indicates better performance.

For a test set W = w₁w₂...wₙ, the perplexity is:

PP(W) = P(w₁w₂...wₙ)^(-1/n)

Let's implement a function to calculate perplexity for a bigram model:

```python
import math

def calculate_perplexity(test_corpus, bigram_probs):
    tokens = nltk.word_tokenize(test_corpus.lower())
    n = len(tokens)
    log_probability = 0

    for i in range(n - 1):
        bigram = (tokens[i], tokens[i+1])
        if bigram in bigram_probs:
            log_probability += math.log2(bigram_probs[bigram])
        else:
            log_probability += math.log2(1e-10)  # Small probability for unseen bigrams

    perplexity = 2 ** (-log_probability / n)
    return perplexity

# Example usage
test_corpus = "the cat sat on the floor ."
perplexity = calculate_perplexity(test_corpus, bigram_probs)
print(f"Perplexity: {perplexity:.2f}")
```

## 5. Visualizing N-gram Models

To better understand N-gram models, let's create a visualization of bigram transitions:

```python
import matplotlib.pyplot as plt
import networkx as nx

def visualize_bigrams(bigram_probs, top_n=5):
    G = nx.DiGraph()

    for (w1, w2), prob in bigram_probs.items():
        G.add_edge(w1, w2, weight=prob)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=3000, font_size=10, font_weight='bold')

    edge_labels = {(w1, w2): f"{prob:.2f}" for (w1, w2), prob in bigram_probs.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Bigram Transition Probabilities")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_bigrams(bigram_probs)
```

This visualization shows the transitions between words in our bigram model, with edge weights representing transition probabilities.

## Conclusion

In this session, we've introduced the concept of language models and delved into N-gram models, particularly focusing on bigrams. We've covered:

1. The definition and applications of language models
2. N-gram models and the Markov assumption
3. Calculating N-gram probabilities
4. Smoothing techniques for handling unseen N-grams
5. Evaluating language models using perplexity
6. Visualizing N-gram models

In the next session, we'll explore more advanced topics in statistical language modeling and begin to look at neural approaches to language modeling.
