# Week 3 Session 2: Advanced Statistical Language Models

## 1. Advanced N-gram Techniques

### 1.1 Interpolation and Backoff

Interpolation and backoff are techniques used to combine different order N-gram models to improve performance.

#### Linear Interpolation

Linear interpolation combines probabilities from different order N-grams:

P(wᵢ|wᵢ₋₂wᵢ₋₁) = λ₃P(wᵢ|wᵢ₋₂wᵢ₋₁) + λ₂P(wᵢ|wᵢ₋₁) + λ₁P(wᵢ)

Where λ₁ + λ₂ + λ₃ = 1

```python
def interpolate(trigram_prob, bigram_prob, unigram_prob, lambda1, lambda2, lambda3):
    return lambda3 * trigram_prob + lambda2 * bigram_prob + lambda1 * unigram_prob

# Example usage
trigram_prob = 0.002
bigram_prob = 0.01
unigram_prob = 0.1
lambda1, lambda2, lambda3 = 0.1, 0.3, 0.6

interpolated_prob = interpolate(trigram_prob, bigram_prob, unigram_prob, lambda1, lambda2, lambda3)
print(f"Interpolated probability: {interpolated_prob:.4f}")
```

#### Katz Backoff

Katz backoff uses higher-order N-grams when available, but "backs off" to lower-order N-grams for unseen sequences:

```python
def katz_backoff(trigram, bigram, unigram, trigram_counts, bigram_counts, unigram_counts, k=0.5):
    if trigram in trigram_counts and trigram_counts[trigram] > k:
        return trigram_counts[trigram] / bigram_counts[trigram[:2]]
    elif bigram in bigram_counts:
        alpha = (1 - k * len([t for t in trigram_counts if t[:2] == bigram])) / (1 - k * len([b for b in bigram_counts if b[0] == bigram[0]]))
        return alpha * (bigram_counts[bigram] / unigram_counts[bigram[0]])
    else:
        return unigram_counts[unigram] / sum(unigram_counts.values())

# Example usage (simplified)
trigram_counts = {('the', 'cat', 'sat'): 2, ('cat', 'sat', 'on'): 3}
bigram_counts = {('the', 'cat'): 5, ('cat', 'sat'): 4, ('sat', 'on'): 3}
unigram_counts = {'the': 10, 'cat': 8, 'sat': 6, 'on': 5}

prob = katz_backoff(('the', 'cat', 'sat'), ('cat', 'sat'), 'sat', trigram_counts, bigram_counts, unigram_counts)
print(f"Katz backoff probability: {prob:.4f}")
```

### 1.2 Skip-gram Models

Skip-gram models allow for gaps in the N-gram sequence, capturing longer-range dependencies:

```python
from collections import defaultdict
import nltk

def create_skipgram_model(text, n=3, k=1):
    tokens = nltk.word_tokenize(text.lower())
    skipgram_counts = defaultdict(int)
    context_counts = defaultdict(int)

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        skipgram_counts[ngram] += 1

        for j in range(1, k+1):
            if i+n+j <= len(tokens):
                skipgram = tuple(tokens[i:i+n-1] + [tokens[i+n+j-1]])
                skipgram_counts[skipgram] += 1

        context = tuple(tokens[i:i+n-1])
        context_counts[context] += 1

    skipgram_probs = {gram: count / context_counts[gram[:-1]]
                      for gram, count in skipgram_counts.items()}

    return skipgram_probs

# Example usage
text = "the quick brown fox jumps over the lazy dog"
skipgram_probs = create_skipgram_model(text, n=3, k=1)

print("Some skip-gram probabilities:")
for gram, prob in list(skipgram_probs.items())[:5]:
    print(f"P({gram[-1]}|{' '.join(gram[:-1])}) = {prob:.2f}")
```

## 2. Class-based Language Models

Class-based models group words into classes, reducing the number of parameters and addressing data sparsity:

```python
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

def create_class_based_model(sentences, num_classes=10):
    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Cluster word vectors
    word_vectors = [model.wv[word] for word in model.wv.key_to_index]
    kmeans = KMeans(n_clusters=num_classes)
    kmeans.fit(word_vectors)

    # Assign words to classes
    word_classes = {word: kmeans.predict([model.wv[word]])[0] for word in model.wv.key_to_index}

    # Count class transitions
    class_transitions = defaultdict(int)
    class_counts = defaultdict(int)

    for sentence in sentences:
        for i in range(len(sentence) - 1):
            c1, c2 = word_classes[sentence[i]], word_classes[sentence[i+1]]
            class_transitions[(c1, c2)] += 1
            class_counts[c1] += 1

    # Calculate transition probabilities
    class_transition_probs = {(c1, c2): count / class_counts[c1]
                              for (c1, c2), count in class_transitions.items()}

    return word_classes, class_transition_probs

# Example usage
sentences = [
    ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
    ['a', 'quick', 'brown', 'dog', 'barks', 'at', 'the', 'lazy', 'cat']
]

word_classes, class_transition_probs = create_class_based_model(sentences)

print("Word classes:")
for word, class_ in list(word_classes.items())[:5]:
    print(f"{word}: Class {class_}")

print("\nSome class transition probabilities:")
for (c1, c2), prob in list(class_transition_probs.items())[:5]:
    print(f"P(Class {c2}|Class {c1}) = {prob:.2f}")
```

## 3. Maximum Entropy Language Models

Maximum Entropy (MaxEnt) models, also known as log-linear models, allow for incorporating various features into the language model:

```python
import numpy as np
from scipy.optimize import minimize

def maxent_model(features, labels):
    def neg_log_likelihood(weights):
        scores = np.dot(features, weights)
        probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        return -np.sum(np.log(probs[np.arange(len(labels)), labels]))

    initial_weights = np.zeros(features.shape[1])
    result = minimize(neg_log_likelihood, initial_weights, method='L-BFGS-B')
    return result.x

# Example usage (simplified)
features = np.array([
    [1, 0, 1],  # Feature vector for "the cat"
    [1, 1, 0],  # Feature vector for "the dog"
    [0, 1, 1]   # Feature vector for "a dog"
])
labels = np.array([0, 1, 1])  # 0: "cat", 1: "dog"

weights = maxent_model(features, labels)
print("MaxEnt model weights:", weights)
```

## 4. Introduction to Neural Language Models

Neural language models use neural networks to learn distributed representations of words and predict probabilities.

### 4.1 Word Embeddings

Word embeddings are dense vector representations of words that capture semantic relationships:

```python
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def train_word2vec(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def visualize_embeddings(model, words):
    word_vectors = [model.wv[word] for word in words]
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        x, y = embeddings_2d[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points')
    plt.title("Word Embeddings Visualization")
    plt.xlabel("t-SNE feature 0")
    plt.ylabel("t-SNE feature 1")
    plt.show()

# Example usage
sentences = [
    ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
    ['a', 'quick', 'brown', 'dog', 'barks', 'at', 'the', 'lazy', 'cat']
]

model = train_word2vec(sentences)
words_to_plot = ['quick', 'brown', 'fox', 'dog', 'lazy', 'cat']
visualize_embeddings(model, words_to_plot)
```

### 4.2 Simple Feed-forward Neural Language Model

A basic feed-forward neural language model using TensorFlow:

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Input
from tensorflow.keras.models import Model

def create_ffnn_lm(vocab_size, embedding_dim, context_size):
    inputs = Input(shape=(context_size,))
    embedding = Embedding(vocab_size, embedding_dim)(inputs)
    flattened = tf.keras.layers.Flatten()(embedding)
    hidden = Dense(128, activation='relu')(flattened)
    output = Dense(vocab_size, activation='softmax')(hidden)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

# Example usage (simplified)
vocab_size = 1000
embedding_dim = 50
context_size = 3

model = create_ffnn_lm(vocab_size, embedding_dim, context_size)
model.summary()
```

## 5. Comparing Statistical and Neural Language Models

Let's compare the perplexity of a trigram model with a simple neural language model:

```python
import numpy as np
from collections import defaultdict
import tensorflow as tf

def calculate_trigram_perplexity(test_data, trigram_probs):
    log_prob = 0
    n = 0
    for sentence in test_data:
        for i in range(2, len(sentence)):
            trigram = tuple(sentence[i-2:i+1])
            if trigram in trigram_probs:
                log_prob += np.log2(trigram_probs[trigram])
            else:
                log_prob += np.log2(1e-10)  # Smoothing for unseen trigrams
            n += 1
    perplexity = 2 ** (-log_prob / n)
    return perplexity

def calculate_neural_perplexity(test_data, model, word_to_id):
    log_prob = 0
    n = 0
    for sentence in test_data:
        for i in range(2, len(sentence)):
            context = [word_to_id.get(w, 0) for w in sentence[i-2:i]]
            target = word_to_id.get(sentence[i], 0)
            probs = model.predict(np.array([context]))[0]
            log_prob += np.log2(probs[target])
            n += 1
    perplexity = 2 ** (-log_prob / n)
    return perplexity

# Example usage (simplified)
train_data = [
    ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
    ['a', 'quick', 'brown', 'dog', 'barks', 'at', 'the', 'lazy', 'cat']
]

test_data = [
    ['the', 'brown', 'fox', 'jumps', 'over', 'a', 'lazy', 'cat'],
    ['a', 'quick', 'dog', 'barks', 'at', 'the', 'brown', 'fox']
]

# Calculate trigram probabilities
trigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)

for sentence in train_data:
    for i in range(len(sentence) - 2):
        trigram = tuple(sentence[i:i+3])
        bigram = tuple(sentence[i:i+2])
        trigram_counts[trigram] += 1
        bigram_counts[bigram] += 1

trigram_probs = {trigram: count / bigram_counts[trigram[:2]]
                 for trigram, count in trigram_counts.items()}

# Train neural model
vocab = list(set(word for sentence in train_data + test_data for word in sentence))
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

X = []
y = []
for sentence in train_data:
    for i in range(2, len(sentence)):
        X.append([word_to_id[w] for w in sentence[i-2:i]])
        y.append(word_to_id[sentence[i]])

X = np.array(X)
y = np.array(y)

model = create_ffnn_lm(len(vocab), 50, 2)
model.fit(X, y, epochs=50, verbose=0)

# Calculate and compare perplexities
trigram_perplexity = calculate_trigram_perplexity(test_data, trigram_probs)
neural_perplexity = calculate_neural_perplexity(test_data, model, word_to_id)

print(f"Trigram model perplexity: {trigram_perplexity:.2f}")
print(f"Neural model perplexity: {neural_perplexity:.2f}")
```

## Conclusion

In this session, we've explored advanced techniques in statistical language modeling and introduced neural approaches to language modeling. We've covered:

1. Advanced N-gram techniques, including interpolation and backoff methods
2. Skip-gram models for capturing longer-range dependencies
3. Class-based language models to address data sparsity
4. Maximum Entropy language models for incorporating various features
5. Introduction to neural language models, including word embeddings and a simple feed-forward neural network model
6. Comparison of statistical and neural language models using perplexity

These advanced techniques address some of the limitations of basic N-gram models that we discussed in the previous session. They offer improved performance and flexibility in modeling language.

## Key Takeaways

1. **Interpolation and backoff** techniques combine different order N-grams to improve model robustness.
2. **Skip-gram models** allow for gaps in the N-gram sequence, capturing longer-range dependencies.
3. **Class-based models** group words into classes, reducing the number of parameters and addressing data sparsity.
4. **Maximum Entropy models** provide a flexible framework for incorporating various linguistic features.
5. **Neural language models** learn distributed representations of words and can capture complex patterns in language.
6. **Word embeddings** represent words as dense vectors, capturing semantic relationships between words.

## Future Directions

As we move forward in our study of NLP and language models, we'll explore more advanced neural architectures:

1. **Recurrent Neural Networks (RNNs)**: These models are designed to handle sequential data and can capture long-range dependencies in text.

2. **Long Short-Term Memory (LSTM) networks**: A type of RNN that addresses the vanishing gradient problem, allowing for better modeling of long-term dependencies.

3. **Transformer models**: These attention-based models have revolutionized NLP, leading to state-of-the-art performance on various language tasks.

4. **Pre-trained language models**: Models like BERT, GPT, and their variants, which are pre-trained on large corpora and can be fine-tuned for specific tasks.

## Practical Considerations

When working with language models in real-world applications, consider the following:

1. **Model complexity vs. data size**: More complex models require more data to train effectively. Choose a model appropriate for your dataset size.

2. **Domain-specific adaptation**: Pre-trained models may need to be fine-tuned or adapted for specific domains or tasks.

3. **Computational resources**: Neural models, especially large pre-trained models, can be computationally expensive to train and use. Consider the available resources when choosing a model.

4. **Interpretability**: While neural models often perform better, they can be less interpretable than statistical models. Consider the trade-off between performance and interpretability for your specific application.

5. **Ethical considerations**: Be aware of potential biases in language models, especially when using pre-trained models or working with sensitive applications.

## Exercise

To reinforce your understanding of the concepts covered in this session:

1. Implement a trigram model with linear interpolation using the provided code snippets as a starting point.
2. Train the model on a small corpus (e.g., a collection of news articles or book chapters).
3. Compare the perplexity of your interpolated model with a simple trigram model without interpolation.
4. Experiment with different interpolation weights and observe their effect on the model's performance.
5. (Optional) If you're comfortable with neural networks, try implementing a simple recurrent neural network (RNN) language model and compare its performance with your N-gram models.

```python
# Your code here
```

By completing this exercise, you'll gain hands-on experience with advanced statistical language modeling techniques and begin to appreciate the transition from traditional methods to neural approaches.
