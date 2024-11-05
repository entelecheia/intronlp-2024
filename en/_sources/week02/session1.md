# Week 2 Session 1: Text Preprocessing Fundamentals

## Introduction

In this lecture, we'll dive deep into the fundamentals of text preprocessing, a crucial step in any Natural Language Processing (NLP) pipeline. Text preprocessing involves transforming raw text data into a clean, standardized format that can be easily analyzed by machine learning algorithms.

Common preprocessing steps include:

1. Text cleaning
2. Lowercase conversion
3. Tokenization
4. Stop word removal
5. Stemming or lemmatization
6. Text representation

```{mermaid}
:align: center
graph TD
    A[Raw Text] --> B[Text Cleaning]
    B --> C[Lowercase Conversion]
    C --> D[Tokenization]
    D --> E[Stop Word Removal]
    E --> F[Stemming/Lemmatization]
    F --> G[Text Representation]
    G --> H[Processed Text]
```

Let's start by importing the necessary libraries:

```python
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 1. Importance of Text Preprocessing

Text preprocessing is essential in NLP for several reasons:

1. It helps remove noise and irrelevant information from the text.
2. It standardizes the text, making it easier for algorithms to process.
3. It can significantly improve the performance of downstream NLP tasks.

Let's visualize the typical text preprocessing pipeline:

```python
import networkx as nx

def create_preprocessing_pipeline_graph():
    G = nx.DiGraph()
    steps = ["Raw Text", "Text Cleaning", "Lowercase Conversion", "Tokenization",
             "Stop Word Removal", "Stemming/Lemmatization", "Processed Text"]
    G.add_edges_from(zip(steps[:-1], steps[1:]))

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=3000, font_size=10, font_weight='bold',
            arrows=True, edge_color='gray')

    plt.title("Text Preprocessing Pipeline")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

create_preprocessing_pipeline_graph()
```

This pipeline illustrates the typical steps in text preprocessing. Now, let's explore each step in detail.

## 2. Text Cleaning

Text cleaning involves removing or replacing elements in the text that are not relevant to the analysis. This step is particularly important when dealing with web-scraped data or social media content.

Common text cleaning operations include:

- Removing HTML tags
- Handling special characters
- Removing or replacing URLs and email addresses
- Dealing with numbers and dates

Let's implement a basic text cleaning function:

```python
def clean_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Example usage
sample_text = "<p>Check out our website at https://example.com or email us at info@example.com. Special offer: 50% off!</p>"
cleaned_text = clean_text(sample_text)
print("Original text:", sample_text)
print("Cleaned text:", cleaned_text)
```

## 3. Lowercase Conversion

Converting text to lowercase helps standardize the text and reduce the vocabulary size. However, it's important to consider when case information might be relevant (e.g., for named entity recognition).

```python
def to_lowercase(text):
    return text.lower()

# Example usage
sample_text = "The Quick Brown Fox Jumps Over The Lazy Dog"
lowercased_text = to_lowercase(sample_text)
print("Original text:", sample_text)
print("Lowercased text:", lowercased_text)
```

Let's visualize the effect of lowercase conversion on word frequency:

```python
def plot_word_frequency(text, title):
    words = word_tokenize(text)
    word_freq = nltk.FreqDist(words)
    plt.figure(figsize=(12, 6))
    word_freq.plot(30, cumulative=False)
    plt.title(title)
    plt.show()

plot_word_frequency(sample_text, "Word Frequency (Original)")
plot_word_frequency(lowercased_text, "Word Frequency (Lowercased)")
```

As you can see, lowercase conversion reduces the number of unique words, which can be beneficial for many NLP tasks.

## 4. Tokenization

Tokenization is the process of breaking text into smaller units, typically words or sentences. It's a fundamental step in many NLP tasks.

```python
def tokenize_text(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return words, sentences

# Example usage
sample_text = "This is the first sentence. And here's another one. What about a question?"
words, sentences = tokenize_text(sample_text)
print("Words:", words)
print("Sentences:", sentences)
```

Let's visualize the tokenization process:

```python
def visualize_tokenization(text):
    words, sentences = tokenize_text(text)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Word tokenization
    ax1.bar(range(len(words)), [1]*len(words), align='center')
    ax1.set_xticks(range(len(words)))
    ax1.set_xticklabels(words, rotation=45, ha='right')
    ax1.set_title('Word Tokenization')
    ax1.set_ylabel('Token')

    # Sentence tokenization
    ax2.bar(range(len(sentences)), [1]*len(sentences), align='center')
    ax2.set_xticks(range(len(sentences)))
    ax2.set_xticklabels(sentences, rotation=0, ha='center', wrap=True)
    ax2.set_title('Sentence Tokenization')
    ax2.set_ylabel('Sentence')

    plt.tight_layout()
    plt.show()

visualize_tokenization(sample_text)
```

This visualization helps understand how the text is broken down into individual words and sentences.

## 5. Stop Word Removal

Stop words are common words that usually don't contribute much to the meaning of a text. Removing them can help reduce noise in the data.

```python
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Example usage
sample_text = "This is a sample sentence with some very common words."
filtered_text = remove_stopwords(sample_text)
print("Original text:", sample_text)
print("Text without stopwords:", filtered_text)
```

Let's visualize the effect of stop word removal:

```python
def plot_word_cloud(text, title):
    from wordcloud import WordCloud

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

plot_word_cloud(sample_text, "Word Cloud (With Stop Words)")
plot_word_cloud(filtered_text, "Word Cloud (Without Stop Words)")
```

As you can see, removing stop words helps focus on the more meaningful content words.

## 6. Stemming and Lemmatization

Stemming and lemmatization are techniques used to reduce words to their root form, which can help in reducing vocabulary size and grouping similar words.

### Stemming

```python
def stem_words(text):
    ps = PorterStemmer()
    words = word_tokenize(text)
    stemmed_words = [ps.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Example usage
sample_text = "The runner runs quickly through the running track"
stemmed_text = stem_words(sample_text)
print("Original text:", sample_text)
print("Stemmed text:", stemmed_text)
```

### Lemmatization

```python
def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Example usage
sample_text = "The children are playing with their toys"
lemmatized_text = lemmatize_words(sample_text)
print("Original text:", sample_text)
print("Lemmatized text:", lemmatized_text)
```

Let's compare stemming and lemmatization:

```python
def compare_stem_lemma(text):
    stemmed = stem_words(text)
    lemmatized = lemmatize_words(text)

    words = word_tokenize(text)
    stem_words = word_tokenize(stemmed)
    lemma_words = word_tokenize(lemmatized)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(words))
    ax.plot(x, [1]*len(words), 'bo', label='Original')
    ax.plot(x, [0.66]*len(stem_words), 'ro', label='Stemmed')
    ax.plot(x, [0.33]*len(lemma_words), 'go', label='Lemmatized')

    for i, (orig, stem, lemma) in enumerate(zip(words, stem_words, lemma_words)):
        ax.text(i, 1, orig, ha='center', va='bottom')
        ax.text(i, 0.66, stem, ha='center', va='bottom', color='red')
        ax.text(i, 0.33, lemma, ha='center', va='bottom', color='green')

    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend()
    ax.set_title('Comparison of Original, Stemmed, and Lemmatized Words')

    plt.tight_layout()
    plt.show()

compare_stem_lemma("The runner runs quickly through the running track")
```

This visualization helps understand the differences between stemming and lemmatization.

## 7. Putting It All Together

Now that we've covered all the individual steps, let's create a comprehensive text preprocessing function:

```python
def preprocess_text(text, remove_stopwords=True, stem=False, lemmatize=True):
    # Clean the text
    text = clean_text(text)

    # Convert to lowercase
    text = to_lowercase(text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords if specified
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

    # Stem or lemmatize
    if stem:
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words]
    elif lemmatize:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# Example usage
sample_text = """<p>The quick brown fox jumps over the lazy dog.
                  This sentence is often used in typography to showcase fonts.
                  Visit https://example.com for more information.</p>"""

preprocessed_text = preprocess_text(sample_text)
print("Original text:", sample_text)
print("Preprocessed text:", preprocessed_text)
```

## 8. Conclusion and Best Practices

In this lecture, we've covered the fundamental techniques of text preprocessing for NLP tasks. Here are some best practices to keep in mind:

1. Always consider your specific task and dataset when deciding which preprocessing steps to apply.
2. Be consistent in your preprocessing across training and testing data.
3. Be cautious about removing too much information (e.g., numbers might be important for some tasks).
4. Document your preprocessing steps clearly, as they can significantly impact your results.
5. Consider the trade-offs between different techniques (e.g., stemming vs. lemmatization).
6. When working with multilingual data, ensure your preprocessing techniques are language-appropriate.

Remember, preprocessing is not just a technical step, but an analytical one that requires careful consideration of your research goals and the nature of your data.

In the next session, we'll explore more advanced preprocessing techniques and dive into text representation methods. Stay tuned!

## Exercise

Try applying the `preprocess_text` function to a paragraph of your choice. Experiment with different combinations of parameters (e.g., with and without stop word removal, using stemming vs. lemmatization) and observe how they affect the output. Reflect on which combination might be most appropriate for different types of NLP tasks.

```python
# Your code here
```

This exercise will help you understand the impact of different preprocessing choices and develop intuition for when to use each technique.
