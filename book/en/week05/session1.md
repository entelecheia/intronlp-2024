# Week 5 Session 1 - Transformers and Their Applications

## 1. Introduction to Transformers

Transformers, introduced in 2017 by Google Brain researchers in the paper "Attention Is All You Need", have revolutionized Natural Language Processing (NLP) tasks. They are built upon the concept of attention, a mechanism that allows the model to focus on specific parts of the input data.

```{figure} figs/entelecheia_transformers.png
:alt: transformers
:class: bg-primary mb-1
:width: 70%
:align: center
```

Key applications of Transformers include:

- Machine translation
- Text summarization
- Question answering
- Image classification
- Speech recognition

The Transformer model has shown great promise in various domains, from NLP to computer vision and beyond.

```{figure} figs/transformers-history.jpeg
---
width: 70%
name: fig-transformers-history
---
History of Transformers
```

## 2. Why Transformers?

Transformers were developed to address several limitations of Recurrent Neural Networks (RNNs):

1. **Parallel processing**: RNNs process input sequentially, while Transformers can process all elements of a sequence simultaneously. This enables Transformers to take full advantage of parallel computing resources.

2. **Long-range dependencies**: RNNs struggle to capture relationships between distant elements in a sequence due to the vanishing gradient problem. Transformers, through their attention mechanism, can directly model relationships between all elements in the input.

3. **Computational efficiency**: The parallel nature of Transformers allows for better utilization of modern hardware, leading to faster training and inference times.

```{figure} figs/problem-rnn.gif
---
width: 70%
name: fig-problem-rnn
---
Problem with RNNs
```

This animation illustrates the sequential nature of RNNs, highlighting the challenges they face in processing long sequences efficiently.

## 3. Attention Mechanism

The attention mechanism is the core component of the Transformer architecture. It allows the model to focus on different parts of the input data that are most relevant to the task at hand.

```{figure} figs/attention.gif
---
width: 70%
name: fig-attention
---
Attention
```

Consider the example sentence: "I gave my dog Charlie some food."

The attention mechanism can identify important relationships:

1. "Who gave the food?" - Focus on "I"
2. "To whom was the food given?" - Focus on "dog" and "Charlie"
3. "What was given?" - Focus on "food"

Implementation of Attention:

1. Encode words as vectors (keys)
2. Define a query vector
3. Compute similarity between query and keys
4. Normalize scores using softmax
5. Compute context vector
6. Apply linear transformation

```{figure} figs/attention-calculate.gif
---
width: 70%
name: fig-attention-calculate
---
Attention mechanism
```

This animation demonstrates the process of calculating attention scores and generating the context vector.

## 4. Multi-head Attention

Multi-head attention is an extension of the basic attention mechanism that uses multiple sets of queries, keys, and values ("heads"). Each head captures different aspects of relationships between words in the input.

```{figure} figs/attention-multihead.jpeg
---
width: 70%
name: fig-attention-multihead
---
Multi-head attention
```

Process:

1. Compute attention for each head
2. Concatenate results from all heads
3. Apply final linear transformation

Multi-head attention enhances the model's ability to capture a more comprehensive understanding of the relationships between words in the input.

## 5. Transformer Architecture

The Transformer architecture consists of two primary components: the encoder and the decoder.

```{figure} figs/transformer-architecture.gif
---
width: 70%
name: fig-transformer-architecture
---
Transformer architecture
```

### Encoder:

1. Tokenization: Break input sentence into individual words
2. Word Embedding: Convert words to vectors
3. Positional Encoding: Add information about word order
4. Multi-head Attention: Capture relationships between words
5. Normalization and Feed-Forward Neural Network
6. Stacking Encoder Layers: Repeat process for deeper understanding

### Decoder:

1. Input Preparation: Take previously translated words and encoder output
2. Positional Encoding and Multi-head Attention
3. Concatenation and Recalculation: Combine with encoder output
4. Normalization and Feed-Forward Neural Network
5. Predicting Next Word
6. Iterative Decoding: Repeat until full translation is generated

## 6. Challenges with Transformers

Despite their success, Transformers face some challenges:

1. **Computational Complexity**: The attention mechanism requires O(N^2) calculations for N words.
2. **Quadratic Scaling**: Attention calculation fills an NxN matrix, leading to resource-intensive operations for long sequences.
3. **Masked Attention**: A partial solution that reduces complexity but doesn't fully resolve the issue for very long sequences.

```{figure} figs/transformer-N2.gif
---
width: 70%
name: fig-transformer-N2
---
Attention matrix
```

This animation illustrates the quadratic scaling problem in Transformers, showing how the attention matrix grows with the input size.

## 7. Beyond Attention: "Attention Is Not All You Need"

Recent research has challenged the idea that self-attention alone is key to Transformer success. Google researchers found that self-attention in isolation converges to a rank 1 matrix, which is ineffective without other components.

The true power of Transformers comes from the combination of:

1. Self-attention mechanism
2. Skip connections
3. MLP (Multilayer Perceptron)

```{figure} figs/transformer-tug-of-war.jpeg
---
width: 70%
name: fig-transformer-tug-of-war
---
Tug of war between the self-attention mechanism and skip connections and MLP
```

This image illustrates the interplay between different components in the Transformer architecture, emphasizing that attention is not the sole contributor to its success.

## 8. Vision Transformers (ViT)

Vision Transformers apply the Transformer architecture to image processing tasks.

```{figure} figs/vit.jpeg
---
width: 70%
name: fig-vit
---
Vision Transformer
```

Process:

1. Divide image into patches
2. Convert patches to vectors
3. Add positional encoding
4. Process through Transformer layers
5. Output to classifier

```{figure} figs/vit-architecture.gif
---
width: 70%
name: fig-vit-architecture
---
Vision Transformer Architecture
```

This animation shows the step-by-step process of how a Vision Transformer processes an input image.

## 9. Transformer in Transformer (TnT)

TnT addresses the loss of spatial information in Vision Transformers by preserving pixel arrangement within patches.

```{figure} figs/tnt.gif
---
width: 70%
name: fig-tnt
---
Transformer in Transformer
```

Process:

1. Transform patch to c-channel tensor
2. Divide tensor into smaller parts
3. Obtain vectors carrying spatial information
4. Concatenate and project vectors
5. Combine with original patch vector

```{figure} figs/tnt-architecture.gif
---
width: 70%
name: fig-tnt
---
Transformer in Transformer Architecture
```

This animation demonstrates how TnT preserves spatial information within image patches.

## 10. TimeSformers

TimeSformers extend Vision Transformers to video processing by introducing a temporal dimension.

```{figure} figs/timesformer-architecture.gif
---
width: 70%
name: fig-timesformer-architecture
---
TimeSformer Architecture
```

Key innovation: Divided Space-Time Attention

- Compute spatial attention across entire frame
- Calculate temporal attention within same patch across frames

```{figure} figs/timesformer-attentions.jpeg
---
width: 70%
name: fig-timesformer-attentions
---
TimeSformer Attention Mechanisms
```

This image shows different attention mechanisms in TimeSformers, with Divided Space-Time Attention proving most effective.

## 11. Multimodal Machine Learning

Multimodal Machine Learning aims to develop models capable of processing and combining different types of data. Transformers show promise in handling diverse data sources within a single model.

Examples:

- VATT (Visual-Audio Text Transformer): Processes video, audio, and text simultaneously

```{figure} figs/vatt.gif
---
width: 70%
name: fig-vatt
---
VATT Architecture
```

- GATO: Generalist Agent capable of multiple tasks and input types

```{figure} figs/gato.gif
---
width: 70%
name: fig-gato
---
GATO: A Generalist Agent
```

## Conclusion

Transformers have significantly impacted NLP and are expanding into computer vision and multimodal learning. Their ability to process various data types and perform multiple tasks suggests a bright future in advancing AI systems. As research continues, we can expect further innovations and applications of Transformer-based models across various domains.
