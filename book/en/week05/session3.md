# Week 5 Session 3: Practical Implementation and Visualization of Transformers

## Introduction

In this session, we will focus on the practical aspects of the Transformer architecture. We'll visualize how attention mechanisms work and walk through implementing a Transformer model using Python and PyTorch. This hands-on approach will deepen your understanding of how Transformers process and generate language.

---

## Visualization of Attention Mechanisms

Understanding how attention works internally is crucial. Visualizations can help demystify the "black box" nature of neural networks.

### Visualizing Attention Scores

Attention scores indicate how much focus the model places on different parts of the input when generating each part of the output.

#### Example: Attention Heatmap

Suppose we have an input sentence:

- **Input**: "The cat sat on the mat."

When processing this sentence, the model computes attention scores between each pair of words.

#### Diagram: Attention Heatmap Matrix

```
           The   cat   sat   on   the   mat
        +-------------------------------------
    The | 0.1   0.2   0.3   0.1   0.2   0.1
    cat | 0.2   0.1   0.3   0.1   0.2   0.1
    sat | 0.1   0.2   0.1   0.3   0.2   0.1
     on | 0.1   0.1   0.2   0.1   0.3   0.2
    the | 0.2   0.1   0.1   0.2   0.1   0.3
    mat | 0.1   0.2   0.1   0.2   0.1   0.3
```

_Figure 1: An example attention heatmap showing attention scores between words._

#### Interpretation

- **High Scores**: Indicate strong attention between words.
- **Symmetry**: In self-attention, the matrix is often symmetric.
- **Focus**: Words like "the" may have lower attention scores due to being common.

### Interpreting Attention Maps

Visual attention maps help in understanding which parts of the input influence the output.

#### Example: Translation Task

- **Source Sentence**: "Das ist ein Beispiel."
- **Target Sentence**: "This is an example."

#### Diagram: Cross-Attention Map

```
             This   is   an   example
        +-----------------------------
     Das | 0.7   0.1   0.1   0.1
     ist | 0.1   0.7   0.1   0.1
      ein | 0.1   0.1   0.7   0.1
Beispiel | 0.1   0.1   0.1   0.7
```

_Figure 2: Cross-attention map between source and target sentences._

#### Interpretation

- **Alignment**: High attention scores align words with their translations.
- **Disambiguation**: Helps identify how the model handles word order and syntax differences.

---

## Practical Implementation of a Transformer

Let's dive into building a simple Transformer model to solidify our understanding.

### Building a Simple Transformer from Scratch

We will implement a minimal Transformer model using PyTorch, focusing on key components:

- **Embedding Layer**
- **Positional Encoding**
- **Multi-Head Attention**
- **Feed-Forward Network**
- **Encoder and Decoder Layers**

### Implementing with PyTorch

#### Libraries Needed

```python
import torch
import torch.nn as nn
import math
```

### Code Walkthrough

#### 1. Positional Encoding

Positional encoding injects sequence order information.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x
```

#### 2. Multi-Head Attention Module

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # Linear projections
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        x = torch.matmul(attention, v)

        # Concatenate heads
        x = x.transpose(1,2).contiguous().view(bs, -1, self.d_k * self.n_heads)
        return self.out_proj(x)
```

#### 3. Feed-Forward Network

```python
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
```

#### 4. Encoder Layer

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-Attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        # Feed-Forward Network
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x
```

#### 5. Decoder Layer

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked Self-Attention
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn_output)
        # Cross-Attention
        cross_attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + cross_attn_output)
        # Feed-Forward Network
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        return x
```

#### 6. Assembling the Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, num_layers=6):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embedding and Positional Encoding
        src = self.pos_encoder(self.src_embedding(src))
        tgt = self.pos_encoder(self.tgt_embedding(tgt))

        # Encoder
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # Decoder
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        output = self.fc_out(tgt)
        return output
```

#### 7. Generating Masks

Masks are essential to prevent the model from attending to future tokens during training.

```python
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
```

---

## Advanced Topics

### Transformer Variants

#### BERT (Bidirectional Encoder Representations from Transformers)

- **Architecture**: Uses only the encoder part of the Transformer.
- **Objective**: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
- **Usage**: Excellent for tasks requiring understanding of the context, such as question answering and sentiment analysis.

#### GPT (Generative Pre-trained Transformer)

- **Architecture**: Utilizes the decoder part of the Transformer.
- **Objective**: Language modeling (predicting the next word).
- **Usage**: Effective for text generation tasks.

### Applications of Transformers

- **Machine Translation**: High-quality translation between languages.
- **Text Summarization**: Generating concise summaries of documents.
- **Question Answering**: Providing answers based on context.
- **Text Generation**: Creative writing, code generation, and more.

---

## Optional Coding Task Walkthrough

### Implementing a Simple Attention Mechanism

#### Self-Attention Function

```python
def simple_self_attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output
```

#### Example Usage

```python
# Sample input tensor
x = torch.rand(1, 5, 64)  # (batch_size, sequence_length, d_model)

# Assuming Q, K, V are all x in self-attention
output = simple_self_attention(x, x, x)
print(output.shape)  # Should output (1, 5, 64)
```

---

## Conclusion

In this session, we've taken a hands-on approach to understanding Transformers. By visualizing attention mechanisms and implementing a Transformer model from scratch, you should now have a deeper appreciation for how these models function internally. This practical experience is invaluable as we move toward working with large language models and APIs in the coming weeks.

---

## References

- Vaswani et al., "Attention is All You Need" ([Paper Link](https://arxiv.org/abs/1706.03762))
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
