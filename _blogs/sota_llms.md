---
layout: blog
title: "Reading Note: From Transformers to LLMs"
date: 2025-08-14
categories: model archtechture 
tags: [llm, archtechture,reading note]
---

# Reading Note: From Transformers to LLMs

This is a backup for myself to understand LLMs better, this is an ongoing reading note :).

This note is focusing on the details about the model architecture and some insights about why we want to have such kind of architecture. I focused on decoder-only (GPT) models, but I would extend to some other new architectures like Mamba, RWKV and Titan. There is no training algorithm introduced (like supervised fine-tuning, reinforcement learning and so on. All those different training algorithms are introduced in another [note]())
As a personal reading note, there are a lot of copies from the original papers and blogs, also a lot AI-generated pieces.


<!-- TOC -->
<!-- /TOC -->
- [Reading Note: From Transformers to LLMs](#reading-note-from-transformers-to-llms)
  - [Transformers and Basic Knowledge](#transformers-and-basic-knowledge)
    - [Scaled Dot-product Self-Attention](#scaled-dot-product-self-attention)
      - [What's the differences between $Q$, $K$ and $V$?](#whats-the-differences-between-q-k-and-v)
      - [Why self-attention?](#why-self-attention)
    - [Positional Encoding](#positional-encoding)
    - [LayerNorm](#layernorm)
    - [MLP](#mlp)
  - [Encoder-Decoder Models and Encoder-only Models](#encoder-decoder-models-and-encoder-only-models)
    - [Encoder-Decoder Models](#encoder-decoder-models)
    - [Encoder-only Models (Masked Language Models - MLMs)](#encoder-only-models-masked-language-models---mlms)
  - [Decoder-only Models: Current LLMs](#decoder-only-models-current-llms)
    - [GPTs](#gpts)
      - [GPT-1: Improving Language Understanding by Generative Pre-Training](#gpt-1-improving-language-understanding-by-generative-pre-training)
      - [GPT-2](#gpt-2)
      - [GPT-3 (3.5)](#gpt-3-35)
    - [Others](#others)
      - [Deepseek](#deepseek)
      - [Llama](#llama)
      - [Gemma](#gemma)
      - [Mistral](#mistral)
      - [Qwen](#qwen)
      - [Phi-4](#phi-4)
  - [Mixture-of-Experts (MoEs)](#mixture-of-experts-moes)
  - [Other architectures](#other-architectures)
    - [Still sequence based, like Mamba, RWKV, Titan, RetNet](#still-sequence-based-like-mamba-rwkv-titan-retnet)
    - [No auto-reggressive, like latent concept model (LCM)](#no-auto-reggressive-like-latent-concept-model-lcm)
  - [Other resources](#other-resources)
    - [Lectures](#lectures)
    - [Blogs/Tutorials](#blogstutorials)
    - [Papers](#papers)



## Transformers and Basic Knowledge

*Attention* is originally a mechanism designed for RNNs to compute a representation of the sequence by relating different positions of a single sequence. Later on [transformers](https://arxiv.org/pdf/1706.03762) is proposed based on the idea of attention.
This paper proposed the transformer architecture for the machine translation problem. As RNN heavily relies on the previous state, the inherently sequential nature precludes parallelization within training examples. Those authors proposed to only use attention mechanisms to draw global dependencies between input and output. **This method allows for more parallelization and enable the possibility for scaling.**
 For previous translation models, they are all encoder-decoder models, each of those step those models are all *auto-regressive*, which means they consumed the previously generated symbols as additional input when generating the next. Transformer follows this overall architecture.

In this paper, they have one multi-head self-attention layer (I'll call it as attention layer in the following for simplicity) in each encoder block, one masked attention layer and one attention layer in the decoder block. The masking here is to ensure the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

### Scaled Dot-product Self-Attention

>The attention section is extended to more variance in another note fofr attention! Here we only have basic introduction for the later understanding of the model archtechture.

We don't go to the details of attention computation, but here are some great resources to understand [attentions](https://lilianweng.github.io/posts/2018-06-24-attention/). At a high-level, attention is used to compute a representation to the pair of input at position $i$ and output at position $t$, $(y_t, x_i)$, based on how well they match. There are a lot of attentions people proposed before, in the Transformer paper, they proposed the scaled dot-product attention. 
\(
\text{Attention}(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V    
\)
It's very related to the dot-product attention but adding a scaling factor to scaling the attention numbers based on the dimension of the source vector and make sure the dot product values remain in a more reasonable range, preventing extremely large or small softmax outputs.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

*Here is a very brief proof about why we got $\sqrt{d_k}$ as the scaling factor:*

What's the problem if we don't do scaling? -- the variance of $QK^T$ is large for a big $d_k$.

Assume $Q, K$ are two random vectors, $Q=\{q_1, ..., q_{d_k}\}$, $K=\{k_1, ..., k_{d_k}\}$, $q_i, k_i \sim N(0,1)$ and independent. 

\(
Var(QK^T) = Var(\sum_i^{d_k}q_ik_i) = \sum_i^{d_k} Var(q_ik_i) \\
Var(q_ik_i) = E[(q_ik_i)^2]-[E(q_ik_i)]^2  = E(q_i)^2E(k_i)^2 = 1 \\
Var(QK^T) = d_k
\)

#### What's the differences between $Q$, $K$ and $V$?
Or ask this question: With this Q and K vectors we calculate attention and multiply it with V .

From the computation perspective, they are different linear projections of the same input sequence, though each serving a different role by our interpretation.


**<span style="color:red">TODO Papers</span>**
 - [Compositional Attention: Disentangling Search and Retrieval, ICLR 2022](https://arxiv.org/pdf/2110.09419)
 - [Exploring the Space of Key-Value-Query Models with Intention](https://arxiv.org/pdf/2305.10203)

#### Why self-attention?

| Layer Type                 | Complexity per Layer   | Sequential Operations | Maximum Path Length  |
|----------------------------|-----------------------|-----------------------|----------------------|
| Self-Attention            | \(O(n^2 \cdot d)\)    | \(O(1)\)              | \(O(1)\)             |
| Recurrent                 | \(O(n \cdot d^2)\)    | \(O(n)\)              | \(O(n)\)             |
| Convolutional             | \(O(k \cdot n \cdot d^2)\) | \(O(1)\)              | \(O(\log_k(n))\)     |
| Self-Attention (restricted) | \(O(r \cdot n \cdot d)\) | \(O(1)\)              | \(O(n/r)\)           |

1. Computation complexity per layer, parallel computing to improve the training efficiency. length $n$ is smaller than the representation dimensionality $d$ -- **this is not ture in the LLM years for long-context**

2. Learning long-range dependencies 
3. Better interpretability

*Related resources for attention machenism:*
- [Blog: Attention is logarithmic, actually](https://supaiku.com/attention-is-logarithmic): really interesting and helpful blog give an insight of how parallel computation today influence the computation complexity we need to consider.

### Positional Encoding

The shortcoming of the scaled dot attention is that it doesn't account for positions - two vectors will have the same dot product, regardless of how close they are to each other in the input sequence. The position encoding is a way to modify each vector in a way that incorporates informationi about its position.

In the original transformers, the positional encoding is happened after the embedding layer to inject some information about the relative or absolute position of the tokens in the sequence. Because the transformer processes input tokens in parallel, so it does not automatically know their positions. 

(Nice insight from the RoPE paper!) The intuition for adding position encoding through production is because if we compute the dot products of two vectors, it measures the closeness of the two vectors. To get the related position information, for two tokens at position \(m\) and position \(n\), we need a function to express the information of \(n-m\). For a token at position $m$, denoted as $x_m$ and a token at position $n$, $x_n$, in the computation of attention:
\(
a_{m,n} = \text{sfotmax}(q_mk_n^T)=\frac{exp(\frac{q_m^T k_n}{\sqrt{d_k}})}{\sum_{j=1}^N exp(\frac{q_m^T k_n}{\sqrt{d_k}})}    \\
o_{m,n} = \sum_{n=1}^N a_{m,n} v_n
\)

In the computation of attention, the information about the two vector's positioin is contained in the dot production of $q_m$ and $k_n$. So, the position encoding function $f$, which is the function for the $q$ and $k$ vector should be able to transform to a function about the relative position $m-n$. Formally defined as:

\(
\langle f_q(x_m, m), f_k(x_m,n)\rangle = g(x_m, x_n, m-n)
\)

So, for the sinusoidal positional encoding. 

\(
PE(pos, 2i) = sin(pos/\theta^{2i/d_{model}}) \\
PE(pos, 2i+1) = cos(pos/\theta^{2i/d_{model}})
\)

$pos$ is the token's position in the sequence, $i$ is the dimension index, $d_{model}$ is the model's embedding size, \(\theta\) is setted to a big number to avoid the model getting same positional representations of different potions in a large distance. This function can make sure (1) each dimension follows a sinusoidal curve, ensuring unique encodings; (2) allows extrapolation to longer sequences, unlike learned embeddings, which are fixed to training sequence lengths, this function can generalize to unseen lengths.

Rotary position encoding (RoPE), which is still the best popular positional encoding technique. The previous sinusoidal function is a pre-defined function, and other learned and relative position encoding methods are all adding the position information to the context representation and thus render them unsuitable for the *linear self-attention* architecture (check the note for [attention]()). 

> A brief introduction about the linear self-attention:
> The self attention can be rewritten in this way:
> \(
> Attention(Q,K,V)_m = \frac{\sum_{n=1}^N sim(q_m,k_n)v_n}{\sum_{n=1}^N sim(q_m,k_n)} = \frac{\sum_{n=1}^N exp(q_m^Tk_n/\sqrt{d_k})v_n}{\sum_{n=1}^N exp(q_m^Tk_n/\sqrt{d_k})}
> \)
> so, in the original self-attention we should compute the inner product of query and key for every pair of tokens, which has a quadratic complexity. But a linear attentions ([paper](https://proceedings.mlr.press/v119/katharopoulos20a.html?ref=mackenziemorehead.com)) reformulate as:
>  \(
> Attention(Q,K,V)_m = \frac{\sum_{n=1}^N \phi(q_m)^T \psi(k_n) v_n}{\sum_{n=1}^N \phi(q_m)^T \psi(k_n)} 
> \), where $\phi()$ and $\psi()$ are non-negative functions and first compute the multiplication between keys and values using the associative property of matrix multiplication.

The previous position encoding can not suitable for the linear self-attention in modern transformers.
To solve this problem, RoPE proposed to encode the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation.

With the RoPE in linear self-attention, the norm of the hidden representation keep unchanged because the position information is injected by rotation matrix:

\(
Attention(Q,K,V)_m = \frac{\sum_{n-1}^N (R_{\Phi,m}\Theta^d \phi(q_m))^T(R_{\Phi,n}\psi(k_n))v_n}{\sum_{n=1}^{N} \phi(q_m)^T\psi(k_n)}    
\)

In the rotary position embedding, the authors found a solution for vectors on a 2D plane by its complex form:

\(
f_q(x_m, m) = (W_q x_m) e^{im\theta}\\
f_k(x_n, n) = (W_k, x_n) e^{in\theta}\\
\langle f_q(x_m, m), f_k(x_m,n)\rangle = g(x_m, x_n, m-n) = Re[(W_qx_m)(W_kx_n)e^i{m-n}\theta]
\)

in this function, $e^{i m \theta}$ is equal to a rotary matrix, for a more general setting, we can have the rotary matrix for dimension $d_k$.

There are also some newer positional encoding methods, there would be more details in the following sections. Here I just list a brief summary of recent newer positional encoding methods:

| **Method**       | **Generalization**                        | **Memory**           | **Used In**              |
|------------------|--------------------------------------|------------------|-------------------------|
| **[Sinusoidal](https://mfaizan.github.io/2023/04/02/sines.html)**   | ✅ Extrapolates to longer sequences  | 🔹 Small overhead | Transformer (original)  |
| **Learned**      | ❌ Fixed to training length          | 🔸 Higher memory  | BERT, GPT               |
| **Relative (RPE)** | ✅ Handles long-range dependencies  | 🔹 Medium         | Transformer-XL, T5, DeBERTa |
| **[RoPE](https://arxiv.org/pdf/2104.09864)**         | ✅ Efficient for long sequences      | 🔹 Medium         | LLaMA, GPT-NeoX        |
| **[ALiBi](https://openreview.net/pdf?id=R8sQPpGCv0)**        | ✅ Extremely scalable               | ✅ No extra memory | MPT, BLOOM    |

Other resources:
    - https://dm.cs.tu-dortmund.de/en/mlbits/neural-nlp-positional-encoding/

### LayerNorm

Proposed for RNN, suitable for transformers well, make the training much faster than batch normalization. Check the resources for the differences between batch normalization and layer normalization: [1. Blog: BatchNorm and LayerNorm](https://medium.com/@florian_algo/batchnorm-and-layernorm-2637f46a998b), [2. Layer Normalization Paper](https://arxiv.org/pdf/1607.06450).

In the transformers, layer normalization is usually applied before the attention input vectors and after we get the attention out vectors.

### MLP

Multi-level perception (MLP) is a fully connected layer, which can be formulated as:

\(
y=\sigma(W x + b)    
\), where $W$ is the MLP weight and $b$ is the bias, $\sigma$ is tha activation function. 

In traditional deep learning, MLP is usually used to capture the complex patterns in data by transformation through multiple hidden layers, it can be used in tasks like classification, regression and embedding generation. In Transformer modes, MLPs are typically used after a self-attention layer, transform the previous attention output to the next attention input. 

**<span style="color:red">TODO Papers</span>**
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)

## Encoder-Decoder Models and Encoder-only Models

>Brief summary generated by chatGPT, not verified yet, need human effort to do the verification!*

After the original Transformer model, various architectures evolved depending on the downstream task requirements. The two most important early categories are encoder-decoder models and encoder-only models, based on how the **transformer** blocks are used.

### Encoder-Decoder Models
These models follow the full Transformer architecture with both an encoder and a decoder. They are generally used for sequence-to-sequence (seq2seq) tasks like translation, summarization, and question answering.

- T5 (Text-to-Text Transfer Transformer), 2019
    - Paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683)
    - Architecture: Standard encoder-decoder transformer.
    - Key Idea: Reframes all NLP tasks (translation, summarization, classification, etc.) into a text-to-text format. Both the input and output are text strings.
    - Training Objective: Uses a span-corruption objective, where a span of text is masked and replaced with a sentinel token; the model learns to predict the missing span.
    - Pretraining + Finetuning: Trained on a large dataset (C4), then fine-tuned on downstream tasks in a unified format.
    - Positional Encoding: Relative positional encoding, improving generalization for longer sequences.
    - Takeaway: T5 shows the power of unifying tasks into the same framework and the flexibility of encoder-decoder models.
    - Other examples: BART (denoising autoencoder for pretraining seq2seq models), mT5 (multilingual T5), UL2 (unified pretraining with multiple objectives).

### Encoder-only Models (Masked Language Models - MLMs)
These models only use the encoder part of the Transformer and are trained to understand a full sentence or document at once. The most common task is Masked Language Modeling (MLM).

BERT (Bidirectional Encoder Representations from Transformers)
Paper: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Architecture: Transformer encoder stack only.
Pretraining Task: MLM — Randomly masks 15% of tokens and trains the model to predict them using the context on both sides. Also includes Next Sentence Prediction (NSP) as a secondary task (though later studies showed it's not always necessary).
Bidirectionality: Unlike GPT, BERT uses full context from both left and right, which makes it very strong for understanding tasks.
Downstream Tasks: Sentence classification (e.g., sentiment analysis), question answering, NER — mostly tasks where full input is known in advance.
Limitation: Not well-suited for generation tasks (e.g., translation, summarization), as it's not auto-regressive.
Variants include:

RoBERTa: Robustly optimized BERT with longer training and no NSP.
DeBERTa: Introduced disentangled attention and relative positional encoding.
ELECTRA: Replaces MLM with a more sample-efficient replaced token detection.


## Decoder-only Models: Current LLMs

(Copy from [GPT-1 wiki](https://en.wikipedia.org/wiki/GPT-1))

Generative Pre-trained Transformer 1 (GPT-1) was the first of OpenAI's large language models following Google's invention of the transformer architecture in 2017. In June 2018, OpenAI released a paper entitled "Improving Language Understanding by Generative Pre-Training", in which they introduced that initial model along with the general concept of a generative pre-trained transformer.

### GPTs

A visualization of GPT models architecture: [visualization](https://bbycroft.net/llm).

#### GPT-1: Improving Language Understanding by Generative Pre-Training

#### GPT-2

#### GPT-3 (3.5)


### Others

#### Deepseek

#### Llama 

#### Gemma

#### Mistral

#### Qwen

#### Phi-4

## Mixture-of-Experts (MoEs)


## Other architectures 

### Still sequence based, like Mamba, RWKV, Titan, RetNet


### No auto-reggressive, like latent concept model (LCM)


## Other resources

### Lectures

- [Standford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)

### Blogs/Tutorials


### Papers