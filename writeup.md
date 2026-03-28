# CSCI 544 Homework 3: NER with BiLSTM

## Task 1: Simple BiLSTM

### Architecture

The model follows the specified architecture: Embedding -> BLSTM -> Linear -> ELU -> classifier.

| Hyper-parameter      | Value |
|----------------------|-------|
| Embedding dim        | 100   |
| Number of LSTM layers| 1     |
| LSTM hidden dim      | 256   |
| LSTM dropout         | 0.33  |
| Linear output dim    | 128   |

### Training Hyper-parameters

| Hyper-parameter      | Value |
|----------------------|-------|
| Optimizer            | SGD   |
| Learning rate        | 0.1   |
| Momentum             | 0.9   |
| Batch size           | 16    |
| Epochs               | 25    |
| LR scheduler         | ReduceLROnPlateau (factor=0.1, patience=2) |
| Gradient clipping    | 5.0   |

The vocabulary is built from the training set. Words unseen at test time are mapped to an UNK token. No pre-trained embeddings are used; embeddings are learned from scratch.

A high initial learning rate (0.1) with aggressive ReduceLROnPlateau scheduling was found to work best. This allows the model to make large updates early and decay quickly once the F1 score plateaus.

### Dev Results

| Metric    | Score |
|-----------|-------|
| Precision | XX.XX |
| Recall    | XX.XX |
| F1        | XX.XX |

---

## Task 2: BiLSTM with GloVe Embeddings

### Architecture

Same architecture as Task 1. The embedding layer is initialized with GloVe 100d vectors.

| Hyper-parameter      | Value |
|----------------------|-------|
| Embedding dim        | 100   |
| Number of LSTM layers| 1     |
| LSTM hidden dim      | 256   |
| LSTM dropout         | 0.33  |
| Linear output dim    | 128   |

### Training Hyper-parameters

| Hyper-parameter      | Value |
|----------------------|-------|
| Optimizer            | Adam  |
| Learning rate        | 0.001 |
| Batch size           | 32    |
| Epochs               | 50    |
| LR scheduler         | ReduceLROnPlateau (factor=0.3, patience=2) |
| Gradient clipping    | 5.0   |
| GloVe embeddings     | Yes (glove.6B.100d) |

### Capitalization Strategy

GloVe is case-insensitive (all lowercase), but capitalization is critical for NER (e.g., "Apple" the company vs. "apple" the fruit). Our strategy for handling this:

1. **Case-sensitive vocabulary**: The word2idx mapping preserves the original casing from the training data. "Apple" and "apple" are separate vocabulary entries with distinct indices.

2. **Fallback GloVe lookup**: When building the embedding matrix, we first try an exact match in GloVe. If that fails (which it will for capitalized words since GloVe is lowercase), we fall back to the lowercased version of the word. This way, "Apple" still gets a meaningful GloVe initialization (from "apple") rather than a random vector.

3. **Fine-tuning**: The embedding layer is not frozen, so during training the model can learn to differentiate the embeddings of "Apple" and "apple" based on the NER supervision signal, even though they start from the same GloVe vector.

This approach ensures that every word with a GloVe-matchable lowercase form gets a good initialization, while the case-sensitive vocabulary allows the model to learn capitalization-dependent behavior through fine-tuning.

### Dev Results

| Metric    | Score |
|-----------|-------|
| Precision | XX.XX |
| Recall    | XX.XX |
| F1        | XX.XX |

---

## Bonus: CNN-BiLSTM-CRF

### Architecture

The bonus model extends Task 2 with a character-level CNN and a CRF decoder:

Character Embedding -> Conv1d -> MaxPool -> concat with Word Embedding -> BLSTM -> Linear -> ELU -> CRF

| Hyper-parameter        | Value |
|------------------------|-------|
| Char embedding dim     | 30    |
| CNN filters            | 30    |
| CNN kernel size        | 3     |
| CNN layers             | 1     |
| CNN padding            | 1 (kernel_size // 2) |
| Word embedding dim     | 100   |
| LSTM input dim         | 130 (30 + 100) |
| LSTM hidden dim        | 200   |
| Number of LSTM layers  | 1     |
| LSTM dropout           | 0.5   |
| Linear output dim      | 128   |

The character-level CNN applies a 1D convolution over the character embeddings of each word, followed by ReLU and max-over-time pooling. This produces a fixed-size representation for each word that captures morphological features (prefixes, suffixes, shape). The character representation is concatenated with the GloVe word embedding before being fed into the BiLSTM.

The CRF layer models transition scores between tags, enforcing valid BIO sequences (e.g., I-PER cannot follow B-LOC). During training, we minimize the CRF negative log-likelihood (forward algorithm for the partition function minus the gold path score). During inference, we use Viterbi decoding to find the highest-scoring tag sequence.

### Training Hyper-parameters

| Hyper-parameter      | Value |
|----------------------|-------|
| Optimizer            | SGD   |
| Learning rate        | 0.015 |
| Momentum             | 0.9   |
| Weight decay         | 1e-4  |
| Batch size           | 10    |
| Epochs               | 35    |
| LR scheduler         | LambdaLR: lr / (1 + 0.05 * epoch) |
| Gradient clipping    | 5.0   |
| GloVe embeddings     | Yes (glove.6B.100d) |
| Char dropout         | 0.5   |

### Dev Results

| Metric    | Score |
|-----------|-------|
| Precision | XX.XX |
| Recall    | XX.XX |
| F1        | XX.XX |
