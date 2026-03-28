import gzip
import torch
import torch.nn as nn
import numpy as np

def load_glove(glove_path):
    word2vec = {}
    with gzip.open(glove_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            word2vec[word] = vec
    return word2vec

def build_embedding(word2idx, word2vec, embed_dim):
    vocab_size = len(word2idx) + 2  # +2 for pad and unk
    matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    found = 0
    for word, idx in word2idx.items():
        if word in word2vec:
            matrix[idx] = word2vec[word]
            found += 1
        elif word.lower() in word2vec:
            matrix[idx] = word2vec[word.lower()]
            found += 1
        else:
            matrix[idx] = np.random.normal(scale=0.6, size=(embed_dim,))
    print(f"GloVe coverage: {found}/{len(word2idx)} ({100*found/len(word2idx):.1f}%)")
    embedding = nn.Embedding.from_pretrained(torch.tensor(matrix), padding_idx=len(word2idx), freeze=False)
    return embedding

if __name__ == "__main__":
    from dataset import NERDataset

    glove_path = "glove.6B.100d.gz"
    word2vec = load_glove(glove_path)
    print(f"Loaded {len(word2vec)} words, dim={len(next(iter(word2vec.values())))}")

    dataset = NERDataset("data/train")
    embedding = build_embedding(dataset.word2idx, word2vec, embed_dim=100)
    print(f"Embedding shape: {embedding.weight.shape}")
