import torch
import numpy as np
import random
import json
import argparse
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from functools import partial
from sklearn.metrics import precision_score, recall_score, f1_score
from build_glove import load_glove, build_embedding

from blstm_cnn import blstm_cnn
from dataset import NERDataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

@dataclass
class TrainConfig:
    # data 
    train_dir: str = "data/train"
    dev_dir: str = "data/dev"
    test_dir: str = "data/test"

    # model
    char_vocab_size: int = 100
    char_embed_size: int = 30
    num_filters: int = 50
    kernel_size: int = 3
    word_vocab_size: int = 100
    word_embed_size: int = 100
    embed_size: int = 150  # num_filters + word_embed_size
    hidden_size: int = 256
    output_size: int = 128
    num_classes: int = 9
    num_layers: int = 1
    num_cnn_layers: int = 1
    dropout: float = 0.33
    case_embed_size: int = 0

    # training 
    batch_size: int = 16
    epochs: int = 25
    lr: float = 0.001
    momentum: float = 0.9
    val_iter: int = 2
    optimizer: str = "adam" # or adam
    use_glove: bool = True
    lr_factor: float = 0.1
    rho: float = 0.05

    

def train_model(model, train_dataloader, test_dataloader, config: TrainConfig, device="cuda"):
    model = model.to(device)

    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=1e-4)
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))        

    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + config.rho * epoch)
    )
    
    best_val_f1 = 0.0

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            input_ids = batch['input_ids'].to(device)
            char_ids = batch['char_ids'].to(device)
            tag_labels = batch['labels'].to(device)
            case_ids = batch['case_ids'].to(device)
            mask = tag_labels != -1

            optimizer.zero_grad()
            emissions = model(input_ids, char_ids, mask, case_ids=case_ids)
            loss = model.crf_loss(emissions, tag_labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        scheduler.step()
        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % config.val_iter == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    char_ids = batch['char_ids'].to(device)
                    tag_labels = batch['labels'].to(device)
                    case_ids = batch['case_ids'].to(device)
                    mask = tag_labels != -1

                    emissions = model(input_ids, char_ids, mask, case_ids=case_ids)
                    loss = model.crf_loss(emissions, tag_labels, mask)
                    val_loss += loss.item()
                    val_batches += 1

                    pred_paths = model.decode(emissions, mask)
                    for preds_seq, labels_row, m in zip(pred_paths, tag_labels, mask):
                        all_preds.extend(preds_seq)
                        all_labels.extend(labels_row[m].cpu().tolist())

            avg_val_loss = val_loss / val_batches
            precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
            recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
            f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            print(f"Val Loss: {avg_val_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

            if f1 > best_val_f1:
                best_val_f1 = f1
                torch.save(model.state_dict(), "blstm_cnn.pt")
                print(f"Saved best checkpoint (F1: {f1:.4f})")

def collate_fn(batch, pad_idx, label_pad_idx):
    inputs = [item[0] for item in batch]
    char_ids = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    case_ids = [item[3] for item in batch]

    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=label_pad_idx)
    padded_chars = pad_sequence(char_ids, batch_first=True, padding_value=0)
    padded_case_ids = pad_sequence(case_ids, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_inputs,
        "char_ids": padded_chars,
        "labels": padded_labels,
        "case_ids": padded_case_ids,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the config
    if args.config:
        with open(args.config, "r") as f:
            raw = json.load(f)
        flat = {}
        for v in raw.values():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[v] = v
        config = TrainConfig(**{k: v for k, v in flat.items() if hasattr(TrainConfig, k)})
    else:
        config = TrainConfig()

    train_dataset = NERDataset(config.train_dir)
    dev_dataset = NERDataset(config.dev_dir, word2idx=train_dataset.word2idx, tag2idx=train_dataset.tag2idx, char2idx=train_dataset.char2idx)

    my_collate = partial(collate_fn, pad_idx=len(train_dataset.word2idx), label_pad_idx=-1)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)

    # update the config vocab
    config.char_vocab_size = len(train_dataset.char2idx) + 1  # +1 for padding at index 0
    config.word_vocab_size = len(train_dataset.word2idx) + 2  # +2 for unk and unseen tokens
    config.num_classes = len(train_dataset.tag2idx)
    config.embed_size = config.num_filters + config.word_embed_size

    model = blstm_cnn(
        char_vocab_size=config.char_vocab_size,
        char_embed_size=config.char_embed_size,
        num_filters=config.num_filters,
        kernel_size=config.kernel_size,
        word_vocab_size=config.word_vocab_size,
        word_embed_size=config.word_embed_size,
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        num_classes=config.num_classes,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_cnn_layers=config.num_cnn_layers,
        case_embed_size=config.case_embed_size,
    )

    # check if we need to build with glove embeddings
    if config.use_glove:
        glove_path = "glove.6B.100d.gz"
        word2vec = load_glove(glove_path)
        print(f"Loaded {len(word2vec)} words, dim={len(next(iter(word2vec.values())))}")

        embedding = build_embedding(train_dataset.word2idx, word2vec, embed_dim=config.word_embed_size)
        print(f"Embedding shape: {embedding.weight.shape}")

        model.word_embedding = embedding
    
    train_model(model, train_dataloader, dev_dataloader, config, device)

if __name__ == "__main__":
    main()