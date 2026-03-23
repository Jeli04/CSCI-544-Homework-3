import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from functools import partial
from sklearn.metrics import precision_score, recall_score, f1_score
from build_glove import load_glove, build_embedding

from blstm import blstm
from dataset import NERDataset

@dataclass
class TrainConfig:
    # data 
    train_dir: str = "/Users/jerryli/Desktop/CSCI544/HW3 2/data/train"
    dev_dir: str = "/Users/jerryli/Desktop/CSCI544/HW3 2/data/dev"
    test_dir: str = "/Users/jerryli/Desktop/CSCI544/HW3 2/data/test"

    # model
    vocab_size: int = 100 # default one 
    embed_size: int = 100
    hidden_size: int = 256
    output_size: int = 128
    num_classes: int = 9
    num_layers: int = 1
    dropout: float = 0.33

    # training 
    batch_size: int = 16
    epochs: int = 20
    lr: float = 0.01
    momentum: float = 0.9
    val_iter: int = 2
    use_glove: bool = False
    

def train_model(model, train_dataloader, test_dataloader, config: TrainConfig):
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
        optimizer,
        mode="max",  
        factor=0.75, # half the lr
        patience=1,
        verbose=True
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    best_val_f1 = 0.0

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            input_ids = batch['input_ids']
            tag_labels = batch['labels']

            optimizer.zero_grad()
            logits = model(input_ids).permute(0, 2, 1)  # [batch, num_classes, seq_len]
            loss = loss_fn(logits, tag_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f}")

        if epoch % config.val_iter == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Validation"):
                    input_ids = batch['input_ids']
                    tag_labels = batch['labels']

                    logits = model(input_ids).permute(0, 2, 1)
                    loss = loss_fn(logits, tag_labels)
                    val_loss += loss.item()
                    val_batches += 1

                    preds = logits.argmax(dim=1)  # [batch, seq_len]
                    mask = tag_labels != -1
                    all_preds.extend(preds[mask].cpu().tolist())
                    all_labels.extend(tag_labels[mask].cpu().tolist())

            avg_val_loss = val_loss / val_batches
            precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
            recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
            f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            print(f"Val Loss: {avg_val_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

            scheduler.step(f1)

            if f1 > best_val_f1:
                best_val_f1 = f1
                torch.save(model.state_dict(), "blstm.pt")
                print(f"Saved best checkpoint (F1: {f1:.4f})")

def collate_fn(batch, pad_idx, label_pad_idx):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=label_pad_idx)
    
    return {
        "input_ids": padded_inputs, 
        "labels": padded_labels
    }


def main():
    config = TrainConfig()

    train_dataset = NERDataset(config.train_dir)
    dev_dataset = NERDataset(config.dev_dir, word2idx=train_dataset.word2idx, tag2idx=train_dataset.tag2idx)

    my_collate = partial(collate_fn, pad_idx=len(train_dataset.word2idx), label_pad_idx=-1)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)

    # update the config vocab
    config.vocab_size = len(train_dataset.word2idx) + 2 # additional for unk and unseen tokens 
    config.num_classes = len(train_dataset.tag2idx)

    model = blstm(
        vocab_size=config.vocab_size, 
        embed_size=config.embed_size, 
        hidden_size=config.hidden_size, 
        output_size=config.output_size, 
        num_classes=config.num_classes,
        num_layers=config.num_layers, 
        dropout=config.dropout
    )

    # check if we need to build with glove embeddings
    if config.use_glove:
        glove_path = "/Users/jerryli/Desktop/CSCI544/HW3 2/glove.6B.100d.gz"
        word2vec = load_glove(glove_path)
        print(f"Loaded {len(word2vec)} words, dim={len(next(iter(word2vec.values())))}")

        embedding = build_embedding(train_dataset.word2idx, word2vec, embed_dim=config.embed_size)
        print(f"Embedding shape: {embedding.weight.shape}")

        model.embed = embedding
    
    train_model(model, train_dataloader, dev_dataloader, config)

if __name__ == "__main__":
    main()