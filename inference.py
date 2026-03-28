import torch
import json
import argparse
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from sklearn.metrics import f1_score

from blstm import blstm
from dataset import NERDataset

LABEL_PAD = -1


def collate_fn(batch, pad_idx, label_pad_idx):
    inputs = [item[0] for item in batch]
    labels = [item[2] for item in batch]  # item[1] is char_ids

    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=label_pad_idx)

    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
    }


def run_inference(model, dataset, collate, device, output_path, pad_idx, eval_f1=False):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate)
    idx2tag = {v: k for k, v in dataset.tag2idx.items()}

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            tag_labels = batch["labels"].to(device)

            logits = model(input_ids).permute(0, 2, 1)  # [B, num_classes, L]
            preds = logits.argmax(dim=1)  # [B, L]

            for i in range(preds.size(0)):
                mask = input_ids[i] != pad_idx
                all_preds.append(preds[i][mask].cpu().tolist())
                if eval_f1:
                    all_labels.extend(tag_labels[i][mask].cpu().tolist())

    # write output
    with open(output_path, "w") as f:
        for sent_words, sent_preds in zip(dataset.words, all_preds):
            for j, (word, pred) in enumerate(zip(sent_words, sent_preds), start=1):
                tag = idx2tag.get(pred, "O")
                f.write(f"{j} {word} {tag}\n")
            f.write("\n")

    print(f"Wrote {len(all_preds)} sentences to {output_path}")

    if eval_f1:
        flat_preds = [p for seq in all_preds for p in seq]
        f1 = f1_score(all_labels, flat_preds, average="macro", zero_division=0)
        print(f"Dev F1: {f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load config
    from dataclasses import dataclass

    @dataclass
    class InferenceConfig:
        train_dir: str = "data/train"
        dev_dir: str = "data/dev"
        test_dir: str = "data/test"
        vocab_size: int = 100
        embed_size: int = 100
        hidden_size: int = 256
        output_size: int = 128
        num_classes: int = 9
        num_layers: int = 1
        dropout: float = 0.33

    if args.config:
        with open(args.config, "r") as f:
            raw = json.load(f)
        flat = {}
        for v in raw.values():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[v] = v
        config = InferenceConfig(**{k: v for k, v in flat.items() if hasattr(InferenceConfig, k)})
    else:
        config = InferenceConfig()

    # build vocab from train so indices match what the model was trained on
    train_dataset = NERDataset(config.train_dir)
    dev_dataset = NERDataset(config.dev_dir, word2idx=train_dataset.word2idx, tag2idx=train_dataset.tag2idx)
    test_dataset = NERDataset(config.test_dir, word2idx=train_dataset.word2idx, tag2idx=train_dataset.tag2idx)

    config.vocab_size = len(train_dataset.word2idx) + 2
    config.num_classes = len(train_dataset.tag2idx)

    model = blstm(
        vocab_size=config.vocab_size,
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        num_classes=config.num_classes,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    pad_idx = len(train_dataset.word2idx)
    my_collate = partial(collate_fn, pad_idx=pad_idx, label_pad_idx=LABEL_PAD)

    run_inference(model, dev_dataset, my_collate, device, "outputs/dev.out", pad_idx, eval_f1=True)
    run_inference(model, test_dataset, my_collate, device, "outputs/test.out", pad_idx)


if __name__ == "__main__":
    main()
