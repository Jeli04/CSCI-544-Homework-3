import torch
import json
import argparse
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from sklearn.metrics import f1_score

from blstm import blstm
from blstm_cnn import blstm_cnn
from dataset import NERDataset

LABEL_PAD = -1


def collate_fn(batch, pad_idx, label_pad_idx):
    inputs = [item[0] for item in batch]
    labels = [item[2] for item in batch]
    case_ids = [item[3] for item in batch]

    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=label_pad_idx)
    padded_case_ids = pad_sequence(case_ids, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "case_ids": padded_case_ids,
    }


def collate_fn_cnn(batch, pad_idx, label_pad_idx):
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


def run_inference(model, dataset, collate, device, output_path, pad_idx, eval_f1=False, use_crf=False):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate)
    idx2tag = {v: k for k, v in dataset.tag2idx.items()}

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            tag_labels = batch["labels"].to(device)

            if use_crf:
                char_ids = batch["char_ids"].to(device)
                case_ids = batch["case_ids"].to(device)
                mask = tag_labels != -1
                emissions = model(input_ids, char_ids, mask, case_ids=case_ids)
                pred_paths = model.decode(emissions, mask)
                for preds_seq, labels_row, m in zip(pred_paths, tag_labels, mask):
                    all_preds.append(preds_seq)
                    if eval_f1:
                        all_labels.extend(labels_row[m].cpu().tolist())
            else:
                case_ids = batch["case_ids"].to(device) if "case_ids" in batch else None
                logits = model(input_ids, case_ids=case_ids).permute(0, 2, 1)  # [B, num_classes, L]
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
    parser.add_argument("--model", type=str, default="blstm", choices=["blstm", "blstm_cnn"], help="Model type")
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
        
        # blstm_cnn specific
        char_vocab_size: int = 100
        char_embed_size: int = 30
        num_filters: int = 50
        kernel_size: int = 3
        word_embed_size: int = 100
        num_cnn_layers: int = 1
        case_embed_size: int = 0

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
    use_cnn = args.model == "blstm_cnn"

    train_dataset = NERDataset(config.train_dir)
    dev_dataset = NERDataset(config.dev_dir, word2idx=train_dataset.word2idx, tag2idx=train_dataset.tag2idx, char2idx=train_dataset.char2idx)
    test_dataset = NERDataset(config.test_dir, word2idx=train_dataset.word2idx, tag2idx=train_dataset.tag2idx, char2idx=train_dataset.char2idx)

    config.num_classes = len(train_dataset.tag2idx)
    pad_idx = len(train_dataset.word2idx)

    if use_cnn:
        config.char_vocab_size = len(train_dataset.char2idx) + 1
        config.vocab_size = len(train_dataset.word2idx) + 2

        model = blstm_cnn(
            char_vocab_size=config.char_vocab_size,
            char_embed_size=config.char_embed_size,
            num_filters=config.num_filters,
            kernel_size=config.kernel_size,
            word_vocab_size=config.vocab_size,
            word_embed_size=config.word_embed_size,
            embed_size=config.num_filters + config.word_embed_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_classes=config.num_classes,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_cnn_layers=config.num_cnn_layers,
            case_embed_size=config.case_embed_size,
        )
        my_collate = partial(collate_fn_cnn, pad_idx=pad_idx, label_pad_idx=LABEL_PAD)
    else:
        config.vocab_size = len(train_dataset.word2idx) + 2

        model = blstm(
            vocab_size=config.vocab_size,
            embed_size=config.embed_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_classes=config.num_classes,
            num_layers=config.num_layers,
            dropout=config.dropout,
            case_embed_size=config.case_embed_size,
        )
        my_collate = partial(collate_fn, pad_idx=pad_idx, label_pad_idx=LABEL_PAD)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    run_inference(model, dev_dataset, my_collate, device, "outputs/dev.out", pad_idx, eval_f1=True, use_crf=use_cnn)
    run_inference(model, test_dataset, my_collate, device, "outputs/test.out", pad_idx, use_crf=use_cnn)


if __name__ == "__main__":
    main()
