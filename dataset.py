import torch
from torch.utils.data import Dataset

# note this is only for the train and dev splits
class NERDataset(Dataset):
    def __init__(self, file_path, word2idx={}, tag2idx={}):
        self.ids = []
        self.labels = []
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.preload_vocab = False if len(word2idx) == 0 and len(tag2idx) == 0 else True  
        self._load(file_path)

    def _load(self, file_path):
        words, tags = [], []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if words:
                        self.ids.append(words)
                        self.labels.append(tags)
                        words, tags = [], []
                else:
                    parts = line.split()
                    # columns: idx, word, tag
                    _, word, tag = parts[0], parts[1], parts[2]

                    if not self.preload_vocab:
                        if word not in self.word2idx:
                            self.word2idx[word] = len(self.word2idx)

                        if tag not in self.tag2idx:
                            self.tag2idx[tag] = len(self.tag2idx)

                    words.append(self.word2idx[word] if word in self.word2idx else len(self.word2idx) + 1) # else unk token
                    tags.append(self.tag2idx[tag] if tag in self.tag2idx else -1)

            if words:
                self.ids.append(words)
                self.labels.append(tags)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx]), torch.tensor(self.labels[idx])

if __name__ == "__main__":
    dataset = NERDataset("/Users/jerryli/Desktop/CSCI544/HW3 2/data/train")
    print(len(dataset))
    print(dataset[0])
    print(len(dataset.word2idx))    
    print(len(dataset.tag2idx))