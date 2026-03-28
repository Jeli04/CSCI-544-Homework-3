import torch
from torch.utils.data import Dataset

# tag column is optional (absent in test set)
class NERDataset(Dataset):
    NUM_CASE_TYPES = 5  # lower, upper, title, has_digit, other

    def __init__(self, file_path, word2idx={}, tag2idx={}, char2idx={}):
        self.ids = []
        self.labels = []
        self.words = []  # raw words per sentence for char-level encoding
        self.case_ids = []  # case type per word per sentence
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.char2idx = char2idx
        self.preload_vocab = False if len(word2idx) == 0 and len(tag2idx) == 0 else True
        self.max_word_len = 0
        self._load(file_path)

    @staticmethod
    def _get_case_id(word):
        if any(c.isdigit() for c in word):
            return 3
        elif word.islower():
            return 0
        elif word.isupper():
            return 1
        elif word[0].isupper():
            return 2
        else:
            return 4

    def _load(self, file_path):
        words, tags, raw_words, case_ids = [], [], [], []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if words:
                        self.ids.append(words)
                        self.labels.append(tags)
                        self.words.append(raw_words)
                        self.case_ids.append(case_ids)
                        words, tags, raw_words, case_ids = [], [], [], []
                else:
                    parts = line.split()
                    # columns: idx, word, [tag] (tag absent in test set)
                    word = parts[1]
                    tag = parts[2] if len(parts) > 2 else None

                    if not self.preload_vocab:
                        if word not in self.word2idx:
                            self.word2idx[word] = len(self.word2idx)
                        if tag is not None and tag not in self.tag2idx:
                            self.tag2idx[tag] = len(self.tag2idx)
                        for c in word:
                            if c not in self.char2idx:
                                self.char2idx[c] = len(self.char2idx) + 1  # reserve 0 for padding

                    self.max_word_len = max(self.max_word_len, len(word))
                    words.append(self.word2idx[word] if word in self.word2idx else len(self.word2idx) + 1)
                    tags.append(self.tag2idx.get(tag, -1) if tag is not None else -1)
                    raw_words.append(word)
                    case_ids.append(self._get_case_id(word))

            if words:
                self.ids.append(words)
                self.labels.append(tags)
                self.words.append(raw_words)
                self.case_ids.append(case_ids)

    def __len__(self):
        return len(self.ids)

    def _words_to_char_ids(self, words):
        """Convert a list of words to a (seq_len, max_word_len) tensor of char indices."""
        padded = []
        for word in words:
            ids = [self.char2idx.get(c, 0) for c in word]
            ids += [0] * (self.max_word_len - len(ids))
            padded.append(ids)
        return torch.tensor(padded)

    def __getitem__(self, idx):
        char_ids = self._words_to_char_ids(self.words[idx])
        return torch.tensor(self.ids[idx]), char_ids, torch.tensor(self.labels[idx]), torch.tensor(self.case_ids[idx])

if __name__ == "__main__":
    dataset = NERDataset("data/train")
    print(len(dataset))
    print(dataset[0])
    print(len(dataset.word2idx))    
    print(len(dataset.tag2idx))