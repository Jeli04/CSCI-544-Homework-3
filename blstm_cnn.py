import torch
import torch.nn as nn

class blstm_cnn(nn.Module):
    def __init__(
            self,
            char_vocab_size: int,
            char_embed_size: int,
            num_filters: int,
            kernel_size: int,
            word_vocab_size: int,
            word_embed_size: int,
            embed_size: int,
            hidden_size: int,
            output_size: int,
            num_classes: int,
            num_layers: int = 1,
            dropout: float = 0.1,
            padding_idx: int = 0,
            num_cnn_layers: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_size, padding_idx=padding_idx)
        self.word_embedding = nn.Embedding(word_vocab_size, word_embed_size)

        cnn_layers = []
        in_channels = char_embed_size
        for _ in range(num_cnn_layers):
            cnn_layers.append(nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size // 2))
            cnn_layers.append(nn.ReLU())
            in_channels = num_filters
        self.char_dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(*cnn_layers)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_size*2, output_size),
            nn.ELU(),
            nn.Linear(output_size, num_classes)
        )

        self.START_TAG = num_classes
        self.STOP_TAG = num_classes + 1
        self.total_tags = num_classes + 2
        self.transitions = nn.Parameter(torch.randn(self.total_tags, self.total_tags))
        self.transitions.data[self.START_TAG, :] = -10000.0
        self.transitions.data[:, self.STOP_TAG] = -10000.0

    def forward(self, word_ids, char_ids, mask):
        B, L, W = char_ids.shape
        char_ids = char_ids.reshape(B*L, W)

        char_embeds = self.char_embedding(char_ids)
        char_embeds = self.char_dropout(char_embeds)
        char_embeds = char_embeds.permute(0, 2, 1)
        char_x = self.conv(char_embeds) # (B*L, num_filters, T)
        char_x = torch.max(char_x, dim=2).values  # (B*L, num_filters)
        char_x = char_x.view(B, L, -1)  # (B, L, num_filters)

        word_embeds = self.word_embedding(word_ids)
        x = torch.cat([char_x, word_embeds], dim=-1)

        x, (h_n, c_n) = self.lstm(x)
        x = self.dropout(x)
        x = self.head(x)  # (B, L, num_classes)
        return x

    def crf_loss(self, emissions, tags, mask):
        B, L, _ = emissions.shape
        device = emissions.device
        safe_tags = tags.clamp(min=0)

        gold_score = (
            self.transitions[safe_tags[:, 0], self.START_TAG]
            + emissions[:, 0].gather(1, safe_tags[:, 0:1]).squeeze(1)
        )

        for t in range(1, L):
            mt = mask[:, t].float()
            trans = self.transitions[safe_tags[:, t], safe_tags[:, t-1]]
            emit = emissions[:, t].gather(1, safe_tags[:, t:t+1]).squeeze(1)
            gold_score = gold_score + (trans + emit) * mt

        lengths = mask.long().sum(dim=1) - 1
        last_tags = safe_tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        gold_score = gold_score + self.transitions[self.STOP_TAG, last_tags]

        alpha = torch.full((B, self.total_tags), -10000.0, device=device)
        alpha[:, self.START_TAG] = 0.0

        for t in range(L):
            mt = mask[:, t].float().unsqueeze(1)
            emit_padded = torch.cat([emissions[:, t], torch.zeros(B, 2, device=device)], dim=1)
            next_alpha = torch.logsumexp(
                alpha.unsqueeze(1) + self.transitions.unsqueeze(0), dim=2
            ) + emit_padded
            alpha = next_alpha * mt + alpha * (1.0 - mt)

        partition = torch.logsumexp(alpha + self.transitions[self.STOP_TAG].unsqueeze(0), dim=1)
        return (partition - gold_score).mean()

    # viterbi decoding
    def decode(self, emissions, mask):
        B, L, _ = emissions.shape
        device = emissions.device

        score = torch.full((B, self.total_tags), -10000.0, device=device)
        score[:, self.START_TAG] = 0.0
        backpointers = []

        for t in range(L):
            emit_t = emissions[:, t]
            mask_t = mask[:, t].unsqueeze(1)

            next_score = torch.full((B, self.total_tags), -10000.0, device=device)
            backptr_t = torch.zeros((B, self.total_tags), dtype=torch.long, device=device)

            for next_tag in range(self.num_classes):
                candidates = score + self.transitions[next_tag].unsqueeze(0)
                best_score, best_tag = candidates.max(dim=1)
                next_score[:, next_tag] = best_score + emit_t[:, next_tag]
                backptr_t[:, next_tag] = best_tag

            score = torch.where(mask_t, next_score, score)
            backpointers.append(backptr_t)

        score = score + self.transitions[self.STOP_TAG].unsqueeze(0)
        _, best_last_tag = score.max(dim=1)

        lengths = mask.long().sum(dim=1)
        best_paths = []
        for b in range(B):
            seq_len = lengths[b].item()
            best_tag = best_last_tag[b].item()
            path = []
            for t in range(seq_len - 1, -1, -1):
                path.append(best_tag)
                best_tag = backpointers[t][b, best_tag].item()
            path.reverse()
            path = [tag for tag in path if tag < self.num_classes]
            best_paths.append(path)

        return best_paths


if __name__ == "__main__":
    char_vocab_size = 50
    word_vocab_size = 100
    char_embed_size = 30
    num_filters = 50
    kernel_size = 8
    word_embed_size = 50
    embed_size = num_filters + word_embed_size
    hidden_size = 256
    output_size = 128
    num_classes = 9

    model = blstm_cnn(
        char_vocab_size=char_vocab_size,
        char_embed_size=char_embed_size,
        num_filters=num_filters,
        kernel_size=kernel_size,
        word_vocab_size=word_vocab_size,
        word_embed_size=word_embed_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_classes=num_classes,
        num_layers=1,
        dropout=0.33,
        num_cnn_layers=1,
    )

    B, L, W = 2, 10, 8
    word_ids = torch.randint(0, word_vocab_size, (B, L))
    char_ids = torch.randint(0, char_vocab_size, (B, L, W))
    mask = torch.ones(B, L, dtype=torch.bool)
    tags = torch.randint(0, num_classes, (B, L))

    emissions = model(word_ids, char_ids, mask)
    print(f"Emissions shape: {emissions.shape}")  # (2, 10, 9)

    loss = model.crf_loss(emissions, tags, mask)
    print(f"CRF loss: {loss.item():.4f}")

    paths = model.decode(emissions, mask)
    print(f"Decoded path lengths: {[len(p) for p in paths]}")  # [10, 10]
