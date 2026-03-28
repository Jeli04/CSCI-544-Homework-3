import torch
import torch.nn as nn

class blstm(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            hidden_size,
            output_size,
            num_classes,
            num_layers=1,
            dropout=0.1,
            case_embed_size=0,
            num_case_types=5,
        ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.case_embed_size = case_embed_size
        if case_embed_size > 0:
            self.case_embed = nn.Embedding(num_case_types, case_embed_size)

        lstm_input = embed_size + case_embed_size
        self.lstm = nn.LSTM(lstm_input, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_size*2, output_size),     # we need the *2 for a blstm
            nn.ELU(),
            nn.Linear(output_size, num_classes)
        )

    def forward(self, x, case_ids=None):
        x = self.embed(x)

        if self.case_embed_size > 0 and case_ids is not None:
            x = torch.cat([x, self.case_embed(case_ids)], dim=-1)

        x, (h_n, c_n) = self.lstm(x)

        x = self.dropout(x) # dropout not applied when lstm layer = 1

        x = self.head(x)

        return x
    
if __name__ == "__main__":
    model = blstm(vocab_size=100, embed_size=100, hidden_size=256, output_size=128, num_classes=9, num_layers=1, dropout=0.33)



    