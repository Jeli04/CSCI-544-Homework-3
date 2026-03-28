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
        ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_size*2, output_size),     # we need the *2 for a blstm
            nn.ELU(),
            nn.Linear(output_size, num_classes)
        )
        
    def forward(self, x):
        x = self.embed(x)

        x, (h_n, c_n) = self.lstm(x)

        x = self.dropout(x) # dropout not applied when lstm layer = 1

        x = self.head(x)

        return x
    
if __name__ == "__main__":
    model = blstm(vocab_size=100, embed_size=100, hidden_size=256, output_size=128, num_classes=9, num_layers=1, dropout=0.33)



    