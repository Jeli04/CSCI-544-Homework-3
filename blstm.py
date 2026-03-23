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
            embeddings=None
        ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) if embeddings is None else embeddings

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, output_size),  
            nn.ELU(),
            nn.Linear(output_size, num_classes)
        )
        
    def forward(self, x):
        x = self.embed(x)

        x, (h_n, c_n) = self.lstm(x)

        x = self.head(x)

        return x
    
if __name__ == "__main__":
    model = blstm(vocab_size=100, embed_size=100, hidden_size=256, output_size=128, num_classes=9, num_layers=1, dropout=0.33)



    