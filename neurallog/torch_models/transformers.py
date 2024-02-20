import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encodings import PositionEmbedding

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.transformer_layer_1 = nn.TransformerEncoderLayer(embed_dim, num_heads, ff_dim, rate, batch_first=True)
        self.transformer_layer_2 = nn.TransformerEncoderLayer(embed_dim, num_heads, ff_dim, rate, batch_first=True)

    def forward(self, inputs):
        out = self.transformer_layer_1(inputs)
        return self.transformer_layer_2(out)

class TransformerClassifier(nn.Module):
    def __init__(self, embed_dim, ff_dim, max_len, num_heads, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.embedding_layer = PositionEmbedding(max_len, embed_dim)
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(embed_dim, 32)
        self.dense2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer_block(x)
        x = self.global_avg_pooling(x.permute(0, 2, 1)).squeeze(2)
        x = self.dropout(x)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.softmax(self.dense2(x), dim=1)
        return x

# # Instantiate the model
# embed_dim = 768
# num_heads = 12
# ff_dim = 2048
# max_len = 20
# dropout = 0.1
# model = TransformerClassifier(embed_dim, ff_dim, max_len, num_heads, dropout)
