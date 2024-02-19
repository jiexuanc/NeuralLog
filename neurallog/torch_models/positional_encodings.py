import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.register_buffer('pos_encoding', self.positional_encoding(max_len, embed_dim))

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(torch.arange(position).unsqueeze(1),
                                     torch.arange(d_model).unsqueeze(0),
                                     d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads.unsqueeze(0)

        return pos_encoding.float()

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        return x