import torch
import torch.nn as nn
import math

class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=4,
                 dim_feedforward=2048, dropout=0.1,mask_token_idx=4423):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size-1)
        # self.sos_idx = sos_idx
        self.mask_token_idx = mask_token_idx
        
    def forward(self, src, tgt=None, max_len=None):
        # Embedding和位置编码
        src = self.token_embedding(src)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        output = self.output_layer(src)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)