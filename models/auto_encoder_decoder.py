import torch
import torch.nn as nn
import math

class AutoEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=4,
                      mask_token_idx=4424,
                     dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.output_layer = nn.Linear(d_model, vocab_size-1)

        self.mask_token_idx = mask_token_idx
        

    def get_latents_from_tokens(self, src):
   
        src = self.token_embedding(src)

        return src
    def get_logits_from_latents(self, memory,tgt=None,if_sampling=False):

        embedding_weights = self.token_embedding.weight[:-1]  
        
    
        memory_norm = memory / memory.norm(dim=-1, keepdim=True)
        embedding_weights_norm = embedding_weights / embedding_weights.norm(dim=-1, keepdim=True)
        logits = torch.matmul(memory_norm, embedding_weights_norm.t())  # [batch_size, seq_len, vocab_size-2]

        token_probs = torch.softmax(logits, dim=-1).argmax(dim=-1)
        

        
        return token_probs

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