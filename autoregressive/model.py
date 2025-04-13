import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ARModel(nn.Module):
    def __init__(self, vocab_size,travel_location, d_model=512, nhead=8,
                 num_decoder_layers=4, dim_feedforward=2048, dropout=0.1,max_len=24):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(travel_location, d_model)

        self.condition_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer encoder和decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)

        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, num_decoder_layers)

        self.output_layer = nn.Linear(d_model,travel_location)

        self.sos_idx=0
        self.max_len=max_len
        self.time_embedding=TemporalEmbedding(d_model,length=self.max_len)
    def forward(self,condition,tgt=None,batch_size=512):
        condition=condition.long()

        time_index=torch.arange(self.max_len).unsqueeze(0).repeat(batch_size,1).to(condition.device)
        time_embedding=self.time_embedding(time_index)
        if self.training and tgt is not None:
            condition_embedding = self.condition_embedding(condition).unsqueeze(1)
           
            tgt=tgt.long()
        
            tgt = self.token_embedding(tgt)+time_embedding
            tgt = torch.cat([condition_embedding,tgt],dim=1)
            tgt = self.pos_encoder(tgt)
            tgt = tgt[:, :-1, :]
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            tgt=self.transformer_decoder(tgt,is_causal=True,mask=tgt_mask)

            output = self.output_layer(tgt)
            return output
        
        # 推理模式：从起始符开始自回归生成
        else:
            # 如果没有指定max_len，则使用输入序列长度
            outputs = []
            condition_embedding = self.condition_embedding(condition).unsqueeze(1)
            for i in range(self.max_len): 
                if i==0:
                    ys=condition.unsqueeze(1)
                    tgt=condition_embedding
                else:
                    tgt1 = self.token_embedding(ys[:,1:])+time_embedding[:,:i]
                    tgt = torch.cat([condition_embedding,tgt1],dim=1)
                tgt = self.pos_encoder(tgt)
                tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
                
                output = self.transformer_decoder(tgt,is_causal=True,mask=tgt_mask)
                logits = self.output_layer(output)
                outputs.append(logits[:, -1:, :])  # 只取最后一个时间步的logits
                
                # 获取预测的下一个token
                next_token = torch.multinomial(torch.softmax(logits[:, -1, :], dim=-1),1)
                if i>0:
                    max_token=torch.max(ys[:,1:],dim=1)[0]+1
                    max_token=max_token.unsqueeze(1)
                    if len(torch.where(next_token>max_token)[0])>0:
                        next_token[torch.where(next_token>max_token)[0]]=max_token[torch.where(next_token>max_token)[0]]
                ys = torch.cat([ys, next_token], dim=1)
            
            outputs = torch.cat(outputs, dim=1)
            return outputs,ys[:,1:]
    
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
    

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model,length=24):
        super(TemporalEmbedding, self).__init__()


        hour_size = 24
        weekday_size = 7
        self.length=length
        self.hour_embed = nn.Embedding(hour_size, d_model)
        if length==168:
            self.weekday_embed = nn.Embedding(weekday_size, d_model)


    def forward(self, time_index):
        time_index = time_index.long()
        hour_index= time_index//7

        weekday_index=time_index//24
        weekday_index=weekday_index.long()
        
        if self.length==168:
            hour_x = self.hour_embed(hour_index)
            weekday_x = self.weekday_embed(weekday_index)
            return hour_x + weekday_x
        else:
            hour_x = self.hour_embed(hour_index)
            return hour_x