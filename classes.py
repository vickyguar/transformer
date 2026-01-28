import math
import torch
import torch.nn as nn
from constants import MAX_SEQ_LEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN):
        pass

    def forward(self, x):
        pass


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        pass

    def forward(self, x, mask=None): # el encoder tiene mascara porque no queremos atender a los tokens de padding
        pass


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        pass

    def forward(self, x, encoder_output, target_mask, encoder_mask=None):
        pass


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, input_vocab_size, target_vocab_size, max_len=MAX_SEQ_LEN, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(d_model, max_len)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.fc_out = nn.Linear(d_model, target_vocab_size)

    def forward(self, source, target):
        source_mask, target_mask = self.mask(source, target)
        
        # encoder
        source = self.encoder_embedding(source) * math.sqrt(self.encoder_embedding.embedding_dim)
        source = self.pos_embedding(source)
        encoder_output = self.encoder(source, source_mask)

        # decoder
        target = self.decoder_embedding(target) * math.sqrt(self.decoder_embedding.embedding_dim)
        target = self.pos_embedding(target)
        decoder_output = self.decoder(target, encoder_output, target_mask, source_mask)
        output = self.fc_out(decoder_output)
        return output # logits !!!

    def mask(self, source, target):
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2) # agrego dimensionalidad
        target_mask = (target != 0).unsqueeze(1).unsqueeze(2)
        size = target.size(1)
        no_mask = torch.tril(torch.ones((1, size, size), device=device)).bool()
        target_mask = target_mask & no_mask
        return source_mask, target_mask