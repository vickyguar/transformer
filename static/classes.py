import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from static.constants import MAX_SEQ_LEN
from utils.utils import get_device
from typing import Tuple

DEVICE = get_device()

#region POSICIONAL EMBEDDING
# ! REVISAR
class PositionalEmbedding(nn.Module):
    """
    Clase para agregar embeddings posicionales a las secuencias de entrada.
    
    Args:
        d_model (int): Dimensión del modelo.
        max_len (int): Longitud máxima de secuencia.
    """
    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer('pos_ebed_matrix', self._create_positional_matrix(d_model, max_len))
        
    def _create_positional_matrix(self, d_model: int, max_len: int) -> torch.Tensor:
        """
        Crea la matriz de embeddings posicionales.
        
        Args:
            d_model (int): Dimensión del modelo.
            max_len (int): Longitud máxima.
        
        Returns:
            torch.Tensor: Matriz de posiciones.
        """
        pos_ebed_matrix = torch.zeros(max_len, d_model, device=DEVICE)
        token_pos = torch.arange(0, max_len, dtype=torch.float, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=DEVICE).float() * (-math.log(10000.0) / d_model))
        pos_ebed_matrix[:, 0::2] = torch.sin(token_pos * div_term) # elementos pares
        if d_model % 2 == 1:  # si d_model es impar, handle the last dimension
            pos_ebed_matrix[:, 1::2] = torch.cos(token_pos * div_term[:, :-1])
        else:
            pos_ebed_matrix[:, 1::2] = torch.cos(token_pos * div_term) # elementos impares
        return pos_ebed_matrix
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Agrega embeddings posicionales a la entrada.
        
        Args:
            x (torch.Tensor): Tensor de entrada.
        
        Returns:
            torch.Tensor: Tensor con posiciones agregadas.
        """
        seq_len = x.size(1)
        return x + self.pos_ebed_matrix[:seq_len, :x.size(2)] 
#endregion

#region MULTIHEAD ATTENTION
class MultiheadAttention(nn.Module):
    """
    Implementación de atención multi-cabeza.
    
    Args:
        d_model (int): Dimensión del modelo.
        num_heads (int): Número de cabezas.
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        self.d_v = d_model // num_heads
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        # Matrices q, k, v
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.linear_out = nn.Linear(d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula la atención multi-cabeza.
        
        Args:
            Q (torch.Tensor): Query.
            K (torch.Tensor): Key.
            V (torch.Tensor): Value.
            mask (torch.Tensor, optional): Máscara.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Valores ponderados y atención.
        """
        batch_size = Q.size(0)
        Q=self.w_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2) # (batch_size, num_heads, seq_len, d_k)
        K=self.w_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2) # (batch_size, num_heads, seq_len, d_k)
        V=self.w_v(V).view(batch_size, -1, self.num_heads, self.d_v).transpose(1,2) # (batch_size, num_heads, seq_len, d_v)
        weighted_values, attention = self.scale_dot_product(Q,K,V,mask)
        
        weighted_values = weighted_values.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        weighted_values = self.linear_out(weighted_values)
        
        return weighted_values, attention

    def scale_dot_product(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Producto punto escalado para atención.
        
        Args:
            Q, K, V (torch.Tensor): Queries, Keys, Values.
            mask (torch.Tensor, optional): Máscara.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Valores y atención.
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Cuando sea softmax den valores de cero. Matriz triangular baja
        attention = F.softmax(scores, dim=-1)
        weighted_values = torch.matmul(attention, V) # Con la V de values
        
        return weighted_values, attention
#endregion

#region ENCODER
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # parte medular del transformer 
        # TODO probar con nn.MultiheadAttention
        self.self_attn = MultiheadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        

    def forward(self, x, mask=None):
        # Self-attention
        attention_score, _ = self.self_attn(x, x, x, mask) # tres veces x
        x = x + self.dropout1(attention_score)
        x = self.norm1(x)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1): # el encoder se repite N veces, en el paper 6 veces
        super().__init__()
        # self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout) for _ in range(num_layers)])
        # TODO Probar con eso
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None): # el encoder tiene mascara porque no queremos atender a los tokens de padding
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
#endregion

#region DECODER
class PositionFeedForward(nn.Module):
    """
    Red feed-forward posicional.
    
    Args:
        d_model (int): Dimensión de entrada/salida.
        d_ff (int): Dimensión interna.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Entrada.
        
        Returns:
            torch.Tensor: Salida.
        """
        return self.linear2(F.relu(self.linear1(x)))


class DecoderLayer(nn.Module):
    """
    Capa del decoder del Transformer.
    
    Args:
        d_model (int): Dimensión.
        num_heads (int): Cabezas.
        d_ff (int): Feed-forward.
        dropout (float): Dropout.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads)
        self.cross_attn = MultiheadAttention(d_model, num_heads) # atención cruzada pq mira también al encoder
        self.feed_foward = PositionFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, target_mask: torch.Tensor, encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass de la capa del decoder.
        
        Args:
            x (torch.Tensor): Entrada del decoder.
            encoder_output (torch.Tensor): Salida del encoder.
            target_mask (torch.Tensor): Máscara del target.
            encoder_mask (torch.Tensor, optional): Máscara del encoder.
        
        Returns:
            torch.Tensor: Salida.
        """
        attention_score, _ = self.self_attn(x, x, x, target_mask)
        x = x + self.dropout1(attention_score)
        x = self.norm1(x)

        encoder_attn, _ = self.cross_attn(x, encoder_output, encoder_output, encoder_mask)
        x = x + self.dropout2(encoder_attn)
        x = self.norm2(x)

        output = self.feed_foward(x)
        x = x + self.dropout3(output)
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    """
    Decoder completo del Transformer.
    
    Args:
        d_model (int): Dimensión.
        num_heads (int): Cabezas.
        d_ff (int): Feed-forward.
        num_layers (int): Capas.
        dropout (float): Dropout.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, target_mask: torch.Tensor, encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass del decoder.
        
        Args:
            x (torch.Tensor): Entrada.
            encoder_output (torch.Tensor): Salida del encoder.
            target_mask (torch.Tensor): Máscara del target.
            encoder_mask (torch.Tensor, optional): Máscara del encoder.
        
        Returns:
            torch.Tensor: Salida del decoder.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, target_mask, encoder_mask)
        return self.norm(x)
#endregion

class Transformer(nn.Module):
    """
    Modelo Transformer completo para traducción seq2seq.
    
    Args:
        d_model (int): Dimensión del modelo.
        num_heads (int): Número de cabezas de atención.
        d_ff (int): Dimensión de feed-forward.
        num_layers (int): Número de capas en encoder/decoder.
        input_vocab_size (int): Tamaño del vocabulario de entrada.
        target_vocab_size (int): Tamaño del vocabulario de salida.
        max_len (int): Longitud máxima de secuencia.
        dropout (float): Probabilidad de dropout.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, input_vocab_size: int, target_vocab_size: int, max_len: int = MAX_SEQ_LEN, dropout: float = 0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(d_model, max_len)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.fc_out = nn.Linear(d_model, target_vocab_size)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del Transformer.
        
        Args:
            source (torch.Tensor): Secuencia de entrada.
            target (torch.Tensor): Secuencia de salida.
        
        Returns:
            torch.Tensor: Logits de salida.
        """
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

    def mask(self, source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Crea máscaras para encoder y decoder.
        
        Args:
            source (torch.Tensor): Fuente.
            target (torch.Tensor): Target.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Máscaras.
        """
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2) # agrego dimensionalidad
        target_mask = (target != 0).unsqueeze(1).unsqueeze(2)
        size = target.size(1)
        no_mask = torch.tril(torch.ones((1, size, size), device=DEVICE)).bool()
        target_mask = target_mask & no_mask
        return source_mask, target_mask 