import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from static.constants import MAX_SEQ_LEN


class Traductor(Dataset):
    """
    Dataset para traducción español-inglés.
    
    Args:
        spanish (List[str]): lista de oraciones en español.
        english (List[str]): lista de oraciones en inglés.
        spanish_palabra_idx (Dict[str, int]): vocabulario español.
        english_palabra_idx (Dict[str, int]): vocabulario inglés.
        transform: transformación opcional.
    """
    def __init__(self, spanish: List[str], english: List[str], spanish_palabra_idx: Dict[str, int], english_palabra_idx: Dict[str, int], transform=None):
        self.spanish = spanish
        self.english = english
        self.spanish_palabra_idx = spanish_palabra_idx
        self.english_palabra_idx = english_palabra_idx

    def __len__(self) -> int:
        return len(self.spanish) # o english

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        english_sentence = self.english[idx]
        spanish_sentence = self.spanish[idx]
        
        # Los tokens:
        english_tokens = [self.english_palabra_idx.get(palabra, self.english_palabra_idx['<unk>']) for palabra in english_sentence.split()]
        spanish_tokens = [self.spanish_palabra_idx.get(palabra, self.spanish_palabra_idx['<unk>']) for palabra in spanish_sentence.split()]
        
        return torch.tensor(english_tokens), torch.tensor(spanish_tokens)


# TODO: estudiar qué es una collate_fn (definición y usos)
def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Función collate para el DataLoader.
    
    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): Batch de datos.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batch padded.
    """
    english_batch, spanish_batch = zip(*batch) # descomprime la lista de tuplas
    
    english_batch = [seq[:MAX_SEQ_LEN].clone().detach() for seq in english_batch] # el detach para que no calcule ningun gradiente
    spanish_batch = [seq[:MAX_SEQ_LEN].clone().detach() for seq in spanish_batch]
    
    # Agrego pading
    padded_english_tokens = torch.nn.utils.rnn.pad_sequence(english_batch, batch_first=True, padding_value=0)
    padded_spanish_tokens = torch.nn.utils.rnn.pad_sequence(spanish_batch, batch_first=True, padding_value=0)
    
    return padded_english_tokens, padded_spanish_tokens