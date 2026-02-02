import torch
from torch.utils.data import Dataset

class Traductor(Dataset):
    def __init__(self, spanish, english, spanish_palabra_idx, english_palabra_idx, transform=None):
        self.spanish = spanish
        self.english = english
        self.spanish_palabra_idx = spanish_palabra_idx
        self.english_palabra_idx = english_palabra_idx

    def __len__(self):
        return len(self.spanish_enunciado) # también puede ser english

    def __getitem__(self, idx):
        english_sentence = self.english[idx]
        spanish_sentence = self.spanish[idx]
        
        # Los tokens:
        english_tokens = [self.english_palabra_idx.get(palabra, self.english_palabra_idx['<unk>']) for palabra in english_sentence.split()]
        spanish_tokens = [self.spanish_palabra_idx.get(palabra, self.spanish_palabra_idx['<unk>']) for palabra in spanish_sentence.split()]
        
        return torch.tensor(english_tokens), torch.tensor(spanish_tokens)
