import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from constants import DEVICE, MAX_SEQ_LEN


class Traductor(Dataset):
    def __init__(self, spanish, english, spanish_palabra_idx, english_palabra_idx, transform=None):
        self.spanish = spanish
        self.english = english
        self.spanish_palabra_idx = spanish_palabra_idx
        self.english_palabra_idx = english_palabra_idx

    def __len__(self):
        return len(self.spanish) # también puede ser english

    def __getitem__(self, idx):
        english_sentence = self.english[idx]
        spanish_sentence = self.spanish[idx]
        
        # Los tokens:
        english_tokens = [self.english_palabra_idx.get(palabra, self.english_palabra_idx['<unk>']) for palabra in english_sentence.split()]
        spanish_tokens = [self.spanish_palabra_idx.get(palabra, self.spanish_palabra_idx['<unk>']) for palabra in spanish_sentence.split()]
        
        return torch.tensor(english_tokens), torch.tensor(spanish_tokens)


# TODO: estudiar qué es una collate_fn (definición y usos)
def collate_fn(batch):
    """
    Se aplica a cada batch en el DataLoader
    """
    english_batch, spanish_batch = zip(*batch) # descomprime la lista de tuplas
    
    english_batch = [seq[:MAX_SEQ_LEN].clone().detach() for seq in english_batch] # el detach para que no calcule ningun gradiente
    spanish_batch = [seq[:MAX_SEQ_LEN].clone().detach() for seq in spanish_batch]
    
    # Agrego pading
    padded_english_tokens = torch.nn.utils.rnn.pad_sequence(english_batch, batch_first=True, padding_value=0)
    padded_spanish_tokens = torch.nn.utils.rnn.pad_sequence(spanish_batch, batch_first=True, padding_value=0)
    
    return padded_english_tokens, padded_spanish_tokens
        

def train(model, dataloader, criterion, optimizer, epochs):
    """
    Recordar que recibe un enunciado input (español) y un target (en inglés).
    El decoder recibe la traducción, el resultado en inglés.
    """
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        # tqdm sobre el dataloader
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False # TODO: se puede probar con este parámetro en True también
        )
        
        for batch_idx, (english_batch, spanish_batch) in enumerate(progress_bar):
            english_batch = english_batch.to(DEVICE)
            spanish_batch = spanish_batch.to(DEVICE)
            
            # Preprocesamiento para decoder
            target_input = english_batch[:, :-1]
            target_output = english_batch[:, 1:].contiguous().view(-1)
            
            optimizer.zero_grad()
            
            output = model(spanish_batch, target_input)
            output = output.view(-1, output.size(-1))
            
            loss = criterion(output, target_output)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Actualizar tqdm con la loss promedio hasta ahora
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
