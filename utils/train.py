"""
Docstring for train
"""

from tqdm import tqdm
from .utils import get_device

DEVICE = get_device()


def train(model, dataloader, criterion, optimizer, epochs:int):
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