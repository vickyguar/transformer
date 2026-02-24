"""
Docstring for train
"""

from tqdm import tqdm
from typing import TYPE_CHECKING
from .utils import get_device

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    import torch

DEVICE = get_device()


def train(model: torch.nn.Module, dataloader: 'DataLoader', criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int) -> None:
    """
    Entrena el modelo Transformer.
    
    Args:
        model (torch.nn.Module): Modelo a entrenar.
        dataloader (DataLoader): DataLoader con datos.
        criterion: Función de pérdida.
        optimizer: Optimizador.
        epochs (int): Número de épocas.
    
    Returns:
        None
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