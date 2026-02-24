import torch
from typing import Dict, List
from static.constants import MAX_SEQ_LEN
from utils.preprocess import preprocesar_enunciado
from utils.utils import get_device

DEVICE = get_device()

def sentence_to_indices(sentence: str, word2idx: Dict[str, int]) -> List[int]:
    """
    Convierte una oración en una lista de índices usando el vocabulario.
    
    Args:
        sentence (str): Oración de entrada.
        word2idx (Dict[str, int]): vocabulario palabra a índice.
    
    Returns:
        List[int]: Lista de índices.
    """
    return [word2idx.get(word, word2idx['<unk>']) for word in sentence.split()]

def indices_to_sentence(indices: List[int], idx2word: Dict[int, str]) -> str:
    """
    Convierte una lista de índices en una oración usando el vocabulario inverso.
    
    Args:
        indices (List[int]): Lista de índices.
        idx2word (Dict[int, str]): Vocabulario índice a palabra.
    
    Returns:
        str: Oración reconstruida.
    """
    return ' '.join([idx2word[idx] for idx in indices if idx in idx2word and idx2word[idx] != '<pad>'])

def translate_sentence(model: torch.nn.Module, sentence: str, spanish_palabra_idx: Dict[str, int], english_idx_palabra: Dict[int, str], english_palabra_idx: Dict[str, int], max_len: int = MAX_SEQ_LEN, device: torch.device = DEVICE) -> str:
    """
    Traduce una oración del español al inglés usando el modelo Transformer.
    
    Args:
        model (torch.nn.Module): Modelo entrenado.
        sentence (str): Oración en español.
        spanish_palabra_idx (Dict[str, int]): Vocabulario español.
        english_idx_palabra (Dict[int, str]): Vocabulario inglés inverso.
        english_palabra_idx (Dict[str, int]): Vocabulario inglés.
        max_len (int): Longitud máxima.
        device (torch.device): Dispositivo.
    
    Returns:
        str: Traducción en inglés.
    """
    model.eval()
    sentence = preprocesar_enunciado(sentence)
    input_indices = sentence_to_indices(sentence, spanish_palabra_idx)
    input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)

    # Initialize the target tensor with <sos> token
    tgt_indices = [english_palabra_idx['<sos>']]
    tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            output = model(input_tensor, tgt_tensor)
            output = output.squeeze(0)
            next_token = output.argmax(dim=-1)[-1].item()
            tgt_indices.append(next_token)
            tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)
            if next_token == english_palabra_idx['<eos>']:
                break

    return indices_to_sentence(tgt_indices, english_idx_palabra)

def evaluate_translations(model: torch.nn.Module, sentences: List[str], spanish_palabra_idx: Dict[str, int], english_idx_palabra: Dict[int, str], english_palabra_idx: Dict[str, int], max_len: int = MAX_SEQ_LEN, device: torch.device = DEVICE) -> None:
    """
    Evalúa traducciones para una lista de oraciones.
    
    Args:
        model (torch.nn.Module): Modelo.
        sentences (List[str]): Lista de oraciones.
        spanish_palabra_idx (Dict[str, int]): Vocabulario español.
        english_idx_palabra (Dict[int, str]): Vocabulario inglés inverso.
        english_palabra_idx (Dict[str, int]): Vocabulario inglés.
        max_len (int): Longitud máxima.
        device (torch.device): Dispositivo.
    
    Returns:
        None
    """
    for sentence in sentences:
        translation = translate_sentence(model, sentence, spanish_palabra_idx, english_idx_palabra, english_palabra_idx, max_len, device)
        print(f'Input sentence: {sentence}')
        print(f'Traducción: {translation}')
        print()