"""
! !!! TODO los docstrings
"""

import torch
from static.constants import MAX_SEQ_LEN
from utils.preprocess import preprocesar_enunciado
from utils.utils import get_device

DEVICE = get_device()

def sentence_to_indices(sentence, word2idx):
    return [word2idx.get(word, word2idx['<unk>']) for word in sentence.split()]

def indices_to_sentence(indices, idx2word):
    return ' '.join([idx2word[idx] for idx in indices if idx in idx2word and idx2word[idx] != '<pad>'])

def translate_sentence(model, sentence, spanish_palabra_idx, english_idx_palabra, english_palabra_idx, max_len=MAX_SEQ_LEN, device=DEVICE):
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

def evaluate_translations(model, sentences, spanish_palabra_idx, english_idx_palabra, english_palabra_idx, max_len=MAX_SEQ_LEN, device=DEVICE):
    for sentence in sentences:
        translation = translate_sentence(model, sentence, spanish_palabra_idx, english_idx_palabra, english_palabra_idx, max_len, device)
        print(f'Input sentence: {sentence}')
        print(f'Traducción: {translation}')
        print()