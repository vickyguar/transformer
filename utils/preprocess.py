"""
# ! Docstring 
"""

from collections import Counter

import pandas as pd


# TODO: add descriptions to docstrings
def open_df(csv_path: str) -> pd.DataFrame:
    """
    Docstring for open_df
    
    :param csv_path: Description
    :type csv_path: str
    :return: Description
    :rtype: DataFrame
    """
    
    df = pd.read_csv(csv_path, sep="\t", header=None, encoding="utf-8")
    df = df.drop(columns=[0, 2])
    df = df.rename(columns={1: "spanish", 3: "english"})
    return df


# TODO: add descriptions to docstrings
# ! probar quitando todos los simbolos, tal vez un regex? az a AZ?
def preprocesar_enunciado(enunciado: str) -> str:
    """
    Docstring for preprocesar_enunciado
    
    :param enunciado: Description
    :type enunciado: str
    :return: Description
    :rtype: str
    """
    enunciado = str(enunciado)
    enunciado = enunciado.lower().strip()
    # enunciado = enunciado.replace(".", " .")
    # enunciado = enunciado.replace(",", " ,")
    # enunciado = enunciado.replace("¿", " ¿ ")
    # enunciado = enunciado.replace("?", " ? ")
    # enunciado = enunciado.replace("¡", " ¡ ")
    # enunciado = enunciado.replace("!", " ! ")
    enunciado = '<sos> ' + enunciado + ' <eos>'
    return enunciado

# TODO: add descriptions to docstrings
def construir_palabras(enunciados:list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Docstring for construir_palabras
    
    :param enunciados: Description
    :type enunciados: list[str]
    :return: Description
    :rtype: tuple[dict[str, int], dict[int, str]]
    """
    palabras = [palabra for enunciado in enunciados for palabra in enunciado.split(" ")]
    palabras_count = Counter(palabras)
    palabras_ordenadas = sorted(palabras_count.items(), key=lambda x: x[1], reverse=True)
    palabra_idx = {palabra: idx for idx, (palabra, _) in enumerate(palabras_ordenadas, 2)}
    palabra_idx["<pad>"] = 0
    palabra_idx["<unk>"] = 1
    
    idx_palabra = {idx: palabra for palabra, idx in palabra_idx.items()}
    return palabra_idx, idx_palabra