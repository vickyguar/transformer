from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd


def open_df(csv_path: str) -> pd.DataFrame:
    """
    Carga un DataFrame desde un archivo TSV.
    
    Args:
        csv_path (str): Ruta al archivo.
    
    Returns:
        pd.DataFrame: DataFrame cargado.
    """
    
    df = pd.read_csv(csv_path, sep="\t", header=None, encoding="utf-8")
    df = df.drop(columns=[0, 2])
    df = df.rename(columns={1: "spanish", 3: "english"})
    return df


# ! probar quitando todos los simbolos, tal vez un regex? az a AZ?
def preprocesar_enunciado(enunciado: str) -> str:
    """
    Preprocesa una oración agregando tokens especiales.
    
    Args:
        enunciado (str): Oración de entrada.
    
    Returns:
        str: Oración preprocesada.
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


def construir_palabras(enunciados: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Construye vocabularios de palabras a índices y viceversa.
    
    Args:
        enunciados (List[str]): Lista de oraciones.
    
    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: Vocabularios.
    """
    palabras = [palabra for enunciado in enunciados for palabra in enunciado.split(" ")]
    palabras_count = Counter(palabras)
    palabras_ordenadas = sorted(palabras_count.items(), key=lambda x: x[1], reverse=True)
    palabra_idx = {palabra: idx for idx, (palabra, _) in enumerate(palabras_ordenadas, 2)}
    palabra_idx["<pad>"] = 0
    palabra_idx["<unk>"] = 1
    
    idx_palabra = {idx: palabra for palabra, idx in palabra_idx.items()}
    return palabra_idx, idx_palabra