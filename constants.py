import torch
import os

MAX_SEQ_LEN = 128

if os.path.exists("./datos.tsv"):
    DATOS_PATH = "./datos.tsv"
elif os.path.exists("/content/transformer/datos.tsv"):
    DATOS_PATH = "/content/transformer/datos.tsv"
else:
    DATOS_PATH = "datos.tsv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"