"""
Docstring 
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import Traductor, collate_fn
from static.classes import Transformer
from static.constants import DATOS_PATH, MAX_SEQ_LEN
from utils.preprocess import construir_palabras, open_df, preprocesar_enunciado
from utils.train import train
from utils.utils import get_device

DEVICE = get_device()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    batch_size: int = 32
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    num_layers: int = 6
    dropout: float = 0.1
    learning_rate: float = 0.0001
    max_len: int = MAX_SEQ_LEN
    epochs: int = 5
    model_save_path: str = "models/transformer_model.pth"
    
    def __post_init__(self):
        """Validar parámetros de configuración."""
        if self.batch_size <= 0:
            raise ValueError("batch_size debe ser positivo")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model debe ser divisible por num_heads")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo")
        logger.info(f"Configuración de entrenamiento inicializada: {self}")


def load_and_preprocess_data(data_path: str) -> Tuple[Dict[str, int], Dict[int, str], DataLoader]:
    """Cargar datos, preprocesar y crear DataLoader."""
    logger.info(f"Cargando datos desde {data_path}")
    df = open_df(data_path)
    spanish = df["spanish"].tolist()
    english = df["english"].tolist()
    logger.info(f"Se cargaron {len(spanish)} pares de oraciones")
    
    logger.info("Preprocesando oraciones")
    spanish_preprocesado = [preprocesar_enunciado(s) for s in spanish]
    english_preprocesado = [preprocesar_enunciado(e) for e in english]
    
    logger.info("Construyendo índices de vocabulario")
    english_palabra_idx, english_idx_palabra = construir_palabras(english_preprocesado)
    spanish_palabra_idx, spanish_idx_palabra = construir_palabras(spanish_preprocesado)
    
    logger.info(f"Tamaño de vocabulario español: {len(spanish_palabra_idx)}")
    logger.info(f"Tamaño de vocabulario inglés: {len(english_palabra_idx)}")
    
    dataset = Traductor(spanish_preprocesado, english_preprocesado, spanish_palabra_idx, english_palabra_idx)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    return spanish_palabra_idx, english_palabra_idx, dataloader


def initialize_model(config: TrainingConfig, spanish_vocab_size: int, english_vocab_size: int) -> Transformer:
    """Inicializar modelo transformer con configuración especificada."""
    logger.info("Inicializando modelo transformer")
    model = Transformer(
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        input_vocab_size=spanish_vocab_size,
        target_vocab_size=english_vocab_size,
        max_len=config.max_len,
        dropout=config.dropout
    )
    model.to(DEVICE)
    logger.info(f"Modelo inicializado en dispositivo: {DEVICE}")
    return model


def initialize_optimizer_and_loss(model: Transformer, config: TrainingConfig) -> Tuple[optim.Optimizer, nn.Module]:
    """Inicializar optimizador y función de pérdida."""
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    logger.info(f"Optimizador inicializado: Adam con lr={config.learning_rate}")
    return optimizer, criterion


def save_model(model: Transformer, config: TrainingConfig) -> None:
    """Guardar checkpoint del modelo en disco."""
    model_path = Path(config.model_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, model_path)
    logger.info(f"Modelo guardado en {model_path}")


def main(args, config: TrainingConfig = None) -> None:
    if config is None:
        config = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            dropout=args.dropout,
            model_save_path=args.data_path
        )
    
    logger.info("Iniciando pipeline de entrenamiento")
    
    # Cargar y preprocesar datos
    spanish_palabra_idx, english_palabra_idx, dataloader = load_and_preprocess_data(DATOS_PATH)
    
    # Inicializar modelo
    spanish_vocab_size = len(spanish_palabra_idx)
    english_vocab_size = len(english_palabra_idx)
    model = initialize_model(config, spanish_vocab_size, english_vocab_size)
    
    # Inicializar optimizador y pérdida
    optimizer, criterion = initialize_optimizer_and_loss(model, config)
    
    # Entrenamiento
    logger.info(f"Iniciando entrenamiento con {config.epochs} épocas")
    train(model, dataloader, criterion, optimizer, epochs=config.epochs)
    
    # Guardar modelo
    save_model(model, config)
    logger.info("Pipeline de entrenamiento completado")
    return model


def parse_args():
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenador de modelo Transformer para traducción"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=DATOS_PATH,
        help="Ruta al archivo de datos",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño del batch",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Número de épocas",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Tasa de aprendizaje",
    )
    
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Dimensión del modelo",
    )
    
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Número de heads de atención",
    )
    
    parser.add_argument(
        "--d-ff",
        type=int,
        default=2048,
        help="Dimensión de feed-forward",
    )
    
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Número de capas",
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Probabilidad de dropout",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args=args, config=None) # TODO: por ahora none