## Transformer

Esto es una práctica/ ejercicio autodidacta para entender el funcionamiento de los transformers. Es una réplica del paper "Attention is all you need" y está hecho from scratch. 

Base de datos para entrenamiento tomada de [tatoeba](https://tatoeba.org/es/downloads). Descargué de allí todas las oraciones en idioma Español que tienen traducciones al idioma Inglés. El archivo de descarga es en formato .tsv (como un csv pero separado por tabs, \t, en lugar de comas), se puede manipular con pandas leyéndolo de la siguiente manera:

 ```python
 datos = pd.read_csv(DATOS_PATH, sep="\t", header=None, encoding="utf-8")
 ```
___
### How to use

La notebook [entrenamiento.ipynb](entrenamiento.ipynb) está armada para correr usando la GPU de Colab. Sino, se puede entrenar el transformer usando:

 ```bash
python transformer.py
 ```

 Se puede consultar los parámetros con:

 
 ```bash
python transformer.py --help
 ```
