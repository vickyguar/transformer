## Traductor

Esto es una práctica/ ejercicio autodidacta para entender el funcionamiento de los transformers. Es una réplica del paper "Attention is all you need" y es un transformer hecho from scratch. 

Base de datos para entrenamiento tomada de [tatoeba](https://tatoeba.org/es/downloads). Descargué de allí todas las oraciones en idioma Español que tienen traducciones al idioma Inglés. El archivo de descarga es en formato .tsv (como un csv pero separado por tabs, \t, en lugar de comas), se puede manipular con pandas leyéndolo de la siguiente manera:

 ```python
 datos = pd.read_csv(DATOS_PATH, sep="\t", header=None, encoding="utf-8")
 ```