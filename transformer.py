#region LIBRERIAS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter # revisar si se usa
import math
import numpy as np
import re # revisar si se usa
#endregion

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




