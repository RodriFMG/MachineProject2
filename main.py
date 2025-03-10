import numpy as np
from ExtractData import ObtenerData
from Constants import Path
from pyts import preprocessing

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y = ObtenerData(Path, "acc")

scaler = preprocessing.StandardScaler()

x = np.array([scaler.fit_transform(x[i]) for i in range(x.shape[0])])

#x = scaler.fit_transform(x)

print(x.shape)

# % de Datos faltantes

# Cada posición del array será un True o False, si es faltante (nan) será True, si no es faltante False.
MissingProb = np.isnan(x).mean() * 100

print(f"Porcentaje de datos faltantes: {MissingProb:.4f}%")

# Cómo hay un 0% de datos faltntes, no será necesario realizar una interpolación en los datos faltantes.
