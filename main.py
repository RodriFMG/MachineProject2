import numpy as np
from ExtractData import ObtenerData, SerieToPoints, PermutacionAleatoria, DividirData
from Constants import Path
from pyts import preprocessing

from DecisionTree.Model import DecisionTree, Accuracy

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Para crear un ejecutador directo al main, se debe poner el if __name__ == "__main__":
if __name__ == "__main__":
    x, y = ObtenerData(Path, "acc")
    x, y = SerieToPoints(x, y)
    x, y = PermutacionAleatoria(x, y)

    # Se pone int(input()) para especificar directament eel tipo de dato, si quiero float
    # pongo float(input()) o str, pues str(input()), para especificar y trabajar correctamente
    # porque los números por default van en tipo flotante con el input o str, en caso sea cadena.
    DataAUtilizar = int(input(f"Data total que se usará (máximo: {len(x)}): "))

    x_train, y_train, x_test, y_test = DividirData(x[:DataAUtilizar], y[:DataAUtilizar], 0.8)

    scaler = preprocessing.StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    # x = scaler.fit_transform(x)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # % de Datos faltantes

    # Cada posición del array será un True o False, si es faltante (nan) será True, si no es faltante False.
    MissingProb = np.isnan(x).mean() * 100

    print(f"Porcentaje de datos faltantes: {MissingProb:.4f}%")

    # Cómo hay un 0% de datos faltntes, no será necesario realizar una interpolación en los datos faltantes.
    # is3D, booleano para ajustar el modelo para coordenas con [ejex, ejey, ejez], si es false, se ajusta
    # solo el modelo a coordenadas de 2 dimensiones [ejex, ejey]
    model = DecisionTree(x_train, y_train, 0.6, is3D=True, Method="Entropy")
    print(f"Accuracy: {Accuracy(model, x_test, y_test, is3D=True)}")