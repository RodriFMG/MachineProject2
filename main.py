import h5py
import numpy
import numpy as np

from Constants import DataDicEjes, DataPath


# TypeData:

# acc: retorna una lista de numpy con el eje X, Y, Z de body_acc
# gyro: retorna una lista de numpy con el eje X, Y, Z de body_gyro
# total: retorna una lista de numpy con el eje X, Y, Z de total_acc
def ObtenerData(DataPath, TypeData):
    Ejes = DataDicEjes[TypeData]
    Data = []
    Labels = []

    with h5py.File(DataPath, "r") as f:
        for clave in f.keys():

            # Para realizar un proceso similar al list.include(data) de js, pues
            # directamente es con if data in PythonList, en python. ( mismo proceso que
            # el include )
            if clave in Ejes:
                Data.append(f[clave][:])
            elif clave == "y":
                Labels.append(f[clave][:])

    # Para verificar que una lista está vacio, es con if not ListPython, o len(ListPython)== 0
    if not Data:
        print("Se colocó la etiqueta, solo se accede con: "
              f"acc, gyro o total. Se colocó: {TypeData}")
        exit(0)

    # Data: Array con los ejes de los cálculos realizados en las acciones
    # Labels: Array con las etiquetas correctas de las acciones.

    x = np.array(Data)
    y = np.array(Labels)


    x = np.transpose(x, (1, 0, 2))
    y = np.reshape(y, (-1))
    return x, y


x, y = ObtenerData(DataPath, "acc")

# x.shape: Cálculos realizados en la actividad de la muestra.

# [0]: # de muestras
# [1]: # de ejes (eje x, y, z)
# [2]: todos los calculos realizados en ese eje para esa muestra

# y.shape: Etiqueta real de la actividad realizada ( representación de la activdad real
# del vector x.

# [0]: # de muestras.

print(x.shape, y.shape)
