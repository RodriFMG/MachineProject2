import h5py
import numpy as np

from Constants import DataDicEjes


# Lo que se busca que si tenemos #muestras x ejes x valores
# el resultado ahora sea: #muestras x sublista
# esa sublista tenga todos los valores de cada uno de los ejes de esa dimensión en una sola lista, respetando el orden
# de los indices.
def EstructurarData(Data):

    # con *list realiza algo similar al ...list de ts o js
    # con zip(), guardo todo el contenido de esas listas en tuplas
    # usando map( list, zip()), cada tupla lo convierto a una lista (sublista, lo que buscamos)
    # y para que la lista retornada por el map sea un python list, ponemos list(map())

    # Recordar que list() es una función de python para convertir un objeto iterable a una lista
    # así que se podría poner como función en el map. sintax map: map(function, objecto iterable)

    # Esto último se realiza porque zip() arroja en tupla (elementos...) y nosotros buscamos en listas
    # [elementos...]

    # list(map(list, zip(*muestra)))

    # O simplemente hacer una transpuesta, que se consigue el mismo objetivo xd.
    # de esta manera o usando np.tranpose(array numpy, (axis)) <- cumple el mismo objetivo xd.
    return np.array([ muestra.T for muestra in Data])


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

    # Transpose de esta manera para tener <- # muestras, # elementos, # elementos por eje.
    x = np.transpose(x, (1, 2, 0))
    y = np.reshape(y, (-1))

    return x, y


# x.shape: Cálculos realizados en la actividad de la muestra.

# [0]: # de muestras
# [1]: # de ejes (eje x, y, z)
# [2]: todos los calculos realizados en ese eje para esa muestra

# y.shape: Etiqueta real de la actividad realizada ( representación de la activdad real
# del vector x.

# [0]: # de muestras.
