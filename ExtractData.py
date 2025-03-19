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
    return np.array([muestra.T for muestra in Data])


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
        raise ValueError("Se colocó la etiqueta, solo se accede con: "
              f"acc, gyro o total. Se colocó: {TypeData}")

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

# # muestras x 128 x 3 -> (# muestras * 128) x 3
def SerieToPoints(Data, Labels):
    [d1, d2, d3] = Data.shape

    # así hago que cada (ejex, ejey, ejez) de las 128 de cada clase, se ponga cada una de forma independiente.
    # [[[1,2,3], [4,5,6]]...] -> [[1,2,3], [4,5,6]...]
    DataAplanado = Data.reshape(d1 * d2, d3)

    '''como
    LabelsForAplanacion = np.empty(0)
    
    for label in Labels:
        # en numpy, una medida eficiente para hacer esto: [1] * 5 <- [1,1,1,1,1]
        # es np.full(#repes, value or array): np.full(5,1) <- numpyarray[1,1,1,1,1]
        # no confundir con np.full(5, [1]), genera: [[1],[1],[1],[1],[1]]

        LabelsForAplanacion = np.concatenate([LabelsForAplanacion, np.full(d2, label)])
    '''

    # o tambien se puede usar directamente

    # Lo que hace es repetir un elemento del array d2 veces, creando un array numpy
    # [primer elemento repetido d2 veces, 2do elemento repetido d2 veces, ...]

    # tambien se puede poner como [], para especificar de manera exacta cuantas veces queremos que se repita
    # cada valor, ejemplo np.repeat([1,2,3], [1,4,2]) -> numpylist[1,2,2,2,2,3,3]
    LabelsForAplanacion = np.repeat(Labels, d2)

    return DataAplanado, LabelsForAplanacion


# recordar que todas las operaciones tradicionales de python, al realizarlas, retornaran una lista de python
# aunque se hayan hecho con un array numpy u otro. Pero todas las funciones de numpy o pytorch retornaran
# un array numpy o tensor como corresponda, ya sea que se aplique cualquier de los objetos iterables ( lista python,
# numpy o tensor )

def PermutacionAleatoria(Data, Labels):
    # Retorna una lista de los indices [0, 1, 2, 3, 4..., len(Data)]
    IndexList = np.arange(Data.shape[0])

    # Permuta aleatoriamente todo el contenido del array (indices)
    np.random.shuffle(IndexList)

    # con Data[indices list], pues creo esa premutación aleatoria, se aplica en los indices para respetar
    # el orden en los labels.
    DataAleatorio = Data[IndexList]
    LabelsAleatorio = Labels[IndexList]

    return DataAleatorio, LabelsAleatorio


def DividirData(Data, Labels, PorcentajeTrain):

    if PorcentajeTrain > 1 or PorcentajeTrain < 0:
        raise ValueError("El porcentaje debe estar en el rango [0, 1]")

    # np.floor función piso
    # np.ceil función techo

    # Estas funciones retornar un floot, si se usa para indices, se para int(np.floor | np.ceil), para que el tipo
    # de dato sea entero y no float, aunque no tenga decimales en teoría.
    NumTrain = int(np.floor(len(Data) * PorcentajeTrain))

    print(NumTrain)

    # 0:NumTrain <- poner :value, es [0, value-1]
    DataTrain, LabelTrain = Data[:NumTrain], Labels[:NumTrain]
    # NumTrain:Data.size <- poner value: es [value, Data.size -1]
    DataTest, LabelTest = Data[NumTrain:], Labels[NumTrain:]

    print(f"Train: {len(DataTrain)}\n"
          f"Test: {len(DataTest)}\n"
          f"Total: {len(Data)}")

    return DataTrain, LabelTrain, DataTest, LabelTest
