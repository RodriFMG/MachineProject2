import numpy as np


# sum lo que hace es sumar todos los elementos de un objeto iterable
# sintaxis: sum(objeto iterable)
# retorna una expresión

# map, similar a js o ts, retorna una nueva lista aplicando, elemento por elemento, una función
# donde cada posición de esa nueva lista será el retorno que tenga el elemento de la lista original
# aplicando esa función

# sintaxis -> map (function return expression, objeto iterable)
# esa función puede ser una lambda o una función que retorne algo

# retorna un objeto iterable, para pasarlo a una lista de python se pone list(map(...))

# Darse cuenta que cada posición de esa lista, será el retorno de la función con cada elemento de la
# lista recibida como parámetro, respetando el orden de los indices.

def EntropyContentNode(Node):
    # Si no es un nodo hoja o si el nodo no contiene elementos, pues retorna 0.
    if Node.value is None or len(Node) == 0:
        return 0

    Clases, ClasesCount = np.unique(Node.value, return_counts=True)
    TotalElements = len(Node.value)

    operacion = lambda x: (x / TotalElements) * np.log2(x / TotalElements)

    Entropy = - sum(map(operacion, ClasesCount))

    return Entropy


def EntropyPadre(left, right):

    EntropyLeft = EntropyContentNode(left)
    EntropyRight = EntropyContentNode(right)

    TotalElements = len(left) + len(right)

    if TotalElements == 0:
        raise ValueError("Los nodos hijos no tienen contenido.")

    Entropy = len(left) / TotalElements * EntropyLeft + len(right) / TotalElements * EntropyRight

    return Entropy


def Ganancia(left, right):

    if right is None or left is None:
        raise ValueError("Se esperaba que ambos hijos sean hojas.")

    # Fucionando ambos arrays de caracteristicas, esto solo funciona en python list
    # en numpy list es np.concatenate((numpylist1, numpylist2, ...))
    LabelsPadre = left.value + right.value


    Labels, LabelsCount = np.unique(LabelsPadre, return_counts=True)
    TotalData = len(LabelsPadre)

    operacion = lambda x: (x / TotalData) * np.log2(x / TotalData)

    HLabels = - sum(map(operacion, LabelsCount))

    Entropy = EntropyPadre(left, right)

    return HLabels - Entropy
