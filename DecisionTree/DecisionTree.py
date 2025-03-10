from Methods import Ganancia, EntropyPadre, EntropyContentNode
from Node import Node
import numpy as np


class DecisionTree:

    # Todo este proceso se realiza con listas de python, por las concatenaciones directas.
    def __init__(self, values, features, umbral, is3D=False, Method="Entropy"):
        '''
        :param values: Lista de las etiquetas
        :param features: Lista de las coordenadas
        :param umbral: Valor mínimo de la Entropía o Ganancia que va a servir como punto de pare.
        :param is3D: Verifica si se pasarán datos con 3 dimensiones: [eje x, eje y, eje z]
        '''

        self.root = Node(value=values, feature=features)
        self.umbral = umbral
        self.method = Method
        self.is3D = is3D

        make_tree(self.root, self.umbral, self.method, self.is3D)

    def Clasificacion(self, cords):
        return prediction(self.root, cords, self.is3D)


def MethodByAxis(node, AxisArray, Axis, method, is3D):
    loss = []

    for index in range(len(AxisArray) - 1):
        MiddlePoint = (AxisArray[index + 1] + AxisArray[index]) / 2
        left, right = node.split(MiddlePoint, Axis, is3D)
        loss_value = EntropyPadre(left, right) if method == "Entropy" else Ganancia(left, right)

        loss.append(np.array([loss_value, MiddlePoint, Axis]))

    # if not ObjectIterable, es para decir algo similar a if len(ObjectIterable) == 0
    # if value in ObjectIterable, es igual a decir if ObjectIterable.include(value), en ts o js.
    if not loss:
        raise ValueError("El nodo padre no tiene elementos, no debería llegar acá.")

    # para numpy es numpylist.size == 0, para ver si está vacia.

    return np.array(loss)


def make_tree(node, umbral, method, is3D):
    # Si es menor que el umbral, indica que está correctamente dividido las clases en ese nodo.
    if EntropyContentNode(node) < umbral:
        return

    eje_x = sorted(node.feature[:, 0])
    eje_y = sorted(node.feature[:, 1])

    # Para que en numpy se pueda realizar algo similar que el PythonList.append(), se usa NumpyList.concatenate()
    # adentro como parámetro recibe el conjunto de listas ov alores que se concatenarán
    # de tal forma que si quiero concatenar una lista (con la misma estructura que la original), pues
    # list = np.concatenate([list, newlist...]), newlist... significa que podemos poner la cantidad de listas
    # que queremos concatenar ahí separados en comas.

    # Recordar, para controlan el tema de que algunas funciones reciben parámetros ilimitados, numpy utiliza:
    # para ejes o dimensiones, se usa: (eje1, eje2, eje3, ...) en una tupla, para lista o valores no relacionados
    # con ejes, se usa: [list1, list2, list3, ...], que mantengan la misma estracutra, si son listas, se concatenan}
    # listas, si son valores, pues valores.

    # en python, usa solamente el parámetro de función *parameter, que es similar a un template<typename...> de c++
    # recibe parámetros infinitos hasta que se declare ese parámetro en cuestion (los que estan antes de ese
    # se declaran de forma normal)
    loss = np.concatenate([
                           MethodByAxis(node, eje_x, "x", method, is3D),
                           MethodByAxis(node, eje_y, "y", method, is3D)])

    if is3D:
        eje_z = sorted(node.feature[:][2])
        loss = np.concatenate([
                               loss,
                               MethodByAxis(node, eje_z, "z", method, is3D)
                               ])

    LossSorted = sorted(loss, key=lambda x: x[0])

    # Rescatamos la división que cause menor error.
    Trazo, Axi = LossSorted[0, 1], LossSorted[0, 2]

    left, right = node.split(Trazo, Axi, is3D)

    node.UpdateNode(left, right, condition=lambda x: x <= Trazo, AxiCondition=Axi)

    make_tree(node.left, umbral, method, is3D)
    make_tree(node.right, umbral, method, is3D)


# Se espera que cords sea un array con la siguiente estructura:
# [eje x, eje y]
def prediction(node, cords, is3D):
    if node.condition is not None:

        # Si es eje x, se sacará el primer indice ( que representa el eje x ), sino
        # el segundo indice ( que representa al eje y ).
        if node.AxiCondition == "x":
            AxiCondition = 0
        elif node.AxiCondition == "y":
            AxiCondition = 1
        elif is3D and node.AxiCondition == "z":
            AxiCondition = 2
        else:
            raise ValueError(f"El eje colocado no existe: {node.AxiCondition}")

        return prediction(node.left, cords, is3D) if node.condition(cords[AxiCondition]) else (
            prediction(node.right, cords, is3D))
    else:
        if node.isLeaf() is True:
            return node.prediction()
        else:
            raise ValueError("Se esperaba que el nodo sea hoja.")
