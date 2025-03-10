import numpy as np


class Node:

    # Para la ganancia, tengo que
    def __init__(self, left=None, right=None, conditional=None, AxiCondition=None, value=None, feature=None):
        self.left = left
        self.right = right
        self.condition = conditional

        # Etiquetas de los elementos que guardará el nodo
        self.value = value

        # Coordenadas de los elementos que guardará el nodo
        self.feature = feature
        self.AxiCondition = AxiCondition

    def prediction(self):

        if self.left is None and self.right is None:
            valores, conteos = np.unique(self.value, return_count=True)
            MaxFrecuencyIndex = np.argmax(conteos)

            return valores[MaxFrecuencyIndex]

        else:
            raise ValueError("Se esperaba que termine en un nodo hoja.")

    def __len__(self):

        # Con raise ValueError(string) <- provoca una excepción, acabando el flujo del programa
        if self.feature is None or self.value is None:
            return 0

        if len(self.feature) != len(self.value):
            raise ValueError("El número de etiquetas es distinta al número de valores.")

        return len(self.feature)

    # condition <- lambda x : x <= value_condition
    def split(self, value_condition, eje, is3D):

        if eje == "x":
            ArraySplit = self.feature[:][0]
        elif eje == "y":
            ArraySplit = self.feature[:][1]
        elif is3D and eje == "z":
            ArraySplit = self.feature[:][2]
        else:
            raise ValueError("Se mando incorrectamente el eje.")

        condition = lambda x: x <= value_condition

        # Array con puro True o False, los True son los que cumplen la condición ( condition )
        # y False los que no.
        ConditionArraySplit = list(map(condition, ArraySplit))

        left_features, left_values = [], []
        right_features, right_values = [], []

        for index, Cumple in enumerate(ConditionArraySplit):
            if Cumple is True:
                left_values.append(self.value[index])
                left_features.append(self.feature[index])
            else:
                right_values.append(self.value[index])
                right_features.append(self.feature[index])

        left = Node(value=left_values, feature=left_features)
        right = Node(value=right_values, feature=right_features)

        return left, right

    def UpdateNode(self, left, right, condition, AxiCondition):

        self.left = left
        self.right = right
        self.condition = condition
        self.AxiCondition = AxiCondition
        self.value = None
        self.feature = None

    def isLeaf(self):
        return self.left is None and self.right is None
