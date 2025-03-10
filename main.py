import h5py
import numpy
import numpy as np

# TypeData:

# acc: retorna una lista de numpy con el eje X, Y, Z de body_acc
# gyro: retorna una lista de numpy con el eje X, Y, Z de
def ObtenerData(DataPath, TypeData):
    x = []
    y = []

    Data = {
        "acc" : [ "", "", ""]
    }

    Data = "C:/Users/RODRIGO/PycharmProjects/MachineProject2/Data1/train.h5"

x = []
y = []

with h5py.File(Data, "r") as f:
    for clave in f.keys():
        xd = np.array(f[clave][:])
        print(clave)