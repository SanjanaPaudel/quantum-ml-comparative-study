
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from sklearn.metrics import accuracy_score, f1_score
from data_preparation import prepare_data


# ---------------------------------------
# 1. Prepare data
# ---------------------------------------

X_train, X_test, y_train, y_test = prepare_data()

X_train = pnp.array(X_train, requires_grad=False)
X_test = np.array(X_test)
y_train = pnp.array(y_train, requires_grad=False)
y_test = np.array(y_test)

n_qubits = 2
n_layers = 2
epochs = 50

dev = qml.device("default.qubit", wires=n_qubits)


