
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from sklearn.metrics import accuracy_score, f1_score
from data_preparation import prepare_data



# 1. Prepare data

X_train, X_test, y_train, y_test = prepare_data()

X_train = pnp.array(X_train, requires_grad=False)
X_test = np.array(X_test)
y_train = pnp.array(y_train, requires_grad=False)
y_test = np.array(y_test)

n_qubits = 2
n_layers = 2
epochs = 50

dev = qml.device("default.qubit", wires=n_qubits)


# 2. Define quantum circuit


@qml.qnode(dev, interface="autograd")
def quantum_circuit(inputs, weights):

    qml.AngleEmbedding(
        inputs,
        wires=range(n_qubits),
        rotation="Y"
    )

    qml.StronglyEntanglingLayers(
        weights,
        wires=range(n_qubits)
    )

    return qml.expval(qml.PauliZ(0))



# 3. Define loss function


def loss_function(weights):

    predictions = pnp.stack([
        quantum_circuit(sample, weights)
        for sample in X_train
    ])

    targets = 2.0 * y_train - 1.0

    return pnp.mean((predictions - targets) ** 2)


 
# 4. Prediction function


def evaluate(weights):

    predictions = []

    for sample in X_test:

        output = quantum_circuit(sample, weights)

        predictions.append(
            1 if output >= 0 else 0
        )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    return accuracy, f1


