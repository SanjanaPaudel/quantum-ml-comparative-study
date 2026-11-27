
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


# 5. Initialize same starting parameters


pnp.random.seed(42)

initial_weights = pnp.array(
    np.random.normal(
        loc=0.0,
        scale=0.1,
        size=(n_layers, n_qubits, 3)
    ),
    requires_grad=True
)



# 6. Train using Adam


print("\nTRAINING WITH ADAM")
print("=" * 50)

adam_weights = pnp.array(
    initial_weights,
    requires_grad=True
)

adam_optimizer = qml.AdamOptimizer(
    stepsize=0.05
)

adam_history = []

start = time.perf_counter()

for epoch in range(epochs):

    adam_weights, loss = adam_optimizer.step_and_cost(
        loss_function,
        adam_weights
    )

    adam_history.append(float(loss))

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}: Loss = {float(loss):.6f}"
        )

adam_time = time.perf_counter() - start

adam_accuracy, adam_f1 = evaluate(adam_weights)



# 7. Estimate an average metric tensor


print("\nCALCULATING QUANTUM METRIC TENSOR")

metric_fn = qml.metric_tensor(quantum_circuit)

# Use 10 evenly spaced training samples
sample_indices = np.linspace(
    0,
    len(X_train) - 1,
    10,
    dtype=int
)

metric_tensors = []

for index in sample_indices:

    metric = metric_fn(
        X_train[index],
        initial_weights
    )

    metric_tensors.append(metric)

average_metric = pnp.mean(
    pnp.stack(metric_tensors),
    axis=0
)

print("Average metric tensor calculated.")


