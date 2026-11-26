
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np


# Number of qubits
n_qubits = 2

# Number of trainable circuit layers
n_layers = 2


# Create a quantum simulator
dev = qml.device(
    "default.qubit",
    wires=n_qubits
)


# Define our quantum circuit
@qml.qnode(dev, interface="autograd")
def quantum_circuit(inputs, weights):

    # STEP 1: Encode classical data into quantum states
    qml.AngleEmbedding(
        inputs,
        wires=range(n_qubits),
        rotation="Y"
    )

    # STEP 2: Apply trainable quantum layers
    qml.StronglyEntanglingLayers(
        weights,
        wires=range(n_qubits)
    )

    # STEP 3: Measure the first qubit
    return qml.expval(qml.PauliZ(0))


# Initialize trainable parameters
pnp.random.seed(42)

initial_weights = pnp.random.normal(
    loc=0,
    scale=0.1,
    size=(n_layers, n_qubits, 3),
    requires_grad=True
)


# Test the quantum circuit
if __name__ == "__main__":

    sample_input = pnp.array(
        [0.5, 0.8],
        requires_grad=False
    )

    output = quantum_circuit(
        sample_input,
        initial_weights
    )

    print("\nQUANTUM CIRCUIT TEST")
    print("--------------------")

    print("Input:", sample_input)
    print("Circuit output:", output)

    print("\nCircuit diagram:")
    print(qml.draw(quantum_circuit)(
        sample_input,
        initial_weights
    ))
