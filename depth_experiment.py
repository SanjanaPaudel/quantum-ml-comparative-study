
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from sklearn.metrics import accuracy_score, f1_score
from data_preparation import prepare_data


   
# 1. Prepare the dataset
   

X_train, X_test, y_train, y_test = prepare_data()

X_train = pnp.array(X_train, requires_grad=False)
X_test = np.array(X_test)
y_train = pnp.array(y_train, requires_grad=False)
y_test = np.array(y_test)

n_qubits = 2
depths = [1, 2, 3]
epochs = 100
learning_rate = 0.05


   
# 2. Create a quantum circuit
   

def create_quantum_circuit(depth):

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs, weights):

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

    return circuit


   
# 3. Define the loss function
   

def calculate_loss(weights, X, y, circuit):

    predictions = pnp.stack([
        circuit(sample, weights)
        for sample in X
    ])

    targets = 2.0 * y - 1.0

    return pnp.mean((predictions - targets) ** 2)


   
# 4. Train and evaluate each depth
   

results = []
all_histories = {}

for depth in depths:

    print("\n" + "=" * 50)
    print(f"EXPERIMENT: {depth} QUANTUM LAYERS")
    print("=" * 50)

    circuit = create_quantum_circuit(depth)

    # Reproducible initialization
    pnp.random.seed(42)

    weights = pnp.array(
        np.random.normal(
            loc=0.0,
            scale=0.1,
            size=(depth, n_qubits, 3)
        ),
        requires_grad=True
    )

    optimizer = qml.AdamOptimizer(
        stepsize=learning_rate
    )

    loss_history = []

    start_time = time.perf_counter()

    for epoch in range(epochs):

        weights, loss = optimizer.step_and_cost(
            lambda w: calculate_loss(
                w,
                X_train,
                y_train,
                circuit
            ),
            weights
        )

        loss_history.append(float(loss))

    training_time = time.perf_counter() - start_time

    # Make predictions on unseen test samples
    predictions = []

    for sample in X_test:

        output = circuit(sample, weights)

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

    final_loss = loss_history[-1]

    all_histories[depth] = loss_history

    results.append({
        "Circuit Depth": depth,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Final Training Loss": final_loss,
        "Training Time (seconds)": training_time
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Training Time: {training_time:.2f} seconds")


   
# 5. Save experiment results
   

results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

results_df.to_csv(
    "results/depth_experiment.csv",
    index=False
)

print("\nCIRCUIT DEPTH COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))


   
# 6. Plot accuracy and F1 score
   

ax = results_df.set_index("Circuit Depth")[
    ["Accuracy", "F1 Score"]
].plot(
    kind="bar",
    figsize=(9, 5),
    width=0.7
)

ax.set_title("Effect of Quantum Circuit Depth")
ax.set_xlabel("Number of Variational Layers")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.3)

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "plots/circuit_depth_comparison.png",
    dpi=300
)

plt.show()


   
# 7. Plot training loss
   

plt.figure(figsize=(9, 5))

for depth, losses in all_histories.items():

    plt.plot(
        range(1, epochs + 1),
        losses,
        label=f"{depth} Layer(s)"
    )

plt.title("Training Loss for Different Circuit Depths")
plt.xlabel("Training Epoch")
plt.ylabel("Mean Squared Error Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "plots/depth_training_loss.png",
    dpi=300
)

plt.show()


print("\nDepth experiment completed successfully!")

print("\nGenerated files:")
print("results/depth_experiment.csv")
print("plots/circuit_depth_comparison.png")
print("plots/depth_training_loss.png")
