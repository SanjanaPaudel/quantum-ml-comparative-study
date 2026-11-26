
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from data_preparation import prepare_data


 
# 1. QUANTUM CIRCUIT CONFIGURATION
 

n_qubits = 2
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)


 
# 2. DEFINE THE QUANTUM CIRCUIT
 

@qml.qnode(dev, interface="autograd")
def quantum_circuit(inputs, weights):

    # Encode classical data into quantum states
    qml.AngleEmbedding(
        inputs,
        wires=range(n_qubits),
        rotation="Y"
    )

    # Trainable quantum layers
    qml.StronglyEntanglingLayers(
        weights,
        wires=range(n_qubits)
    )

    # Measure the first qubit
    return qml.expval(qml.PauliZ(0))


 
# 3. DEFINE THE LOSS FUNCTION
 

def loss_function(weights, X, y):

    outputs = pnp.stack([
        quantum_circuit(sample, weights)
        for sample in X
    ])

    # Convert class labels:
    # Class 0 becomes -1
    # Class 1 becomes +1
    targets = 2.0 * y - 1.0

    # Mean Squared Error
    loss = pnp.mean((outputs - targets) ** 2)

    return loss


 
# 4. MAKE PREDICTIONS
 

def predict(X, weights):

    predictions = []

    for sample in X:

        output = quantum_circuit(sample, weights)

        # Positive output -> Class 1
        # Negative output -> Class 0
        if float(output) >= 0:
            predictions.append(1)
        else:
            predictions.append(0)

    return np.array(predictions)


 
# 5. TRAIN THE QUANTUM MODEL
 

def train_quantum_model(X_train, X_test, y_train, y_test):

    print("\nTRAINING QUANTUM MODEL")
    print("----------------------")

    # Convert training data to PennyLane arrays
    X_train_q = pnp.array(X_train, requires_grad=False)
    y_train_q = pnp.array(y_train, requires_grad=False)

    # Initialize trainable circuit parameters
    pnp.random.seed(42)

    weights = pnp.array(
        pnp.random.normal(
            loc=0,
            scale=0.1,
            size=(n_layers, n_qubits, 3)
        ),
        requires_grad=True
    )

    # Adam optimizer
    optimizer = qml.AdamOptimizer(stepsize=0.05)

    epochs = 100
    loss_history = []

    print("Starting training...\n")

    for epoch in range(epochs):

        weights, current_loss = optimizer.step_and_cost(
            lambda w: loss_function(w, X_train_q, y_train_q),
            weights
        )

        loss_value = float(current_loss)
        loss_history.append(loss_value)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| Loss: {loss_value:.6f}"
            )

    print("\nTraining completed!")

     
    # 6. EVALUATE THE QUANTUM MODEL
     

    predictions = predict(X_test, weights)

    results = {
        "Model": "Variational Quantum Classifier",
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    print("\nQUANTUM MODEL RESULTS")
    print("---------------------")

    for metric, value in results.items():

        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

     
    # 7. SAVE RESULTS
     

    pd.DataFrame([results]).to_csv(
        "results/quantum_results.csv",
        index=False
    )

    pd.DataFrame({
        "Epoch": range(1, epochs + 1),
        "Loss": loss_history
    }).to_csv(
        "results/quantum_training_history.csv",
        index=False
    )

    np.save(
        "results/quantum_weights.npy",
        np.array(weights)
    )

    print("\nQuantum results saved successfully!")

    return {
        "weights": weights,
        "predictions": predictions,
        "results": results,
        "loss_history": loss_history
    }


# 8. RUN THE QUANTUM MODEL


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_data()

    train_quantum_model(
        X_train,
        X_test,
        y_train,
        y_test
    )
