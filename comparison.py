
import pandas as pd
import matplotlib.pyplot as plt
import os


  
# 1. LOAD MODEL RESULTS
  

classical_results = pd.read_csv(
    "results/classical_results.csv"
)

quantum_results = pd.read_csv(
    "results/quantum_results.csv"
)

# Combine all three models
comparison = pd.concat(
    [classical_results, quantum_results],
    ignore_index=True
)

# Save combined results
comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)


  
# 2. DISPLAY COMPARISON TABLE
  

print("\nMODEL PERFORMANCE COMPARISON")
print("=" * 65)

print(comparison.to_string(index=False))


  
# 3. PLOT PERFORMANCE COMPARISON
  

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

ax = comparison.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6),
    width=0.75
)

ax.set_title(
    "Classical vs Quantum Machine Learning Performance"
)

ax.set_ylabel("Score")
ax.set_xlabel("Machine Learning Models")
ax.set_ylim(0, 1.1)

ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)

plt.xticks(rotation=15, ha="right")
plt.tight_layout()

os.makedirs("plots", exist_ok=True)

plt.savefig(
    "plots/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


  
# 4. PLOT QUANTUM TRAINING LOSS
  

history = pd.read_csv(
    "results/quantum_training_history.csv"
)

plt.figure(figsize=(9, 5))

plt.plot(
    history["Epoch"],
    history["Loss"],
    linewidth=2
)

plt.title("Variational Quantum Classifier Training Loss")
plt.xlabel("Training Epoch")
plt.ylabel("Mean Squared Error Loss")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "plots/quantum_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nComparison completed successfully!")

print("\nGenerated files:")
print("results/model_comparison.csv")
print("plots/model_comparison.png")
print("plots/quantum_training_loss.png")
