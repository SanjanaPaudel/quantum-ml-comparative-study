
import pandas as pd

runs = [
    {
        "Seed": 42,
        "Optimizer": "Adam",
        "Accuracy": 1.0,
        "F1 Score": 1.0,
        "Final Loss": 0.025788,
        "Training Time": 36.883108
    },
    {
        "Seed": 42,
        "Optimizer": "QNG",
        "Accuracy": 1.0,
        "F1 Score": 1.0,
        "Final Loss": 0.055667,
        "Training Time": 36.136724
    },
    {
        "Seed": 7,
        "Optimizer": "Adam",
        "Accuracy": 1.0,
        "F1 Score": 1.0,
        "Final Loss": 0.025071,
        "Training Time": 38.831590
    },
    {
        "Seed": 7,
        "Optimizer": "QNG",
        "Accuracy": 1.0,
        "F1 Score": 1.0,
        "Final Loss": 0.054107,
        "Training Time": 37.327642
    },
    {
        "Seed": 21,
        "Optimizer": "Adam",
        "Accuracy": 1.0,
        "F1 Score": 1.0,
        "Final Loss": 0.026346,
        "Training Time": 40.019232
    },
    {
        "Seed": 21,
        "Optimizer": "QNG",
        "Accuracy": 1.0,
        "F1 Score": 1.0,
        "Final Loss": 0.059049,
        "Training Time": 34.955295
    }
]

df = pd.DataFrame(runs)

df.to_csv(
    "results/optimizer_runs.csv",
    index=False
)

summary = df.groupby("Optimizer").agg({
    "Accuracy": "mean",
    "F1 Score": "mean",
    "Final Loss": ["mean", "std"],
    "Training Time": "mean"
})

print("\nTHREE-RUN OPTIMIZER SUMMARY")
print("=" * 65)
print(summary)

print("\nIndividual experiment results saved to:")
print("results/optimizer_runs.csv")
