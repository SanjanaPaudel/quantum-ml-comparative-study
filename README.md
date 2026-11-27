# Comparative Study of Classical and Quantum Machine Learning

## Overview

This project investigates the training behavior of a Variational Quantum Classifier (VQC) and compares it with classical machine learning models.

The study focuses on two experimental questions:

1. How does variational circuit depth affect model performance and training cost?
2. How do Adam and Quantum Natural Gradient (QNG) optimization methods compare when training a variational quantum classifier?

The experiments use PennyLane's state-vector simulator and the Iris dataset.

## Research Objectives

- Implement a Variational Quantum Classifier using PennyLane.
- Compare quantum and classical machine learning models.
- Investigate the effect of circuit depth on training behavior.
- Compare Adam and Quantum Natural Gradient optimizers.
- Examine the consistency of optimizer results across multiple random initializations.

## Technologies Used

- Python
- PennyLane
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Dataset

The experiments use the Iris dataset from Scikit-learn.

For binary classification, the study selects:

- Setosa (Class 0)
- Versicolor (Class 1)

Two features are used:
- Petal length
- Petal width

Dataset preparation:
- Total samples: 100
- Training samples: 70
- Testing samples: 30
- Train/test split: 70/30
- Random state: 42
- Stratified sampling
- StandardScaler fitted on training data only

## Methodology

### 1. Classical Machine Learning

Two classical models are implemented as baselines:

- Logistic Regression
- Support Vector Machine (RBF kernel)

### 2. Variational Quantum Classifier

The quantum classifier uses:

- 2 qubits
- Angle Embedding with Y rotations
- Strongly Entangling Layers
- Pauli-Z expectation value as the output
- Mean Squared Error as the training loss
- Adam optimizer

The classifier is implemented using PennyLane's `default.qubit` simulator.

### 3. Circuit Depth Experiment

Three circuit configurations are evaluated:

- 1 variational layer
- 2 variational layers
- 3 variational layers

The experiments compare test accuracy, F1 score, final training loss, and simulation training time.

### 4. Optimizer Comparison

Two optimization approaches are investigated:

- Adam optimizer
- Quantum Natural Gradient optimizer

The QNG experiment uses an approximate metric tensor calculated from 10 training samples at the initial circuit parameters. This metric estimate is held fixed during training.

The optimizer comparison is repeated using three random initializations: 42, 7, and 21.

## Experimental Results

### Classical vs Quantum Models

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| SVM | 1.00 | 1.00 | 1.00 | 1.00 |
| Variational Quantum Classifier | 1.00 | 1.00 | 1.00 | 1.00 |

### Circuit Depth Comparison

| Circuit Depth | Accuracy | F1 Score | Final Loss | Training Time (s) |
|---|---:|---:|---:|---:|
| 1 Layer | 1.00 | 1.00 | 0.066538 | 70.62 |
| 2 Layers | 1.00 | 1.00 | 0.019643 | 96.80 |
| 3 Layers | 1.00 | 1.00 | 0.022711 | 103.94 |

The two-layer circuit achieved the lowest final training loss among the tested configurations. Increasing circuit depth increased simulation time without improving test accuracy.

### Adam vs Quantum Natural Gradient

Results averaged across three initializations:

| Metric | Adam | Quantum Natural Gradient |
|---|---:|---:|
| Mean Accuracy | 1.00 | 1.00 |
| Mean F1 Score | 1.00 | 1.00 |
| Mean Final Loss | 0.025735 | 0.056274 |
| Mean Training Time (s) | 38.58 | 36.14 |

Adam achieved lower final training loss across all three runs. Both optimizers achieved 100% test accuracy.

## Key Observations

- All three machine learning models achieved perfect accuracy on the selected test split.
- The two-layer quantum circuit achieved the lowest final training loss in the circuit-depth experiment.
- Increasing circuit depth increased simulation time.
- Adam achieved lower final training loss than the approximate QNG implementation across three initializations.
- Both optimizers achieved the same test accuracy and F1 score.

## Limitations

- The Iris binary classification task is relatively simple and uses a small dataset.
- The test set contains only 30 samples.
- All reported experiments use a classical quantum simulator, not physical quantum hardware.
- The circuit-depth experiment uses one initialization per depth.
- The QNG experiment uses an approximate metric tensor held fixed during optimization.
- These results do not demonstrate quantum advantage or quantum computational superiority.

## Project Structure

```text
quantum-ml-comparative-study/
│
├── data_preparation.py
├── classical_models.py
├── quantum_model.py
├── comparison.py
├── depth_experiment.py
├── optimizer_experiment.py
├── summarize_optimizer_runs.py
│
├── results/
│   ├── classical_results.csv
│   ├── quantum_results.csv
│   ├── quantum_training_history.csv
│   ├── model_comparison.csv
│   ├── depth_experiment.csv
│   ├── optimizer_comparison.csv
│   └── optimizer_runs.csv
│
├── plots/
│   ├── model_comparison.png
│   ├── quantum_training_loss.png
│   ├── circuit_depth_comparison.png
│   ├── depth_training_loss.png
│   └── optimizer_training_comparison.png
│
├── requirements.txt
└── README.md
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/SanjanaPaudel/quantum-ml-comparative-study.git

cd quantum-ml-comparative-study
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the experiments

Classical models:

```bash
python classical_models.py
```

Quantum classifier:

```bash
python quantum_model.py
```

Model comparison:

```bash
python comparison.py
```

Circuit depth experiment:

```bash
python depth_experiment.py
```

Optimizer comparison:

```bash
python optimizer_experiment.py
```

Summarize optimizer runs:

```bash
python summarize_optimizer_runs.py
```

## Future Work

- Evaluate the models on more challenging classification datasets.
- Investigate different quantum circuit architectures.
- Study the effect of circuit initialization and learning rates.
- Explore dynamically updated quantum metric tensors.
- Investigate quantum Fisher information and natural-gradient optimization in greater depth.
- Evaluate circuit performance under simulated quantum noise.

## Author

**Sanjana Paudel**

Software Engineering Student  
Nepal College of Information Technology (NCIT)

GitHub: [SanjanaPaudel](https://github.com/SanjanaPaudel)

---

*This project is an educational and experimental study of variational quantum machine learning. It does not claim quantum advantage.*