
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data():

    # 1. Load Iris dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # 2. Select only two classes:
    # 0 = Setosa
    # 1 = Versicolor
    selected_indices = np.where(y < 2)

    X = X[selected_indices]
    y = y[selected_indices]

    # 3. Select two features:
    # Petal length and petal width
    X = X[:, [2, 3]]

    # 4. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # 5. Standardize the features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 6. Display dataset information
    print("\nDATASET INFORMATION")
    print("-------------------")

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Number of features:", X_train.shape[1])

    print("\nTraining class distribution:")
    print("Class 0:", np.sum(y_train == 0))
    print("Class 1:", np.sum(y_train == 1))

    print("\nTesting class distribution:")
    print("Class 0:", np.sum(y_test == 0))
    print("Class 1:", np.sum(y_test == 1))

    print("\nFirst five training samples:")
    print(X_train[:5])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    prepare_data()
