
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model_name, y_test, predictions):

    results = {
        "Model": model_name,
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

    print(f"\n{model_name} RESULTS")
    print("-" * 40)

    for metric, value in results.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return results


def train_classical_models(X_train, X_test, y_train, y_test):

    results = []

   
    # MODEL 1: LOGISTIC REGRESSION
 
    print("\nTraining Logistic Regression...")

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    logistic_model.fit(X_train, y_train)

    logistic_predictions = logistic_model.predict(X_test)

    logistic_results = evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_predictions
    )

    results.append(logistic_results)

    
    # MODEL 2: SUPPORT VECTOR MACHINE
    

    print("\nTraining Support Vector Machine...")

    svm_model = SVC(
        kernel="rbf",
        C=1.0,
        random_state=42
    )

    svm_model.fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_test)

    svm_results = evaluate_model(
        "SVM",
        y_test,
        svm_predictions
    )

    results.append(svm_results)

    
    # SAVE RESULTS
    

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/classical_results.csv",
        index=False
    )

    print("\nClassical model results saved!")

    return {
        "logistic_model": logistic_model,
        "svm_model": svm_model,
        "logistic_predictions": logistic_predictions,
        "svm_predictions": svm_predictions,
        "results": results
    }


if __name__ == "__main__":

    from data_preparation import prepare_data

    X_train, X_test, y_train, y_test = prepare_data()

    train_classical_models(
        X_train,
        X_test,
        y_train,
        y_test
    )
