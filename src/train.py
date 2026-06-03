import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def load_data():

    df = pd.read_csv(
        "data/processed/processed_data.csv"
    )

    return df


def split_data(df):

    X = df.drop(
        "is_high_risk",
        axis=1
    )

    y = df["is_high_risk"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def train_logistic_regression(
    X_train,
    y_train
):

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    results = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return results


def tune_random_forest(
    X_train,
    y_train
):

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10]
    }

    model = RandomForestClassifier(
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="f1"
    )

    grid_search.fit(
        X_train,
        y_train
    )

    return grid_search.best_estimator_


def log_model_to_mlflow(
    model,
    accuracy,
    precision,
    recall,
    f1,
    roc_auc
):

    with mlflow.start_run():

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        mlflow.sklearn.log_model(
            model,
            "credit_risk_model"
        )


if __name__ == "__main__":

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(
        df
    )

    model = train_logistic_regression(
        X_train,
        y_train
    )

    results = evaluate_model(
        model,
        X_test,
        y_test
    )

    log_model_to_mlflow(
        model,
        results["accuracy"],
        results["precision"],
        results["recall"],
        results["f1"],
        results["roc_auc"]
    )

    print(results)
