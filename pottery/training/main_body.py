import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.feature_selection import rfe
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets

from xgboost import XGBClassifier
from bayes_opt import param_tuning


def main_training_pipeline(X, y):

    # # load and split
    # X, y = get_x_and_y()

    X_train, X_test, y_train, y_test = split_dataset(X, y, 0.33)

    # impute

    # feature_selection
    # split dataset

    # cross_validation and parameter tuning
    best_param, best_values = cross_validation(X_train, y_train, param_tuning_iter=10)
    model = XGBClassifier(random_state=42)
    model.set_params(**best_param)
    print(f"Best_param: {best_param}\n")
    print(f"Best_value: {best_values}\n")
    model.fit(X_train, y_train)
    score = evaluate_model(model, X_test, y_test)
    print(f"Score: {score}")
    return model


def get_x_and_y(df: pd.DataFrame):
    X = df[['gender', 'highest_education', 'imd_band', 'num_of_prev_attempts', 'studied_credits', 'disability']]
    y = df[['final_result']]
    return X, y


def split_dataset(X:pd.DataFrame, y:pd.DataFrame, test_size: float):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    score = model.score(X_test, y_test)
    return score


def cross_validation(train_X: np.array, train_y: np.array, param_tuning_iter: int):
    init_param = {'n_estimators': 100, 'learning_rate': 0.3, 'max_depth': 6, 'gamma': 0, 'random_state': 42}
    model = XGBClassifier()
    model.set_params(**init_param)
    model = param_tuning(param_tuning_iter,
                         evaluation_function=lambda p: objective_function(p, clone(model), train_X, train_y))
    return model


def objective_function(param: dict, model, train_X: np.array, train_y: np.array):
    param = {k: param[k] for k in param if param[k]}
    model.set_params(**param)
    score = cross_val_score(model, train_X, train_y,scoring='accuracy')
    return -np.mean(score)

if __name__ == "__main__":
    X, y = datasets.load_iris(return_X_y=True)
    main_training_pipeline(X, y)