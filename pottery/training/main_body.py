import logging

import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.feature_selection import rfe
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn import datasets
from xgboost import XGBClassifier

import pickle
from bayesian_tuner import param_tuning

logging.basicConfig(filename='output/run.log',level=logging.DEBUG)


def main_training_pipeline(input_training_model_dir, input_training_eval_dir, use_bayes, bayes_iteration):

    # # load and split
    # X, y = get_x_and_y()
    X_eval, y_eval = get_x_and_y(pd.read_csv(input_training_eval_dir))
    X, y = get_x_and_y(pd.read_csv(input_training_model_dir))
    X_train, X_test, y_train, y_test = split_dataset(X, y, 0.33)

    # impute

    # feature_selection
    # split dataset

    # cross_validation and parameter tuning
    logging.info(f"*******Starting training for {bayes_iteration} iterations!")
    model, best_param, best_values = cross_validation(X_train.values, y_train.values.ravel(),
                                                      param_tuning_iter=bayes_iteration, bayes_opt=use_bayes)

    if use_bayes:
        model = XGBClassifier(random_state=49)
        model.set_params(**best_param)
        logging.info(f"Best_param: {best_param}\n")
        logging.info(f"Best_value: {best_values}\n")

    evaluate_training_set(model, X_train, y_train, X_test, y_test)
    model = final_evaluation(best_param, X, y, X_eval, y_eval)

    save_model(model, 'output/beastML.pkl')


def get_x_and_y(df: pd.DataFrame):

    X = df[['gender', 'highest_education', 'imd_band', 'age_band', 'num_of_prev_attempts', 'studied_credits',
            'disability', 'productivity']]
    np.array([[1,0,2,30,5,0,100]])
    y = df[['final_result']]
    return X, y


def split_dataset(X:pd.DataFrame, y:pd.DataFrame, test_size: float):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def evaluate_training_set(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    logging.info(f"************ Training eval score: {score}*****************")


def final_evaluation(best_param, X, y, X_eval, y_eval):
    model = XGBClassifier(random_state=49)
    model.set_params(**best_param)
    model.fit(X.values, y.values)
    score = model.score(X_eval.values, y_eval.values)
    logging.info(f"************ FINAL eval score: {score}*****************")
    logging.debug(model.get_params())
    return model


def cross_validation(train_X: np.array, train_y: np.array, param_tuning_iter: int, bayes_opt: bool):
    # init_param = {'n_estimators': 100, 'learning_rate': 0.3, 'max_depth': 6, 'gamma': 0, 'random_state': 42}
    model = XGBClassifier(random_state=42)
    # model.set_params(**init_param)
    best_parameters, best_values = None, None
    if bayes_opt:
        best_parameters, best_values = param_tuning(param_tuning_iter,
                             evaluation_function=lambda p: objective_function(p, train_X, train_y))
        logging.info(f"Best params are: {best_parameters}")
        logging.info(f"Best value is: {best_values}")
    else:
        cv_results = cross_validate(model, train_X, train_y, scoring='accuracy')
        logging.info(cv_results)
    return model, best_parameters, best_values


def objective_function(param: dict, train_X: np.array, train_y: np.array):
    # param = {k: param[k] for k in param if param[k]}
    model = XGBClassifier(random_state=42)
    model.set_params(**param)
    logging.info(param)
    score = cross_val_score(model, train_X, train_y, scoring='accuracy')
    logging.info(score, np.mean(score))
    return -np.mean(score)


def save_model(model, f_name):
    pickle.dump(model, open(f_name, 'wb'))


if __name__ == "__main__":
    main_training_pipeline(input_training_model_dir="input/df2013.csv", input_training_eval_dir="input/df2014.csv",
                           use_bayes=True, bayes_iteration=5)