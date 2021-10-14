# -*- coding: utf-8 -*-

"""
Functions for training Gradient Boosting with hyperparameter optimization, 
computing metrics and running in inference mode.
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    gb_clf = GradientBoostingClassifier()
    params = dict(
        learning_rate = [0.05, 0.1],
        n_estimators = [100, 150, 200, 250],
        max_depth = [2, 3, 4, 5],
        min_samples_leaf = [5, 10, 25]
    )
    rnd_search = RandomizedSearchCV(
        estimator=gb_clf,
        param_distributions=params,
        cv=3,
        n_iter=25,
        n_jobs=-1,
        refit=True
    )
    rnd_search.fit(X_train, y_train)
    
    return rnd_search.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    
    return preds
