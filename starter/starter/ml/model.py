# -*- coding: utf-8 -*-

"""
Functions for training Gradient Boosting with hyperparameter optimization, 
computing metrics and running in inference mode.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV


def train_model(X_train, y_train, hp_iter=5):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    hp_iter: int
        Number of hp tuning iterations.
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
        n_iter=hp_iter,
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


def get_slices_performance(model, data, categorical_features, process_data_fn,
                           one_hot_encoder, label_binarizer, label="salary"):
    """
    Output performance of the model on slices of the data for categorical
    features with comparison to overall performance.
    
    Inputs
    ------
    model : ???
        Trained machine learning model.
    data : pd.DataFrame
        Data for computing performance.
    categorical_features : list
        List of categorical features.
    process_data_fn : function
        Function for preprocessing data.
    one_hot_encoder : sklearn.preprocessing.OneHotEncoder
        Fitted instance of OneHotEncoder.
    label_binarizer : sklearn.preprocessing.LabelBinarizer
        Fitted instance of LabelBinarizer.
    label : str
        Column name with y value.
    Returns
    -------
    performance : pd.DataFrame
        Data frame with performance on slices and overall score.
    """
    performance = []
    categorical_features.insert(0, 'overall')
    data['overall'] = 'Overall'

    for feature in categorical_features:
        if 'overall' in categorical_features:
            categorical_features.remove('overall')
        for val in data[feature].unique():
            idx = (data[feature] == val)
            data_slice = data[idx].copy()
            if 'overall' in data_slice.columns:
                data_slice.drop(columns=['overall'], inplace=True)
            X_test, y_test, _, _ = process_data_fn(
                X=data_slice,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=one_hot_encoder, 
                lb=label_binarizer
            )
            y_pred = inference(model, X_test)
            pr, rc, fb = compute_model_metrics(y_test, y_pred)
            performance.append({
                'feature': feature, 'value': val, 'samples': len(data_slice),
                'precision': pr, 'recall': rc, 'fbeta': fb
            })
    
    performance = pd.DataFrame(performance)
    return performance
