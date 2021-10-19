# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

[Gradient Boosting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) is adopted for this particular task with a [Randomized Search CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV) utilised for a hyperparameter tuning. The full model pipeline also consists of [One Hot Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) for handling categorical features and [Label Binarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer) used to represent a dependent variable in a binary fashion. Pipeline steps are as follows:
- clean data
- train/test split
- binarize y and encode categorical features
- train model with a hyperparameter tuning
- select best predictor and evaluate on test set
- prepare model for deployment

All artifacts retrieved within the pipeline (datasets, encoders, model) are stored on AWS S3 bucket and versioned with [DVC](https://dvc.org/).

## Intended Use

Prediction task of the model is to determine whether a given person falls into a >50K or <=50K income group (a binary classification problem).

## Training Data

The **Census Income Data Set** was obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income). It consists of fourteen explanatory variables and a `salary` as a response variable. A full dataset has 32,561 samples and its original and cleaned versions are stored on AWS S3 bucket with metadata in `census.csv.dvc` and `census_cleaned.csv.dvc` files available in https://github.com/mrstelmach/Deploy-ML-Model-on-Heroku-with-FastAPI/tree/main/starter/data. For training purposes 80% of samples are used and saved to `census_train.csv`.

## Evaluation Data

Random split with a 20% of the full dataset for model evaluation purposes is used and samples are saved to `census_test.csv` file.

## Metrics

F1 score is used as a main model performance metric. Precision and recall are also tracked. The current performance for both train and test datasets (and additionally for various data slices) is available in [performance.csv](https://github.com/mrstelmach/Deploy-ML-Model-on-Heroku-with-FastAPI/blob/main/starter/data/performance.csv) file. The most recent model achieves **0.73** F1 score for test dataset.

## Ethical Considerations

## Caveats and Recommendations
