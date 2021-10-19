# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

## Intended Use

## Training Data

The **Census Income Data Set** was obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income). It consists of fourteen explanatory variables and a `salary` as a response variable. A full dataset has 32,561 samples and its original and cleaned versions are stored on Amazon S3 bucket with metadata in `census.csv.dvc` and `census_cleaned.csv.dvc` files available in https://github.com/mrstelmach/Deploy-ML-Model-on-Heroku-with-FastAPI/tree/main/starter/data. For training purposes 80% of samples are used and saved as `census_train.csv`.

## Evaluation Data

20% of the full dataset is used for model evaluation.

## Metrics

The current performance for both train and test datasets and additionally for various data slices is available in [performance.csv](https://github.com/mrstelmach/Deploy-ML-Model-on-Heroku-with-FastAPI/blob/main/starter/data/performance.csv) file.

## Ethical Considerations

## Caveats and Recommendations
