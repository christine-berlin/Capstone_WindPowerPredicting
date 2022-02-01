"""Functions for:
- logging with MLflow,
- modelling,
- hyperparameter tuning,
- finding best feature combinations,
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import mlflow
from modeling.config import EXPERIMENT_NAME
TRACKING_URI = open("../.mlflow_uri").read().strip()
warnings.filterwarnings('ignore')


def log_to_mlflow(
        ZONEID=None, Model=None, features=None, train_RMSE=None,test_RMSE=None, 
        hyperparameter=None, model_parameters=None, scaler=None):
    """Logs to mlflow.

    Args:
      ZONEID (int): Zone number. (Default value = None)
      Model (str): Name of model. (Default value = None)
      features (list): List of features used. (Default value = None)
      train_RMSE (float): RMSE score of train data set. (Default value = None)
      test_RMSE (float): RMSE score of test data set. (Default value = None)
      hyperparameter (dict): Dictionary of the hyperparameters.
                             (Default value = None)
      model_parameters (dict): Dictionary with the model parameters.
                               (Default value = None)
      scaler (sklearn.scaler): Scaler that was applied to the data.
                               (Default value = None)

    Returns:
      None
    """

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    params = {}
    
    if model_parameters:
        params['model parameters'] = model_parameters
    if hyperparameter:
        params['hyperparameter'] = hyperparameter

    mlflow.start_run()
    run = mlflow.active_run()
    print(f"\nActive run_id: {run.info.run_id}")

    if ZONEID:
        mlflow.set_tag("ZONEID", ZONEID)
    if Model:
        mlflow.set_tag("Model", Model)
    if features:
        mlflow.set_tag("features", features)
        mlflow.set_tag("n_features", len(features))
    if scaler:
        mlflow.set_tag("scaler", scaler.__class__.__name__)
    if train_RMSE:
        mlflow.log_metric("train-RMSE", train_RMSE)
    if test_RMSE:
        mlflow.log_metric("test-RMSE", test_RMSE)
    if params:
        mlflow.log_params(params)

    mlflow.end_run()


def adjusted_RMSE(y_test, y_pred):
    """Computes the RMSE after the values in y_pred have been adjusted to the
    interval [0, 1].

    Args:
      y_test (numpy.array): Array with the target of the test data set.
      y_pred (numpy.array): Array with the (unadjusted) values of the
                            prediction of the targer variable.

    Returns:
      float: The adjusted RMSE between y_test and y_pred.

    """
    y_pred = [1 if value >= 1 else 0 if value <= 0 else value
              for value in y_pred]
    return mean_squared_error(y_test, y_pred, squared=False)


def get_bestfeatures(df):

    """Get the best feature combination for a model and for each zone.
       Best means, best result in CV.

    Args:
      df (pd.DataFrame): Contains the test/train-score for one model,
                         10 zones and all feature combinations.

    Returns:
      (pd.DataFrame): Contains the test/train-score for one model,
                      10 zones for best feature combination.

    """

    df_results = pd.DataFrame()

    for zone in df.index.unique():
        df_zone = df.loc[zone]

        df_results = pd.concat([df_results,
                                df_zone[df_zone.CV == df_zone.CV.min()]])

    return df_results


def result_to_df(model_dict, testscore, trainscore, cv_score, fc):
    """Stores the results of the modelling as a Pandas Dataframe.

    Args:
      model_dict (dict): Dictionary with the models.
      testscore (dict): Dictionary with the scores of the test data.
      trainscore (dict): Dictionary with the scores of the train data.
      cv_score (list): List with the score of the cross-validation.
      fc (list): List with the features used in the fitting.

    Returns:
      (pd.DataFrame): Dataframe with results.
    """
    df_results = pd.DataFrame(pd.Series([model_dict[i].get_params()
                                        for i in range(1, 11)]),
                                        columns=['BEST_PARAMS'])
    df_results['CV'] = pd.Series([cv_score[i] for i in range(1,11)])
    df_results['ZONE'] = df_results.index
    df_results.ZONE = df_results.ZONE.apply(lambda x: f'ZONE{x+1}')
    df_results = df_results.set_index('ZONE')
    df_results['MODEL'] = model_dict[1].__class__.__name__
    df_results['FC'] = fc
    df_results = df_results.join(pd.DataFrame.from_dict(
                                    testscore,
                                    orient='index',
                                    columns=['TESTSCORE'])) # leave out TOTAL
    df_results = df_results.join(pd.DataFrame.from_dict(
                                    trainscore,
                                    orient='index',
                                    columns=['TRAINSCORE']))
    return df_results


def scaler_func(X_train, X_test, scaler):
    """Scales the train and test data with the provided scaler.

    Args:
      X_train (pd.DataFrame): Dataframe with the train data.
      X_test (pd.DataFrame): Dataframe with the test data.
      scaler (sklearn.Scaler): MinMaxScaler, StandardScaler or None.

    Returns:
      (sklearn.Scaler): Scaled train data.
      (sklearn.Scaler): Scaled test data.

    """
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)    

    return X_train, X_test


def train_test_split_features(data_train, data_test, zone, features):
    """Returns a pd.DataFrame with the explanatory variables and
    a pd.Series with the target variable, for both train and test data.

    Args:
      data_train (pd.DataFrame): Train data set.
      data_tes (pd.DataFrame): Test data set.
      zone (int): The zone id (id of the wind farm).
      features (list): A list of the column names to be used.

    Returns:
      (pd.DataFrame): Explanatory variables of train data set.
      (pd.Series): Target variable fo train data set.
      (pd.DataFrame): Explanatory variables of test data set.
      (pd.Series): Target variable fo test data set.

    """
    X_train = data_train[data_train.ZONEID == zone][features]
    y_train = data_train[data_train.ZONEID == zone].TARGETVAR

    X_test = data_test[data_test.ZONEID == zone][features]
    y_test = data_test[data_test.ZONEID == zone].TARGETVAR
    return X_train, X_test, y_train, y_test


def predict_func(model, X, y):
    """Predicts using a given model
    and adjusts the result in the interval [0,1].
    Predictions can't have values larger than 1 or smaller than 0, because the energy output 
    consists of nornmalized values in [0,1].

    Args:
      model (sklearn.model): Model which to use for predicting.
      X (pd.DataFrame): Dataframe with explanatory variables.
      y (pd:Series) : Target variable of test data

    Returns:
      (np.array): Adjusted result of the prediction.

    """
    y_pred = model.predict(X)
    y_pred = pd.DataFrame(
        [1 if value >= 1 else 0 if value <= 0 else value for value in y_pred],
        index=y.index, columns=['pred'])
    return y_pred









