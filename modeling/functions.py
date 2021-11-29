## load modules
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
import mlflow

from modeling.config import EXPERIMENT_NAME
TRACKING_URI = open("../.mlflow_uri").read().strip()

warnings.filterwarnings('ignore')

# track to MLFLow 
def log_to_mlflow(
        ZONEID=None, Model=None, features=None, train_RMSE=None, test_RMSE=None, 
        nan_removed=False, zero_removed=False, mean=None, 
        hyperparameter=None, model_parameters=None, scaler=None, info=None):
    '''Logs to mlflow. 
    Parameters to log:
    - ZONEID: int
    - Model: string
    - train-RMSE: float
    - test-RMSE: float
    - NaN removed: bool
    - Zero removed: bool
    - statistics: 
        - mean: int
    - hyperparameter: dict
    - model_parameters: dict
    '''

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    params = {}
    params['Missing Value Handling'] = { 
        "nan_removed": nan_removed, 
        "zero_removed": zero_removed
    }
    if model_parameters:
        params['model parameters'] = model_parameters
    if hyperparameter:
        params['hyperparameter'] = hyperparameter
            
    mlflow.start_run()
    run = mlflow.active_run()
    print("Active run_id: {}".format(run.info.run_id))

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
    if mean:
        mlflow.set_tag("mean", mean)
    if params:
        mlflow.log_params(params)
    if info:
        mlflow.set_tag("info", info)

    mlflow.end_run()

    return None


## function for modelling
def modelling(data_train, data_test, features, model, scaler=None, print_scores=True, log=None, infotext_mlflow=None):

    zones = np.sort(data_train.ZONEID.unique())

    # initialize DataFrame where predictions of various zones are saved
    y_trainpred, y_testpred = pd.DataFrame(), pd.DataFrame()

    # save scores of linear regression models for different zones in dictionary
    trainscore, testscore = {}, {}

    # loop over zones
    for zone in zones:

        # split train and test data in feature and TARGETVAR parts and cut data to desired zones
        X_train = data_train[data_train.ZONEID == zone][features]
        y_train = data_train[data_train.ZONEID == zone].TARGETVAR

        X_test = data_test[data_test.ZONEID == zone][features]
        y_test = data_test[data_test.ZONEID == zone].TARGETVAR

        # scale data if scaler is not None
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            if scaler.__class__.__name__ == "MinMaxScaler":
                print(f'Scaler: MinMaxScaler')
                print(f'Scaled X_train min/max: {X_train.min()}, {X_train.max()}')
                print(f'Scaled X_test min/max: {X_test.min()}, {X_test.max()}\n')

            if scaler.__class__.__name__ == "StandardScaler":
                print(f'Scaler: StandardScaler')
                print(f'Scaled X_train mean/std: {X_train.mean()}, {X_train.std()}')
                print(f'Scaled X_test mean/std: {X_test.mean()}, {X_test.std()}\n')

        # train model
        model.fit(X_train, y_train)

        # predict train data with the model and calculate train-score
        y_pred = model.predict(X_train)
        y_pred = pd.DataFrame([1 if value >= 1 else 0 if value <= 0 else value for value in y_pred], index = y_train.index, columns = ['pred'])
        y_trainpred = pd.concat([y_trainpred, y_pred], axis=0)
        trainscore['ZONE' + str(zone)] = mean_squared_error(y_pred, y_train, squared=False)

        # predict test data with the model and calculate test-score
        y_pred = model.predict(X_test)
        y_pred = pd.DataFrame([1 if value >= 1 else 0 if value <= 0 else value for value in y_pred], index = y_test.index, columns = ['pred'])
        y_testpred = pd.concat([y_testpred, y_pred], axis=0)
        testscore['ZONE' + str(zone)] = mean_squared_error(y_pred, y_test, squared=False)

    # merge final train and test predictions with observations to ensure a right order in both data  
    y_trainpred = y_trainpred.join(data_train.TARGETVAR)
    y_testpred = y_testpred.join(data_test.TARGETVAR)

    y_trainpred.rename(columns = {'TARGETVAR':'test'}, inplace=True)
    y_testpred.rename(columns = {'TARGETVAR':'test'}, inplace=True)

    testscore['TOTAL'] = mean_squared_error(y_testpred['test'], y_testpred['pred'], squared=False)
    trainscore['TOTAL'] = mean_squared_error(y_trainpred['test'], y_trainpred['pred'], squared=False)

    # print scores if desired
    if print_scores:
        for key in testscore.keys():
            print(f'train-RMSE/test-RMSE linear regression model for {key}: {round(trainscore[key],3)} {round(testscore[key],3)}')

    # track to MLFLow
    if log:
        for key in testscore.keys():
            log_to_mlflow(ZONEID=key, Model=model.__class__.__name__, features=features, train_RMSE=trainscore[key], test_RMSE=testscore[key], 
                          nan_removed=True, zero_removed=False, mean=None, 
                          hyperparameter=model.get_params(), model_parameters=None, scaler=scaler, info=infotext_mlflow)
            
    return trainscore, testscore


