"""Functions for making
- logging with MLflow,
- modelling,
- hyperparameter tuning,
- choosing features, 
- saving and loading models
more convenient.
"""


import os
import pickle
import warnings
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
import mlflow

from modeling.config import EXPERIMENT_NAME

TRACKING_URI = open("../.mlflow_uri").read().strip()
warnings.filterwarnings('ignore')


def log_to_mlflow(
        ZONEID=None, Model=None, features=None, train_RMSE=None,
        test_RMSE=None, nan_removed=False, zero_removed=False, mean=None,
        hyperparameter=None, model_parameters=None, scaler=None, info=None):
    """Logs to mlflow.

    Args:
      ZONEID: int (Default value = None)
      Model: string (Default value = None)
      train: RMSE
      test: RMSE
      NaN: removed
      Zero: removed
      statistics: 
      mean: int (Default value = None)
      hyperparameter: dict (Default value = None)
      model_parameters: dict (Default value = None)
      features:  (Default value = None)
      train_RMSE:  (Default value = None)
      test_RMSE:  (Default value = None)
      nan_removed:  (Default value = False)
      zero_removed:  (Default value = False)
      scaler:  (Default value = None)
      info:  (Default value = None)

    Returns:

    """

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
    print("\nActive run_id: {}".format(run.info.run_id))

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


# function for modelling
def modelling(data_train, data_test, features, model,
              scaler=None, print_scores=True, log=None,
              infotext_mlflow=None, save_model=True,
              perform_gridCV=True, param_grid=None,
              zone_params=None, n_jobs=-1):
    """

    Args:
      data_train: 
      data_test: 
      features: 
      model: 
      scaler:  (Default value = None)
      print_scores:  (Default value = True)
      log:  (Default value = None)
      infotext_mlflow:  (Default value = None)
      save_model:  (Default value = True)
      perform_gridCV:  (Default value = True)
      param_grid:  (Default value = None)
      zone_params:  (Default value = None)
      n_jobs:  (Default value = -1)

    Returns:

    """
    # Get zones in data.
    zones = np.sort(data_train.ZONEID.unique())

    # Initialize DataFrame where predictions of various zones are saved.
    y_trainpred, y_testpred = pd.DataFrame(), pd.DataFrame()

    # Save scores of linear regression models for different zones
    # in dictionary.
    trainscore, testscore, model_dict, cv_score = {}, {}, {}, {}

    mse_train = 0
    mse_test = 0

    scorer = make_scorer(adjusted_RMSE, greater_is_better=False)

    # loop over zones
    for zone in zones:
        model_clone = clone(model)
        # split train and test data in feature and TARGETVAR parts
        # and cut data to desired zones
        X_train, X_test, y_train, y_test = train_test_feat(
                                                data_train, data_test,
                                                zone, features)
        if zone_params:
            model_clone.set_params(**zone_params[zone])
            perform_gridCV = False

        # scale data if scaler is not None
        if scaler:
            X_train, X_test = scaler_func(X_train, X_test, scaler)

        # train model
        if perform_gridCV:
            if param_grid:
                print(f'ZONEID {zone}')
                cv = GridSearchCV(model_clone, param_grid=param_grid,
                                  scoring=scorer, refit=True,
                                  n_jobs=n_jobs, verbose=2)
                cv.fit(X_train, y_train)
                cv_score[zone] = np.abs(cv.best_score_)
                model_clone = cv.best_estimator_
            else:
                raise ValueError('No parameter grid given for Grid Search')
        else:
            model_clone.fit(X_train, y_train)

        if save_model:
            model_dict[zone] = deepcopy(model_clone)

        # predict train data with the model_clone and calculate train-score
        y_pred = predict_func(model_clone, X_train, y_train)
        y_trainpred = pd.concat([y_trainpred, y_pred], axis=0)
        trainscore['ZONE' + str(zone)] = mean_squared_error(y_pred, y_train,
                                                            squared=False)

        mse_train += np.power(trainscore['ZONE' + str(zone)], 2)\
                     * len(y_train)/len(data_train)

        # predict test data with the model_clone and calculate test-score
        y_pred = predict_func(model_clone, X_test, y_test)
        y_testpred = pd.concat([y_testpred, y_pred], axis=0)
        testscore['ZONE' + str(zone)] = mean_squared_error(y_pred, y_test,
                                                           squared=False)

        mse_test += np.power(testscore['ZONE' + str(zone)], 2) \
                    * len(y_test)/len(data_test)  # 1 / len(zones)

    trainscore['TOTAL'] = np.power(mse_train, 0.5)
    testscore['TOTAL'] = np.power(mse_test, 0.5)

    # print scores if desired
    if print_scores:
        for key in testscore.keys():
            print(f'train-RMSE/test-RMSE {model.__class__.__name__} for {key}: \
                  {round(trainscore[key],3)} {round(testscore[key],3)}\n')

    # track to MLFLow
    if log:
        for key in testscore.keys():
            log_to_mlflow(ZONEID=key, Model=model.__class__.__name__,
                          features=features, train_RMSE=trainscore[key],
                          test_RMSE=testscore[key], nan_removed=True,
                          zero_removed=False, mean=None,
                          hyperparameter=model.get_params(),
                          model_parameters=None, scaler=scaler,
                          info=infotext_mlflow)

    if save_model:
        return trainscore, testscore, model_dict, cv_score
    return trainscore, testscore, cv_score


def get_bestfeatures(df):

    """Get the best feature combination for a model and for each zone.

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


def modelling_fc(data_train, data_test, feature_dict, model, scaler=None,
                 print_scores=True, log=None, infotext_mlflow=None,
                 save_model=True, perform_gridCV=True, param_grid=None,
                 zone_params=None, n_jobs=-1):
    """Performes a grid search for finding the best hyperparameter combination
    for a given model. Optionally saves the results and logs to MLflow.

    Args:
      data_train (pd.Dataframe): Contains the data from the train data set.
      data_test (pd.Dataframe): Contains the data from the test data set.
      feature_dict (dict): Dictionary with the feature combination.
      model (sklearn.model): Model which hyperparameters are to be tuned.
      scaler (sklearn.Scaler):  (Default value = None) Scaler to be applied.
      print_scores (bool):  (Default value = True) Print scores?
      log (bool):  (Default value = None) Log to MLflow?
      infotext_mlflow ():  (Default value = None)
      save_model:  (Default value = True)
      perform_gridCV:  (Default value = True)
      param_grid:  (Default value = None)
      zone_params:  (Default value = None)
      n_jobs:  (Default value = -1)

    Returns:

    """

    if type(param_grid) == dict:
        nfits = len(data_train.ZONEID.unique()) * len(feature_dict.keys()) \
                    * np.prod([len(x) for x in param_grid.values()]) * 5
        print(f'Total number of fits: {nfits}')

    df_results = pd.DataFrame()

    for fc in feature_dict.keys():
        print(f'feature combination: {fc}\n')
        features = feature_dict[fc]
        trainscore, testscore, model_dict, cv_score \
            = modelling(data_train, data_test, features, model, scaler,
                        print_scores, log, infotext_mlflow, save_model,
                        perform_gridCV, param_grid, zone_params, n_jobs)

        df_results_fc = result_to_df(model_dict, testscore,
                                     trainscore, cv_score, fc)
        df_results = pd.concat([df_results, df_results_fc], axis=0)

    df_results = get_bestfeatures(df_results)

    return df_results

def result_to_df(model_dict, testscore, trainscore, cv_score, fc):
    """

    Args:
      model_dict: 
      testscore: 
      trainscore: 
      cv_score: 
      fc: 

    Returns:

    """
    df_results = pd.DataFrame(pd.Series([model_dict[i].get_params()
                                         for i in range(1,11)]),
                                         columns = ['BEST_PARAMS'])
    df_results['CV'] = pd.Series([cv_score[i] for i in range(1,11)])
    df_results['ZONE'] = df_results.index
    df_results.ZONE = df_results.ZONE.apply(lambda x: f'ZONE{x+1}')
    df_results = df_results.set_index('ZONE')
    df_results['MODEL'] = model_dict[1].__class__.__name__
    df_results['FC'] = fc
    df_results = df_results.join(pd.DataFrame.from_dict(testscore,
                                                        orient='index',
                                                        columns=['TESTSCORE'])) # leave out TOTAL
    df_results = df_results.join(pd.DataFrame.from_dict(trainscore,
                                                        orient='index',
                                                        columns=['TRAINSCORE']))
    return df_results


## baseline model for every zone and aggregated over all zones
def baseline(train, test):
    """baseline model for every zone and aggregated over all zones

    Args:
      train: 
      test: 

    Returns:

    """

    # zones to loop over
    zones = np.sort(train.ZONEID.unique())

    # baseline predictions of all sites will be merged into one DataFrame to calculate the RMSE with respect to the observations of all zones
    #finalpred = pd.DataFrame()
    df_results = pd.DataFrame(
                        index=[f'ZONE{zone}' for zone in zones] + ['TOTAL'],
                        columns = ['BEST_PARAMS','CV','MODEL','FC',
                                   'TESTSCORE','TRAINSCORE'])
    df_results.loc['TOTAL'].TRAINSCORE = 0
    df_results.loc['TOTAL'].TESTSCORE = 0
    df_results['MODEL'] = 'Baseline'

    # loop over all zones
    for zone in zones:

        # get train and test data of individual zones
        ytrain = train[train.ZONEID == zone].TARGETVAR
        ytest =  test[test.ZONEID == zone].TARGETVAR

        # baseline predicton for individual zone
        pred_train = np.ones(len(ytrain)) * np.mean(ytrain)
        pred_test = np.ones(len(ytest)) * np.mean(ytrain)

        df_results.loc[f'ZONE{zone}'].TRAINSCORE = \
                        mean_squared_error(ytrain, pred_train, squared=False)
        df_results.loc[f'ZONE{zone}'].TESTSCORE = \
                        mean_squared_error(ytest, pred_test, squared=False)

        df_results.loc['TOTAL'].TRAINSCORE += \
                np.power(df_results.loc[f'ZONE{zone}'].TRAINSCORE,2) \
                * len(ytrain)/len(train)
        df_results.loc['TOTAL'].TESTSCORE += \
                np.power(df_results.loc[f'ZONE{zone}'].TESTSCORE,2) \
                * len(ytest)/len(test)

    df_results.loc['TOTAL'].TRAINSCORE = np.power(
                                        df_results.loc['TOTAL'].TRAINSCORE,.5)
    df_results.loc['TOTAL'].TESTSCORE = np.power(
                                        df_results.loc['TOTAL'].TESTSCORE,.5)

    df_results.index.set_names(['ZONE'], inplace=True)

    return df_results


def scaler_func(X_train, X_test, scaler):
    """

    Args:
      X_train: 
      X_test: 
      scaler: 

    Returns:

    """
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if scaler.__class__.__name__ == "MinMaxScaler":
        print('Scaler: MinMaxScaler')
        print(f'Scaled X_train min/max: {round(X_train.min(),2)}, \
                                        {round(X_train.max(),2)}')
        print(f'Scaled X_test min/max: {round(X_test.min(),2)}, \
                                       {round(X_test.max(),2)}\n')

    if scaler.__class__.__name__ == "StandardScaler":
        print('Scaler: StandardScaler')
        print(f'Scaled X_train mean/std: {round(X_train.mean(),2)}, \
                                         {round(X_train.std(),2)}')
        print(f'Scaled X_test mean/std: {round(X_test.mean(),2)},\
                                        {round(X_test.std(),2)}\n')

    return X_train, X_test

def train_test_feat(data_train, data_test, zone, features):
    """

    Args:
      data_train: 
      data_test: 
      zone: 
      features: 

    Returns:

    """
    X_train = data_train[data_train.ZONEID == zone][features]
    y_train = data_train[data_train.ZONEID == zone].TARGETVAR

    X_test = data_test[data_test.ZONEID == zone][features]
    y_test = data_test[data_test.ZONEID == zone].TARGETVAR
    return X_train, X_test, y_train, y_test


def predict_func(model, X, y):
    """Predicts using a given model 
    and adjusts the result in the interval [0,1].

    Args:
      model (sklearn.model): Model which to use for predicting.
      X (pd.DataFrame): Dataframe with explanatory variables.
      y (np.array): Result of the prediction.

    Returns:
      (np.array): Adjusted result of the prediction.

    """
    y_pred = model.predict(X)
    y_pred = pd.DataFrame(
        [1 if value >= 1 else 0 if value <= 0 else value for value in y_pred],
        index = y.index, columns = ['pred'])
    return y_pred



def get_features(data):
    """Returns a dictionary with different feature combinations 
    of the dataframe.

    Args:
      data (pd.DataFrame): Dataframe from which feature combinations
                           are to be taken.

    Returns:
      (dict): Dictionary with different feature combinations.

    """
    features = data.columns.to_list()
    features = [var for var in features
                if var not in ('ZONEID','TARGETVAR','TIMESTAMP')]

    feature_dict = {}

    feature_dict['all'] = features
    feature_dict['no_deg'] = [var for var in features
            if var not in ('WD100','WD10')]
    feature_dict['no_deg_norm'] = [var for var in features
            if var not in ('WD100','WD10','U100NORM','V100NORM')]
    feature_dict['no_comp'] = [var for var in features
            if var not in ('U10','U100','U100NORM','V10','V100','V100NORM')]
    feature_dict['no_comp_plus_100Norm'] = [var for var in features
            if var not in ('U10','U100','V10','V100')]
    feature_dict['no_ten'] = [var for var in features
            if 'WD10CARD' not in var
            and var not in ('U10','V10','WS10','WD10')]
    feature_dict['no_card'] = [var for var in features if 'CARD' not in var]
    feature_dict['no_card_100Norm'] = [var for var in features
            if 'CARD' not in var and var not in ('U100NORM','V100NORM')]

    return feature_dict

def save_models(model_dict, model_name=None,
                results_train=None, results_test=None):
    """Save models from model_dict.

    Args:
      model_dict (dict): Dictionary with the models.
      model_name (str):  (Default value = None) Model name.
      results_train (dict):  (Default value = None) Scores for the 
                             training data set.
      results_test (dict):  (Default value = None) Scores for the 
                            test data set.

    Returns:
      (str): The path where the results have been written to.

    """
    if model_name:
        pass
    else:
        k1 = list(model_dict)[0]
        k2 = list(model_dict[k1])[0]
        model_name = model_dict[k1][k2].__class__.__name__
    time = datetime.now().strftime("%y%m%d_%H%M")

    dir1 = time+'_'+model_name
    path = '../saved_models/'+dir1
    os.mkdir(path)

    if results_train and results_test:
        save_results(results_train, results_test, path)

    for feat in model_dict.keys():
        dir2 = path+'/'+feat
        os.mkdir(dir2)
        for zone,model in model_dict[feat].items():
            filename = 'zone_'+str(zone).zfill(2)+'.pickle'
            with open(dir2+'/'+filename, 'wb') as outfile:
                pickle.dump(model, outfile)
                outfile.close()
    return path


def load_models(parent_dir):
    """Load models from given parent_dir and return them as a model dictionary.

    Args:
      parent_dir (str): Path where the models are saved.

    Returns:
      (dict) Dictionary with models.

    """
    model_dict = {}
    path = '../saved_models/'+parent_dir
    for feat in os.listdir(path):
        model_dict[feat] = {}
        for zone in os.listdir(path+'/'+feat):
            if len(zone.split('.'))>1:
                pass
            else:
                filename = path+'/'+feat+'/'+zone
                with open(filename,'rb') as infile:
                    loaded = pickle.load(infile)
                    infile.close()
                    model_dict[feat][int(zone.split('.')[0][-2:])] = loaded
    return model_dict


def save_results(results_train, results_test, path):
    """

    Args:
      results_train (dict): Scores for the training data set.
      results_test (dict): Scores for the test data set.
      path (str): Where to save the results.

    Returns:
      None

    """
    features = []
    zones = []
    train_score = []
    test_score = []

    for key in results_train.keys():
        for zone in results_train[key].keys():
            features.append(key)
            zones.append(zone)
            train_score.append(results_train[key][zone])
            test_score.append(results_test[key][zone])

    df = pd.DataFrame({'features':features,'zone': zones,\
                       'train_score': train_score,'test_score': test_score})
    file_name = path.split('/')[-1] + '.csv'
    df.to_csv(path + '/' + file_name, index=False)