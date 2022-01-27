import pandas as pd

def get_feature_combinations():
    """Returns a dictionary with different feature combinations
    of the dataframe.

    Returns:
      (dict): Dictionary with different feature combinations.

    """
    data = pd.read_csv('../data/GEFCom2014Data/Wind/clean_data.csv', parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')

    features = data.columns.to_list()
    features = [var for var in features
                if var not in ('ZONEID', 'TARGETVAR', 'TIMESTAMP')]

    feature_dict = {}

    feature_dict['all'] = features
    feature_dict['no_deg'] = [var for var in features
            if var not in ('WD100', 'WD10')]
    feature_dict['no_deg_norm'] = [var for var in features
            if var not in ('WD100', 'WD10', 'U100NORM', 'V100NORM')]
    feature_dict['no_comp'] = [var for var in features
            if var not in ('U10', 'U100', 'U100NORM', 'V10', 'V100', 'V100NORM')]
    feature_dict['no_comp_plus_100Norm'] = [var for var in features
            if var not in ('U10', 'U100', 'V10', 'V100')]
    feature_dict['no_ten'] = [var for var in features
            if 'WD10CARD' not in var
            and var not in ('U10', 'V10', 'WS10', 'WD10')]
    feature_dict['no_card'] = [var for var in features if 'CARD' not in var]
    feature_dict['no_card_100Norm'] = [var for var in features
            if 'CARD' not in var and var not in ('U100NORM', 'V100NORM')]
    feature_dict['only_ws'] = ['WS100'] 

    return feature_dict
