import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit,StratifiedShuffleSplit




problem_title = 'Predicting the market value of a football player'


_target_column_name = 'log_market_value_in_eur'


Predictions = rw.prediction_types.make_regression()


workflow = rw.workflows.Estimator()



score_types = [
    rw.score_types.RMSE(),
    rw.score_types.RelativeRMSE(name='rel_rmse'),
]



def get_cv(X, y):
    cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=57)
    return cv.split(X)



def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_array = data.drop([_target_column_name], axis=1).values
    return X_array, y_array



def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)



def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)




