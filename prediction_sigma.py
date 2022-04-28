import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn import metrics, svm
from lssvr import *

# Глобальные константы
START_DATE = "2020-11-24"
PRED_COLUMN = "PAY"
PRED_LEN = 366
INPUT_LEN = 180

def make_data(file):
    df = pd.read_csv(file, sep=';')
    df['PAY'] = df['PAY'].apply(lambda x: float(x.replace(',', '.')))
    df['PAY'] = df['PAY'] / 1000
    df['Date'] = pd.to_datetime(df['PAY_DATE'], format='%d.%m.%Y')
    df = df.sort_values(by='Date')
    df = df.set_index(pd.DatetimeIndex(df['Date']))
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(['Date', 'PAY_DATE', 'CNT'], axis=1)
    return df

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def XGBoostModel(train_x, train_y, test_x):
    model = xgboost.XGBRegressor(booster='gblinear', learning_rate=0.2111, random_state=42, n_estimators=197)
    model.fit(train_x, train_y)

    y_pred = model.predict(test_x)

    return y_pred

def LSSVRModel(train_x, train_y, test_x):
    lsclf = LSSVR(kernel='rbf',
                  C=10,
                  gamma=0.04)
    train_ = np.reshape(train_y, train_y.shape[0])
    lsclf.fit(train_x, train_)
    y_pred_lsc = lsclf.predict(test_x)

    return y_pred_lsc

def SVRModel(train_x, train_y, test_x):
    clf = svm.SVR(kernel='rbf',
                  gamma=0.1,
                  C=0.5,
                  verbose=False,
                  tol=1e-10,
                  epsilon=0.0411)
    clf.fit(train_x, train_y)
    y_pred_cvr = clf.predict(test_x)

    return y_pred_cvr

def LGBMModel(train_x, train_y, test_x):
    model_lgb = lgb.LGBMRegressor(learning_rate=0.071, max_depth=11, n_estimators=132, boosting_type='goss',
                                  num_leaves=25, random_state=42)
    model_lgb.fit(train_x, train_y)

    y_pred_lgb = model_lgb.predict(test_x)

    return y_pred_lgb

def get_answer(file, num_model):
    df = make_data(file)
    scaler = MinMaxScaler()
    df_scal = scaler.fit_transform(df)
    df_scal = pd.DataFrame(df_scal, columns=df.columns)

    test = df_scal[df.index >= START_DATE]
    train = df_scal[df.index < START_DATE]

    train_x = []
    for i in range(INPUT_LEN, len(train)):
        train_x.append(train.iloc[i - INPUT_LEN:i])
    train_y = train.iloc[INPUT_LEN:]
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))

    test_x = []
    for i in range(INPUT_LEN, len(test)):
        test_x.append(test.iloc[i - INPUT_LEN:i])
    test_y = test.iloc[INPUT_LEN:]
    test_x = np.array(test_x)
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))

    if num_model == 1:
        y_pred = LSSVRModel(train_x, train_y, test_x)
        temp = df_scal[-PRED_LEN + INPUT_LEN:]
        temp.PAY = y_pred
        prediction_lssvr = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
        prediction_lssvr["PAY"] = prediction_lssvr["PAY"] * 1000
        # prediction_lssvr["Date"] = pd.DatetimeIndex(df.index[-PRED_LEN + INPUT_LEN:])
        indexes = pd.DatetimeIndex(df.index[-PRED_LEN+INPUT_LEN:])
        indexes = indexes.strftime('%Y-%m-%d')
        prediction_lssvr = prediction_lssvr.set_index(indexes)
        return prediction_lssvr
    if num_model == 2:
        y_pred = XGBoostModel(train_x, train_y, test_x)
        temp = df_scal[-PRED_LEN + INPUT_LEN:]
        temp.PAY = y_pred
        prediction_xgb = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
        prediction_xgb["PAY"] = prediction_xgb["PAY"] * 1000
        indexes = pd.DatetimeIndex(df.index[-PRED_LEN + INPUT_LEN:])
        indexes = indexes.strftime('%Y-%m-%d')
        prediction_xgb = prediction_xgb.set_index(indexes)
        return prediction_xgb
    if num_model == 3:
        y_pred = SVRModel(train_x, train_y, test_x)
        temp = df_scal[-PRED_LEN + INPUT_LEN:]
        temp.PAY = y_pred
        prediction_svr = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
        prediction_svr["PAY"] = prediction_svr["PAY"] * 1000
        indexes = pd.DatetimeIndex(df.index[-PRED_LEN + INPUT_LEN:])
        indexes = indexes.strftime('%Y-%m-%d')
        prediction_svr = prediction_svr.set_index(indexes)
        return prediction_svr
    if num_model == 4:
        y_pred = LGBMModel(train_x, train_y, test_x)
        temp = df_scal[-PRED_LEN + INPUT_LEN:]
        temp.PAY = y_pred
        prediction_lgb = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
        prediction_lgb["PAY"] = prediction_lgb["PAY"] * 1000
        indexes = pd.DatetimeIndex(df.index[-PRED_LEN + INPUT_LEN:])
        indexes = indexes.strftime('%Y-%m-%d')
        prediction_lgb = prediction_lgb.set_index(indexes)
        return prediction_lgb

if __name__ == '__main__':
    y_pred = get_answer('pay2021-11-24.csv', 1)