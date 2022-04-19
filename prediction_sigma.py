import numpy as np
from numpy import array, fft
import pandas as pd
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statsmodels.api as sm
import math as m
from math import cos, pi
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings("ignore")
import json
import pickle
import lightgbm as lgb
from sklearn import metrics, svm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, make_scorer
from lssvr import *
import sys

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

def plot_prediction(df, prediction):
    plt.figure(figsize=(25, 12))
    plt.plot(df.index[df.index>=START_DATE], prediction, label="prediction", alpha=.7)
    plt.plot(df.index[df.index>=START_DATE], df.loc[df.index>=START_DATE, PRED_COLUMN], label="real", alpha=.7)
    plt.scatter(df.index[df.index>=START_DATE], prediction, label="prediction", alpha=.7)
    plt.scatter(df.index[df.index>=START_DATE], df.loc[df.index>=START_DATE, PRED_COLUMN], label="real", alpha=.7)
    plt.legend()
    plt.title(PRED_COLUMN + " Prediction")
    plt.show()

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
    # plot_prediction(df[-PRED_LEN + INPUT_LEN:], prediction_cvr.PAY)
    # display(mape(df.PAY[-PRED_LEN + INPUT_LEN:], prediction_cvr.PAY))

def LGBMModel(train_x, train_y, test_x):
    model_lgb = lgb.LGBMRegressor(learning_rate=0.071, max_depth=11, n_estimators=132, boosting_type='goss',
                                  num_leaves=25, random_state=42)
    model_lgb.fit(train_x, train_y)

    y_pred_lgb = model_lgb.predict(test_x)

    return y_pred_lgb

def get_answer(file):
    # file = sys.argv[0]
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
    test_y = np.array(test_y)
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))

    y_pred = LSSVRModel(train_x, train_y, test_x)

    temp = df_scal[-PRED_LEN + INPUT_LEN:]
    temp.PAY = y_pred
    prediction_lssvr = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)

    # prediction_xgb = XGBoostModel(train_x, train_y, test_x) # предсказания модели XGBoostRegressor
    # prediction_lssvr = LSSVRModel(train_x, train_y, test_x)  # предсказания модели LSSVR
    # prediction_svr = SVRModel(train_x, train_y, test_x) # предсказания модели SVR
    # prediction_lgb = LGBMModel(train_x, train_y, test_x) # предсказания модели LightGBMRegressor
    # пример графика
    return prediction_lssvr

if __name__ == '__main__':
    y_pred = get_answer()
    # file = sys.argv[0]
    # df = make_data(file)
    # scaler = MinMaxScaler()
    # df_scal = scaler.fit_transform(df)
    # df_scal = pd.DataFrame(df_scal, columns=df.columns)
    #
    # test = df_scal[df.index >= START_DATE]
    # train = df_scal[df.index < START_DATE]
    #
    # train_x = []
    #
    # for i in range(INPUT_LEN, len(train)):
    #     train_x.append(train.iloc[i - INPUT_LEN:i])
    #
    # train_y = train.iloc[INPUT_LEN:]
    #
    # train_x = np.array(train_x)
    # train_y = np.array(train_y)
    #
    # train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
    #
    # test_x = []
    #
    # for i in range(INPUT_LEN, len(test)):
    #     test_x.append(test.iloc[i - INPUT_LEN:i])
    #
    # test_y = test.iloc[INPUT_LEN:]
    #
    # test_x = np.array(test_x)
    # test_y = np.array(test_y)
    #
    # test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
    #
    # # prediction_xgb = XGBoostModel(train_x, train_y, test_x) # предсказания модели XGBoostRegressor
    # prediction_lssvr = LSSVRModel(train_x, train_y, test_x) # предсказания модели LSSVR
    # # prediction_svr = SVRModel(train_x, train_y, test_x) # предсказания модели SVR
    # # prediction_lgb = LGBMModel(train_x, train_y, test_x) # предсказания модели LightGBMRegressor
    # # пример графика
    # # plot_prediction(df[-PRED_LEN + INPUT_LEN:], prediction_lgb.PAY)