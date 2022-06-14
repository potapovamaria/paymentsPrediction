import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn import svm
from lssvr import *


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

def plot_prediction(df, prediction, START_DATE, PRED_COLUMN):
    plt.figure(figsize=(25, 12)) # создание фигуры 25 на 12
    plt.plot(df.index[df.index>=START_DATE], prediction, label="prediction", alpha=.7) # строим график x - даты(начиная со стартовой даты), y - предсказания, имя графика - prediction, alpha - коэффициент, отвечающий за прозрачность графика
    plt.plot(df.index[df.index>=START_DATE], df.loc[df.index>=START_DATE, PRED_COLUMN], label="real", alpha=.7) # строим график x - даты(начиная со стартовой даты), y - , имя графика - prediction, alpha - коэффициент, отвечающий за прозрачность графика

    plt.scatter(df.index[df.index>=START_DATE], prediction, label="prediction", alpha=.7)
    plt.scatter(df.index[df.index>=START_DATE], df.loc[df.index>=START_DATE, PRED_COLUMN], label="real", alpha=.7)
    plt.legend()
    plt.title(PRED_COLUMN + " Prediction")
    plt.show()

def XGBoostModel(train_x, train_y):
    model = xgboost.XGBRegressor(booster='gblinear', learning_rate=0.2111, random_state=42, n_estimators=197)
    model.fit(train_x, train_y)

    return model

def LSSVRModel(train_x, train_y):
    lsclf = LSSVR(kernel='rbf',
                  C=10,
                  gamma=0.04)
    train_ = np.reshape(train_y, train_y.shape[0])
    lsclf.fit(train_x, train_)

    return lsclf

def SVRModel(train_x, train_y):
    clf = svm.SVR(kernel='rbf',
                  gamma=0.1,
                  C=0.5,
                  verbose=False,
                  tol=1e-10,
                  epsilon=0.0411)
    clf.fit(train_x, train_y)

    return clf

def LGBMModel(train_x, train_y):
    model_lgb = lgb.LGBMRegressor(learning_rate=0.071, max_depth=11, n_estimators=132, boosting_type='goss',
                                  num_leaves=25, random_state=42)
    model_lgb.fit(train_x, train_y)

    return model_lgb


def get_picks(file, START_DATE, INPUT_LEN, PRED_LEN):
    dff = pd.read_csv(file, sep=';')
    dff['PAY'] = dff['PAY'].apply(lambda x: float(x.replace(',', '.')))
    dff['PAY'] = dff['PAY'] / 1000
    dff['Date'] = pd.to_datetime(dff['PAY_DATE'], format='%d.%m.%Y')
    dff['Month'] = dff['Date'].dt.month
    dff['Year'] = dff['Date'].dt.year
    dff['WeekDay'] = dff['Date'].dt.day_of_week
    dff = dff.sort_values(by='Date')
    idx = dff.groupby(['Month', 'Year'])['PAY'].transform(max) == dff['PAY']

    dff = dff.set_index(pd.DatetimeIndex(dff['Date']))

    df_peak = pd.DataFrame()
    grouped = dff.groupby(['Year', 'Month'])
    for condition, group in grouped:
        p = np.percentile(group.PAY, 95)
        df_peak = pd.concat([df_peak, group[group.PAY > p]], axis=0)
    dfq = pd.DataFrame({'Date': pd.date_range(start='2012-01-01', periods=len(dff), freq='D'), 'PAY': range(len(dff))})
    dfq.PAY = np.nan

    for i in df_peak.index:
        dfq.loc[dfq[dfq.Date == i].index, 'PAY'] = df_peak.loc[i].PAY

    dfq.loc[0, 'PAY'] = dfq[:31]['PAY'].min()

    # Интерполяция временного ряда
    dfi = dfq.copy()
    dfi['PAY'] = dfi['PAY'].interpolate(method='linear', order=5)  # 95% 90%

    # Новый интерполированый ряд с пиками изначального ряда
    dfi = dfi.set_index(pd.DatetimeIndex(dfi.Date))
    dfi = dfi.drop(['Date'], axis=1)

    scaler_pick = MinMaxScaler()
    df_scal_pick = scaler_pick.fit_transform(dfi)
    df_scal_pick = pd.DataFrame(df_scal_pick, columns=dfi.columns)

    START_DATE = START_DATE - datetime.timedelta(days=180
                                                 )
    test_pick = df_scal_pick[dfi.index >= START_DATE]
    train_pick = df_scal_pick[dfi.index < START_DATE]

    train_x_pick = []
    for i in range(INPUT_LEN, len(train_pick)):
        train_x_pick.append(train_pick.iloc[i - INPUT_LEN:i])

    train_y_pick = train_pick.iloc[INPUT_LEN:]
    train_x_pick = np.array(train_x_pick)
    train_y_pick = np.array(train_y_pick)
    train_x_pick = train_x_pick[:, :, 0]
    train_y_pick = train_y_pick[:, 0]

    test_x_pick = []
    for i in range(INPUT_LEN, len(test_pick)):
        test_x_pick.append(test_pick.iloc[i - INPUT_LEN:i])

    test_y_pick = test_pick.iloc[INPUT_LEN:]
    test_x_pick = np.array(test_x_pick)
    test_y_pick = np.array(test_y_pick)
    test_x_pick = test_x_pick[:, :, 0]
    test_y_pick = test_y_pick[:, 0]

    lsclf_pick = LSSVR(kernel='linear', C=2, gamma=0.001)
    train_ = np.reshape(train_y_pick, train_y_pick.shape[0])

    lsclf_pick.fit(train_x_pick, train_)

    y_pred_lsc_pick = lsclf_pick.predict(test_x_pick)

    temp_pick = df_scal_pick[-PRED_LEN:]
    temp_pick.PAY = y_pred_lsc_pick
    prediction_lsc_pick = pd.DataFrame(scaler_pick.inverse_transform(temp_pick), columns=df_scal_pick.columns) * 1000

    return prediction_lsc_pick, df_peak


def get_answer(file, num_model, date_1, date_2, pick_check):
    df = make_data(file)
    LAST_REAL = df.index[-1]
    date_1 = pd.to_datetime(date_1, format='%d.%m.%Y')
    date_2 = pd.to_datetime(date_2, format='%d.%m.%Y')

    START_DATE = date_1
    END_DATE = date_2
    PRED_LEN = (LAST_REAL - START_DATE).days + 1
    INPUT_LEN = 180

    scaler = MinMaxScaler()
    df_scal = scaler.fit_transform(df)
    df_scal = pd.DataFrame(df_scal, columns=df.columns)

    train = df_scal[df.index <= LAST_REAL]

    train_x = []
    for i in range(INPUT_LEN, len(train)):
        train_x.append(train.iloc[i - INPUT_LEN:i])
    train_y = train.iloc[INPUT_LEN:]
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))

    if PRED_LEN > 0:
        PRED_LEN_NEW = (END_DATE - LAST_REAL).days
    else:
        PRED_LEN_NEW = (END_DATE - START_DATE).days + 1

    if num_model == 1:
        model = LSSVRModel(train_x, train_y)
        if PRED_LEN > 0 and PRED_LEN_NEW > 0:
            test_x_copy = train_x[-PRED_LEN].copy()

            y_pred_lsc = []
            for i in range(PRED_LEN):
                y_pred_lsc.append(model.predict([test_x_copy]))
                for k in range(1, len(test_x_copy)):
                    test_x_copy[k - 1] = test_x_copy[k]
                test_x_copy[-1] = y_pred_lsc[-1]

            y_pred = []
            for i in range(len(y_pred_lsc)):
                y_pred.append(y_pred_lsc[i][0])

            temp = df_scal[-PRED_LEN:]
            temp.PAY = y_pred
            prediction_lssvr = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
            prediction_lssvr["PAY"] = prediction_lssvr["PAY"] * 1000

            indexes = pd.DatetimeIndex(df.index[-PRED_LEN:])
            indexes = indexes.strftime('%d.%m.%Y')
            prediction_lssvr = prediction_lssvr.set_index(indexes)

            if pick_check == 2:
                prediction_lsc_pick, df_peak = get_picks(file, START_DATE, INPUT_LEN, PRED_LEN)
                prediction_lsc_pick = prediction_lsc_pick.set_index(pd.DatetimeIndex(df.index[-PRED_LEN:]))
                prediction_lsc_copy = prediction_lssvr.copy()
                df_copy = df_peak.index[df_peak.index >= START_DATE].copy()

                for i in range(len(prediction_lssvr)):
                    for j in range(len(df_copy)):
                        if (prediction_lssvr.index[i:i + 1]) == (df_copy[j:j + 1]):
                            prediction_lsc_copy.PAY[i:i + 1] = prediction_lsc_pick.PAY[i:i + 1]
                prediction_lssvr = prediction_lsc_copy.copy()

            test_pred = train_x[-1].copy()
            for i in range(1, len(test_pred)):
                test_pred[i - 1] = test_pred[i]
            test_pred[len(test_pred) - 1] = train_y[-1]
            y_pred_new = []

            for i in range(PRED_LEN_NEW):
                y_pred_new.append(model.predict([test_pred]))
                for k in range(1, len(test_pred)):
                    test_pred[k - 1] = test_pred[k]
                test_pred[-1] = y_pred_new[-1]

            y_pred = []
            for i in range(len(y_pred_new)):
                y_pred.append(y_pred_new[i][0])

            temp = df_scal[:PRED_LEN_NEW]
            temp.PAY = y_pred
            prediction_lssvr_new = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
            prediction_lssvr_new["PAY"] = prediction_lssvr_new["PAY"] * 1000

            start_date = LAST_REAL + datetime.timedelta(days=1)
            end_date = END_DATE

            res = pd.date_range(
                min(start_date, end_date),
                max(start_date, end_date)
            )

            indexes = pd.DatetimeIndex(res)
            indexes = indexes.strftime('%d.%m.%Y')
            prediction_lssvr_new = prediction_lssvr_new.set_index(indexes)

            prediction_lssvr = prediction_lssvr.append(prediction_lssvr_new)
            prediction_lssvr["PAY"] = round(prediction_lssvr["PAY"], 2)

            return prediction_lssvr
        elif (PRED_LEN > 0) and (PRED_LEN_NEW <= 0):
            N = (LAST_REAL - START_DATE).days
            N_END = (LAST_REAL - END_DATE).days
            test_x_copy = train_x[-N - 1].copy()
            y_pred_lsc = []
            for i in range(PRED_LEN - N_END):
                y_pred_lsc.append(model.predict([test_x_copy]))
                for k in range(1, len(test_x_copy)):
                    test_x_copy[k - 1] = test_x_copy[k]
                test_x_copy[-1] = y_pred_lsc[-1]

            y_pred = []
            for i in range(len(y_pred_lsc)):
                y_pred.append(y_pred_lsc[i][0])

            if N_END != 0:
                temp = df_scal[-N - 1:-N_END]
                temp.PAY = y_pred
                prediction_lssvr = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)

                prediction_lssvr["PAY"] = prediction_lssvr["PAY"] * 1000

                indexes = pd.DatetimeIndex(df.index[-N - 1:-N_END])
                indexes = indexes.strftime('%d.%m.%Y')
                prediction_lssvr = prediction_lssvr.set_index(indexes)
            else:
                temp = df_scal[-N - 1:]
                temp.PAY = y_pred
                prediction_lssvr = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)

                prediction_lssvr["PAY"] = prediction_lssvr["PAY"] * 1000

                indexes = pd.DatetimeIndex(df.index[-N - 1:])
                indexes = indexes.strftime('%d.%m.%Y')
                prediction_lssvr = prediction_lssvr.set_index(indexes)

            if pick_check == 2:
                prediction_lsc_pick, df_peak = get_picks(file, START_DATE, INPUT_LEN, PRED_LEN)
                prediction_lsc_pick = prediction_lsc_pick.set_index(pd.DatetimeIndex(df.index[-PRED_LEN:]))
                prediction_lsc_copy = prediction_lssvr.copy()
                df_copy = df_peak.index[df_peak.index >= START_DATE].copy()

                for i in range(len(prediction_lssvr)):
                    for j in range(len(df_copy)):
                        if (prediction_lssvr.index[i:i + 1]) == (df_copy[j:j + 1]):
                            prediction_lsc_copy.PAY[i:i + 1] = prediction_lsc_pick.PAY[i:i + 1]
                prediction_lssvr = prediction_lsc_copy.copy()
            prediction_lssvr["PAY"] = round(prediction_lssvr["PAY"], 2)

            return prediction_lssvr
        elif (PRED_LEN <= 0) and (PRED_LEN_NEW > 0) and START_DATE == (LAST_REAL + datetime.timedelta(days=1)):
            test_pred = train_x[-1].copy()
            for i in range(1, len(test_pred)):
                test_pred[i - 1] = test_pred[i]
            test_pred[len(test_pred) - 1] = train_y[-1]
            y_pred_new = []

            for i in range(PRED_LEN_NEW):
                y_pred_new.append(model.predict([test_pred]))
                for k in range(1, len(test_pred)):
                    test_pred[k - 1] = test_pred[k]
                test_pred[-1] = y_pred_new[-1]

            y_pred = []
            for i in range(len(y_pred_new)):
                y_pred.append(y_pred_new[i][0])

            temp = df_scal[-PRED_LEN_NEW:]
            temp.PAY = y_pred
            prediction_lssvr = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
            prediction_lssvr["PAY"] = prediction_lssvr["PAY"] * 1000

            start_date = LAST_REAL + datetime.timedelta(days=1)
            end_date = END_DATE

            res = pd.date_range(
                min(start_date, end_date),
                max(start_date, end_date)
            )

            indexes = pd.DatetimeIndex(res)
            indexes = indexes.strftime('%d.%m.%Y')
            prediction_lssvr = prediction_lssvr.set_index(indexes)
            prediction_lssvr["PAY"] = round(prediction_lssvr["PAY"], 2)

            return prediction_lssvr
        else:
            N = (END_DATE - LAST_REAL).days
            test_pred = train_x[-1].copy()
            for i in range(1, len(test_pred)):
                test_pred[i - 1] = test_pred[i]
            test_pred[len(test_pred) - 1] = train_y[-1]
            y_pred_new = []

            for i in range(N):
                y_pred_new.append(model.predict([test_pred]))
                for k in range(1, len(test_pred)):
                    test_pred[k - 1] = test_pred[k]
                test_pred[-1] = y_pred_new[-1]

            y_pred = []
            for i in range(len(y_pred_new)):
                y_pred.append(y_pred_new[i][0])

            temp = df_scal[-N:]
            temp.PAY = y_pred
            prediction_lssvr = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
            prediction_lssvr["PAY"] = prediction_lssvr["PAY"] * 1000

            start_date = LAST_REAL + datetime.timedelta(days=1)
            end_date = END_DATE

            res = pd.date_range(
                min(start_date, end_date),
                max(start_date, end_date)
            )

            indexes = pd.DatetimeIndex(res)
            indexes = indexes.strftime('%d.%m.%Y')
            prediction_lssvr = prediction_lssvr.set_index(indexes)
            prediction_lssvr["PAY"] = round(prediction_lssvr["PAY"], 2)

            N_END = (START_DATE - LAST_REAL).days
            return prediction_lssvr[N_END - 1:]

    #
    # if num_model == 2:
    #     model = XGBoostModel(train_x, train_y)
    #     y_pred_xgb = []
    #
    #     test_x_copy = test_x.copy()
    #     for i in range(PRED_LEN - INPUT_LEN):
    #         if len(y_pred_xgb) < INPUT_LEN:
    #             for j in range(1, len(y_pred_xgb) + 1):
    #                 test_x_copy[i][-j] = y_pred_xgb[-j].copy()
    #         else:
    #             test_x_copy[i] = y_pred_xgb[-INPUT_LEN:].copy()
    #         y_pred_xgb.append(model.predict([test_x_copy[i]]))
    #     y_pred = []
    #     for i in range(len(y_pred_xgb)):
    #         y_pred.append(y_pred_xgb[i][0])
    #
    #     temp = df_scal[-PRED_LEN + INPUT_LEN:]
    #     temp.PAY = y_pred
    #     prediction_xgb = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
    #     prediction_xgb["PAY"] = prediction_xgb["PAY"] * 1000
    #
    #     indexes = pd.DatetimeIndex(df.index[-PRED_LEN + INPUT_LEN:])
    #     indexes = indexes.strftime('%d.%m.%Y')
    #     prediction_xgb = prediction_xgb.set_index(indexes)
    #     return prediction_xgb
    # if num_model == 3:
    #     model = SVRModel(train_x, train_y)
    #
    #     y_pred_svr = []
    #     test_x_copy = test_x.copy()
    #     for i in range(PRED_LEN - INPUT_LEN):
    #         if len(y_pred_svr) < INPUT_LEN:
    #             for j in range(1, len(y_pred_svr) + 1):
    #                 test_x_copy[i][-j] = y_pred_svr[-j].copy()
    #         else:
    #             test_x_copy[i] = y_pred_svr[-INPUT_LEN:].copy()
    #         y_pred_svr.append(model.predict([test_x_copy[i]]))
    #     y_pred = []
    #     for i in range(len(y_pred_svr)):
    #         y_pred.append(y_pred_svr[i][0])
    #
    #     temp = df_scal[-PRED_LEN + INPUT_LEN:]
    #     temp.PAY = y_pred
    #     prediction_svr = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
    #     prediction_svr["PAY"] = prediction_svr["PAY"] * 1000
    #     indexes = pd.DatetimeIndex(df.index[-PRED_LEN + INPUT_LEN:])
    #     indexes = indexes.strftime('%d.%m.%Y')
    #     prediction_svr = prediction_svr.set_index(indexes)
    #     return prediction_svr
    # if num_model == 4:
    #     model = LGBMModel(train_x, train_y)
    #     y_pred_lgb = []
    #     test_x_copy = test_x.copy()
    #     for i in range(PRED_LEN - INPUT_LEN):
    #         if len(y_pred_lgb) < INPUT_LEN:
    #             for j in range(1, len(y_pred_lgb) + 1):
    #                 test_x_copy[i][-j] = y_pred_lgb[-j].copy()
    #         else:
    #             test_x_copy[i] = y_pred_lgb[-INPUT_LEN:].copy()
    #         y_pred_lgb.append(model.predict([test_x_copy[i]]))
    #     y_pred = []
    #     for i in range(len(y_pred_lgb)):
    #         y_pred.append(y_pred_lgb[i][0])
    #
    #     temp = df_scal[-PRED_LEN + INPUT_LEN:]
    #     temp.PAY = y_pred
    #     prediction_lgb = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
    #     prediction_lgb["PAY"] = prediction_lgb["PAY"] * 1000
    #     indexes = pd.DatetimeIndex(df.index[-PRED_LEN + INPUT_LEN:])
    #     indexes = indexes.strftime('%d.%m.%Y')
    #     prediction_lgb = prediction_lgb.set_index(indexes)
    #     return prediction_lgb

if __name__ == '__main__':
    y_pred = get_answer('pay2021-11-24.csv', 1, '01.11.2016', '24.11.2016', 1)
    # plt.figure(figsize=(25, 12)) # создание фигуры 25 на 12
    # plt.plot(y_pred.index, y_pred.PAY, label="prediction", alpha=.7) # строим график x - даты(начиная со стартовой даты), y - предсказания, имя графика - prediction, alpha - коэффициент, отвечающий за прозрачность графика
    # plt.show()
    print(y_pred)