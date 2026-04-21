import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from sklearn.model_selection import train_test_split 
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize)

#Helper functions we will need to assess data

def predict(X, model):
    # the built-in get_prediction tool returns an array, so we need to convert to a dataframe
    predictions_df = pd.DataFrame(model.get_prediction(X).predicted, columns=['y_hat'], index=X.index)
    return predictions_df['y_hat']

def mse(y, y_hat):
    # calculate the residual error for each individual record
    resid = y - y_hat
    # square the residual (hence "squared error")
    sq_resid = resid**2
    # calculate the sum of squared errors
    SSR = sum(sq_resid)
    # divide by the number of records to get the mean squared error
    MSE = SSR / y.shape[0]
    return MSE

def plot_avg_actual_vs_predicted_by_feature(X, y, y_pred, feature, target_name="Target"):
    df = pd.DataFrame({
        feature: X[feature],
        "actual": y,
        "predicted": y_pred
    })

    grouped = df.groupby(feature).mean().reset_index()

    plt.figure()
    plt.plot(grouped[feature], grouped["actual"], label="Average Actual")
    plt.plot(grouped[feature], grouped["predicted"], label="Average Predicted")
    plt.xlabel(feature)
    plt.ylabel(target_name)
    plt.legend()
    plt.show()

#Load data
Weather = load_data('Weather')

Weather.dtypes
indep_vars = ["AirTemp", "Humidity","WindDirection", "WindSpeed"]

X = Weather[indep_vars]
y = Weather["TrackTemp"]

#train test split and cleaning
Weather["Time"] = pd.to_timedelta(Weather["Time"]).dt.total_seconds()

weather = Weather.sort_values(by='Time')

split = int(len(weather)*.8)
Train = weather.iloc[:split]
Test = weather.iloc[split:]

X_train = Train.drop(columns=["TrackTemp", "Round Number", "Year"])
y_train = Train["TrackTemp"]

X_test = Test.drop(columns=["TrackTemp", "Round Number", "Year"])
y_test = Test["TrackTemp"]

X_train['intercept'] = np.ones(X_train.shape[0])
X_test['intercept'] = np.ones(X_test.shape[0])

#Create nonlinear effects 
temp_knot = X_train["AirTemp"].median()
windspeed_knot = X_train["WindSpeed"].median()

for df in [Train, Test, X_train, X_test]:
    df["temp_cubicspline"] = np.maximum(0, df["AirTemp"] - temp_knot)**3

for df in [Train, Test, X_train, X_test]:
    df["windspeed_cubicspline"] = np.maximum(0, df["WindSpeed"] - windspeed_knot)**3

for df in [Train, Test, X_train, X_test]:
    df["temp_windspeed"] = df["WindSpeed"] * df["AirTemp"]

for df in [Train, Test, X_train, X_test]:
    df["windspeed_direction"] = df["WindSpeed"] * df["WindDirection"]

for df in [Train, Test, X_train, X_test]:
    df["temp_humidity"] = df["Humidity"] * df["AirTemp"]

vars_to_include_1 = ['intercept', indep_vars]

vars_to_include_2 = ['intercept', indep_vars, "temp_windspeed", "windspeed_direction", "temp_humidity", "temp_cubicspline", "windspeed_cubicspline"]

model_1 = sm.OLS(y_train, X_train[[vars_to_include_2]])
results_1 = model_1.fit()
summarize(results_1)

model_nonlinear = sm.OLS(y_train, X_train[[vars_to_include_2]])
results_2 = model_nonlinear.fit()
summarize(results_2)

predictions_train_1 = predict(X_train[[vars_to_include_1]],results_1)
print('MSE train: ', mse(y_train, predictions_train_1))

predictions_test_1 = predict(X_test[[vars_to_include_1]], results_1)
print('MSE test: ', mse(y_test, predictions_test_1))

predictions_train_2 = predict(X_train[[vars_to_include_2]],results_2)
print('MSE train: ', mse(y_train, predictions_train_2))

predictions_test_2 = predict(X_test[[vars_to_include_2]], results_2)
print('MSE test: ', mse(y_test, predictions_test_2))

plot_avg_actual_vs_predicted_by_feature(X=X_train, 
                                        y=y_train, 
                                        y_pred=predictions_train_1, 
                                        feature=vars_to_include_1, 
                                        target_name="TrackTemp"
                                        )

plot_avg_actual_vs_predicted_by_feature(X=X_train, 
                                        y=y_train, 
                                        y_pred=predictions_train_2, 
                                        feature=vars_to_include_2, 
                                        target_name="TrackTemp"
                                        )