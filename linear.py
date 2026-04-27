import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from sklearn.model_selection import train_test_split 
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize)
from sklearn.preprocessing import StandardScaler

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
Weather = pd.read_csv("data/Weather.csv")

Weather.dtypes

# Create Race ID

Weather["RaceID"] = (
    Weather["Year"].astype(str) + "_" +
    Weather["Round Number"].astype(str)
)


# Split by race


race_ids = Weather["RaceID"].unique()

split_idx = int(len(race_ids) * 0.8)

train_races = race_ids[:split_idx]
test_races = race_ids[split_idx:]

Train = Weather[Weather["RaceID"].isin(train_races)].copy()
Test = Weather[Weather["RaceID"].isin(test_races)].copy()

#Define variables

indep_vars = ["AirTemp", "Humidity", "WindSpeed"]
target = "TrackTemp"

X_train = Train[indep_vars]
y_train = Train[target]

X_test = Test[indep_vars]
y_test = Test[target]

# Add nonlinear effects

temp_knot = X_train["AirTemp"].median()
humidity_knot = X_train["Humidity"].median()

def add_features(df):
    df = df.copy()

    df["temp_cubicspline"] = np.maximum(0, df["AirTemp"] - temp_knot)**3
    df["humidity_cubicspline"] = np.maximum(0, df["Humidity"] - humidity_knot)**3

    df["temp_windspeed"] = df["AirTemp"] * df["WindSpeed"]
    df["temp_humidity"] = df["AirTemp"] * df["Humidity"]

    return df


X_train_fe = add_features(X_train)
X_test_fe = add_features(X_test)

#Scale the data with non linear effects as they sometimes get out of control

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_fe),
    columns=X_train_fe.columns,
    index=X_train_fe.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_fe),
    columns=X_test_fe.columns,
    index=X_test_fe.index
)


# Add intercepts

X_train_base = sm.add_constant(X_train)
X_test_base = sm.add_constant(X_test)

X_train_full = sm.add_constant(X_train_scaled)
X_test_full = sm.add_constant(X_test_scaled)

#Construct models

model_1 = sm.OLS(y_train, X_train_base)
results_1 = model_1.fit()
print(summarize(results_1))

model_2 = sm.OLS(y_train, X_train_full)
results_2 = model_2.fit()
print(summarize(results_2))

#Get predictions

predictions_train_1 = predict(X_train_base,results_1)
predictions_test_1 = predict(X_test_base, results_1)
predictions_train_2 = predict(X_train_full,results_2)
predictions_test_2 = predict(X_test_full, results_2)

#Calculate MSE

print("Model 1 (Raw Linear)")
print('MSE train: ', mse(y_train, predictions_train_1))
print('MSE test: ', mse(y_test, predictions_test_1))

print("Model 2 (Linear with Nonlinear Effects + Scaling)")
print('MSE train: ', mse(y_train, predictions_train_2))
print('MSE test: ', mse(y_test, predictions_test_2))

#Show some plots

plot_avg_actual_vs_predicted_by_feature(X=X_train, 
                                        y=y_train, 
                                        y_pred=predictions_train_1, 
                                        feature="AirTemp", 
                                        target_name="TrackTemp"
                                        )

plot_avg_actual_vs_predicted_by_feature(X=X_train_fe, 
                                        y=y_train, 
                                        y_pred=predictions_train_2, 
                                        feature="AirTemp", 
                                        target_name="TrackTemp"
                                        )

