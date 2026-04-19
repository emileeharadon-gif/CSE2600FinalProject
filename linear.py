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

Aero = load_data('Aero')

Aero.dtypes
indep_vars = ["drag_n", "downforce_n", "wind_angle_deg", "stability_index"]

X = Aero[indep_vars]
y = Aero["speed_kmh"]

X_train, X_test, y_train, y_test, Train, Test = train_test_split(X, y, Aero, 
                                                                 random_state = 4271, 
                                                                 test_size = 0.25, 
                                                                 shuffle = True)

X_train['intercept'] = np.ones(X_train.shape[0])
X_test['intercept'] = np.ones(X_test.shape[0])

drag_knot = X_train["drag_n"].median()
downforce_knot = X_train["downforce_n"].median()

for df in [Train, Test, X_train, X_test]:
    df["drag_cubicspline"] = np.maximum(0, df["drag_n"] - drag_knot)**3

for df in [Train, Test, X_train, X_test]:
    df["downforce_cubicspline"] = np.maximum(0, df["downforce_n"] - downforce_knot)**3

for df in [Train, Test, X_train, X_test]:
    df["drag_downforce"] = df["downforce_n"] * df["drag_n"]

vars_to_include = ['intercept', indep_vars, "drag_downforce", "drag_cubicspline", "downforce_cubicspline"]

model = sm.OLS(y_train, X_train[[vars_to_include]])
results = model.fit()
summarize(results)

predictions_train_1 = predict(X_train[[vars_to_include]],results)
print('MSE train: ', mse(y_train, predictions_train_1))

predictions_test_1 = predict(X_test[[vars_to_include]], results)
print('MSE test: ', mse(y_test, predictions_test_1))