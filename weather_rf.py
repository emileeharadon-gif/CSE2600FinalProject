#imports, we might not actually use all of these so delete what doesn't get used at the end
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from sklearn.model_selection import train_test_split 
from sklearn.tree import \
      (DecisionTreeRegressor as DTR, 
       DecisionTreeClassifier as DTC, 
       plot_tree)
from sklearn.ensemble import \
     (RandomForestRegressor as RFR,
      RandomForestClassifier as RFC,
      GradientBoostingRegressor as GBR)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit


#read data files
weather = pd.read_csv("data/Weather.csv")

#train test split and cleaning
weather["Time"] = pd.to_timedelta(weather["Time"]).dt.total_seconds()

weather = weather.sort_values(by='Time')

split = int(len(weather)*.8)
train = weather.iloc[:split]
test = weather.iloc[split:]

X_train = train.drop(columns=["TrackTemp", "Round Number", "Year", "WindDirection"])
y_train = train["TrackTemp"]

X_test = test.drop(columns=["TrackTemp", "Round Number", "Year", "WindDirection"])
y_test = test["TrackTemp"]


#random forest things TM
rf_weather = RFR(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=1
)

#cross validation
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(rf_weather, X_train, y_train, cv=tscv)
print("Cross validation scores:", scores)
print("Mean cross validation score:", scores.mean())


#fit
rf_weather.fit(X_train, y_train)


#evaluations and feature importance
#add log loss, oob and whatnot
rf_weather.score(X_test, y_test)
error = rf_weather.score(X_test, y_test)
importances = rf_weather.feature_importances_


print("test R^2: ",error)
print("importances: ", importances)