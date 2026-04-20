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
from sklearn.model_selection import cross_val_score


#read data files
aero = pd.read_csv("data/Aero.csv")
weather = pd.read_csv("data/Weather.csv")


#targets, might change as needed
y_weather = weather["TrackTemp"]
X_weather = weather.drop(columns=["TrackTemp", "Time"])

#train test split
X_train, X_test, y_train, y_test = train_test_split(X_weather, y_weather, test_size=0.2, random_state=1)


#random forest things TM
rf_weather = RFR(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=1
)

#cross validation
scores = cross_val_score(rf_weather, X_weather, y_weather, cv=5)
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
print(importances)