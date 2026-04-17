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
y_aero = aero["stability_index"]
X_aero = aero.drop(columns=["stability_index"])

#train test split
X_train, X_test, y_train, y_test = train_test_split(X_aero, y_aero, test_size=0.2, random_state=1)


#random forest things TM
rf_aero = RFR(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=1
)
rf_aero.fit(X_train, y_train)

#evaluations and feature importance
#add log loss, oob and whatnot
rf_aero.score(X_test, y_test)
importances = rf_aero.feature_importances_
