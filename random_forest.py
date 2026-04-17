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
from sklearn.metrics import confusion_matrix, classification_report


#read data files
aero = pd.read_csv("data/Aero.csv")
weather = pd.read_csv("data/Weather.csv")


#targets, might change as needed
X_aero = aero.drop(columns=['stability_index'])
y_aero = (aero['stability_index'] == 100).astype(int)

#train test split
X_train, X_test, y_train, y_test = train_test_split(X_aero, y_aero, test_size=0.2, random_state=1, stratify=y_aero)


#random forest things TM
rf_aero = RFC(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    random_state=1)

#cross validation
scores = cross_val_score(rf_aero, X_aero, y_aero, cv=5)
print("Cross validation scores:", scores)
print("Mean cross validation score:", scores.mean())


#fit
rf_aero.fit(X_train, y_train)


#evaluations and feature importance
y_pred = rf_aero.predict(X_test)
print('test accuracy:', rf_aero.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

importances = rf_aero.feature_importances_
feat_imp = pd.Series(importances, index=X_aero.columns)
print(feat_imp.sort_values(ascending=False))