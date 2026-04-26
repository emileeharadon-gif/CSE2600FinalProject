import pandas as pd 
from sklearn.ensemble import \
     (RandomForestRegressor as RFR)
from sklearn.model_selection import cross_val_score, GroupKFold, train_test_split


#read data files
weather = pd.read_csv("data/Weather.csv")

#train test split and cleaning
weather["Time"] = pd.to_timedelta(weather["Time"]).dt.total_seconds()
weather = weather.sort_values(by=["Year", "Round Number", "Time"])

weather["Time_norm"] = weather.groupby(
    ["Year", "Round Number"]
)["Time"].transform(lambda x: x / x.max())

weather = weather.dropna()

races = weather[["Year", "Round Number"]].drop_duplicates()
split_idx = int(len(races) * 0.8)

train_races = races.iloc[:split_idx]
test_races  = races.iloc[split_idx:]

train = weather.merge(train_races, on=["Year", "Round Number"])
test  = weather.merge(test_races,  on=["Year", "Round Number"])

X_train = train.drop(columns=["TrackTemp", "WindDirection", "Round Number"])
y_train = train["TrackTemp"]

X_test = test.drop(columns=["TrackTemp", "WindDirection", "Round Number"])
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
groups = train["Round Number"]

gkf = GroupKFold(n_splits=5)
scores = cross_val_score(
    rf_weather,
    X_train,
    y_train,
    cv=gkf,
    groups=groups
)
print("Cross validation scores:", scores)
print("Mean cross validation score:", scores.mean())


#fit
rf_weather.fit(X_train, y_train)


#evaluations and feature importance
#add log loss, oob and whatnot
error = rf_weather.score(X_test, y_test)
importances = rf_weather.feature_importances_


print("test R^2: ",error)
print("importances: ", importances)