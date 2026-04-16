import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from sklearn.model_selection import train_test_split 
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize)

Aero = load_data('Aero')
Weather = load_data('Weather')

Aero.dtypes
Weather.dtypes

X_train, X_test, y_train, y_test, Train, Test = train_test_split(X, y, Aero, 
                                                                 test_size = 0.25, 
                                                                 shuffle = True)

X_train, X_test, y_train, y_test, Train, Test = train_test_split(X, y, Weather, 
                                                                 test_size = 0.25, 
                                                                 shuffle = True)