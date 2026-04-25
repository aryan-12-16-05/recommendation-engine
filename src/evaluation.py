import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

ratings = pd.read_csv('../data/ratings.csv')

# Dummy prediction (baseline)
y_true = ratings['rating']
y_pred = np.full_like(y_true, y_true.mean())

rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("Baseline RMSE:", rmse)