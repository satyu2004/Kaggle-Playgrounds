import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse


def pipeline(model_1, model_2, X_train, X_test, y_train, y_test=None, params=None):
  indices_train = X_train['Sex']==0
  indices_test = X_test['Sex']==0

  y_pred = np.zeros(len(X_test))
  # print(f'y_pred shape: {y_pred.shape}, indices_test shape: {indices_test.shape}')

  # Train Model 1
  model_1.fit(X_train[indices_train], y_train[indices_train])
  y_pred_1 = model_1.predict(X_test[indices_test])

  # Train Model 2
  model_2.fit(X_train[~indices_train], y_train[~indices_train])
  y_pred_2 = model_2.predict(X_test[~indices_test])

  # Combine Predictions
  y_pred[indices_test] = y_pred_1
  y_pred[~indices_test] = y_pred_2

  if y_test is not None:
    model_1_rmse = np.sqrt(mse(y_test[indices_test], y_pred_1))
    model_2_rmse = np.sqrt(mse(y_test[~indices_test], y_pred_2))
    combined_rmse = np.sqrt(mse(y_test, y_pred))
    print(f'Model 1 RMSE: {model_1_rmse}, Model 2 RMSE: {model_2_rmse}, Combined RMSE: {combined_rmse}')
    return model_1_rmse, model_2_rmse, combined_rmse


  return y_pred

def crossval(model_1,  model_2, X_train, y_train, k=5):
  kf = KFold(n_splits=k, shuffle=True, random_state=42)
  rmse_scores = []
  for i, (train_index, val_index) in enumerate(kf.split(X_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    rmse_scores.append(pipeline(model_1,  model_2, X_train_fold, X_val_fold, y_train_fold, y_val_fold))
  print(f'Mean RMSE: {np.mean(rmse_scores, axis=0)}')
