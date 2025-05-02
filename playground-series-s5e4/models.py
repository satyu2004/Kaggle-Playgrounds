import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np


def xgb_cross_val_regression(X_train, y_train, params=None, n_folds=5, num_boost_round=100, early_stopping_rounds=10, seed=42):
    """
    Performs k-fold cross-validation with XGBoost for a regression task.

    Args:
        X_train (pd.DataFrame or numpy.ndarray): Training features.
        y_train (pd.Series or numpy.ndarray): Training target variable.
        params (dict, optional): XGBoost parameters. Defaults to a basic set.
        n_folds (int): Number of folds for cross-validation.
        num_boost_round (int): Maximum number of boosting rounds.
        early_stopping_rounds (int): Stops training early if performance doesn't improve.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing:
            - 'cv_results': Pandas DataFrame of cross-validation results.
            - 'oof_predictions': Numpy array of out-of-fold predictions.
            - 'mean_rmse': Mean RMSE across all folds.
            - 'std_rmse': Standard deviation of RMSE across all folds.
            - 'best_n_estimators': The average best number of estimators from each fold.
    """

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_predictions = np.zeros(len(y_train))
    rmse_scores = []
    best_n_estimators_list = []

    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.1,
            'seed': seed
        }

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        print(f"Fold {fold+1}")
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

        cv_results_fold = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )

        best_n_estimators_fold = cv_results_fold.best_iteration + 1
        best_n_estimators_list.append(best_n_estimators_fold)

        y_pred_val = cv_results_fold.predict(dval, iteration_range=(0, cv_results_fold.best_iteration))
        oof_predictions[val_index] = y_pred_val
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
        rmse_scores.append(rmse)
        print(f"Fold {fold+1} RMSE: {rmse}, Best n_estimators: {best_n_estimators_fold}")

    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    avg_best_n_estimators = int(np.mean(best_n_estimators_list))

    print(f"\nMean Cross-Validation RMSE: {mean_rmse:.4f}")
    print(f"Standard Deviation of RMSE: {std_rmse:.4f}")
    print(f"Average Best n_estimators: {avg_best_n_estimators}")

    cv_results_df = pd.DataFrame({'fold': range(1, n_folds + 1), 'rmse': rmse_scores, 'best_n_estimators': best_n_estimators_list})

    return {
        'cv_results': cv_results_df,
        'oof_predictions': oof_predictions,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'best_n_estimators': avg_best_n_estimators
    }

