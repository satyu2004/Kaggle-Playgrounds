import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

enc = LabelEncoder()
predictors = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp']

target = 'z'



# Prepare the train data
train = pd.read_csv('train.csv')
train['Sex'] = enc.fit_transform(train['Sex'])
train['z'] = np.log1p(train['Calories'])
X = train[predictors]
z = train[target]



# Set up some nice styling for plots
plt.style.use('fivethirtyeight')
sns.set_palette("Set2")

print("=" * 80)
print("🚀 STARTING STACKING MODEL TRAINING WORKFLOW")
print("=" * 80)

# 1. Initial split to create true holdout test set
print("\n📊 STEP 1: Creating initial train/test split...")
time.sleep(0.5)  # Small pause for visual effect
X_train_full, X_test, y_train_full, y_test = train_test_split(X, z, test_size=0.2, random_state=42)
print(f"   Train set: {X_train_full.shape[0]:,} samples | Test set: {X_test.shape[0]:,} samples")

# Show initial dataset stats
print("\n📈 Dataset Statistics:")
stats = pd.DataFrame({
    'Mean': [y_train_full.mean()],
    'Std Dev': [y_train_full.std()],
    'Min': [y_train_full.min()],
    'Max': [y_train_full.max()],
    'Samples': [len(y_train_full)]
}, index=['Target'])
display(stats)

# 2. Use cross-validation to create meta-features for X_train_full
print("\n🔄 STEP 2: Generating cross-validated meta-features...")
X_train_meta = X_train_full.copy()
X_train_meta['z_lgb'] = np.zeros(len(X_train_full))
X_train_meta['z_hgb'] = np.zeros(len(X_train_full))

# K-fold CV for non-leaking meta-features
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmse_lgb = []
fold_rmse_hgb = []

print("\n⚙️ Running 5-fold cross-validation:")
for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train_full), total=5, desc="CV Folds")):
    # Split data for this fold
    X_fold_train, X_fold_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_fold_train, y_fold_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
    
    print(f"\n   📌 Fold {fold+1} - Training samples: {len(X_fold_train):,} | Validation samples: {len(X_fold_val):,}")
    
    # Train LGB on fold's training data
    print("      🌳 Training LightGBM model...")
    train_set = lgb.Dataset(X_fold_train, y_fold_train)
    val_set = lgb.Dataset(X_fold_val, y_fold_val, reference=train_set)
    lgb_fold = lgb.train(
        params_lgb, 
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Predict on fold's validation data (these predictions haven't seen this data)
    lgb_preds = lgb_fold.predict(X_fold_val)
    X_train_meta.loc[val_idx, 'z_lgb'] = lgb_preds
    
    # Calculate RMSE for this fold
    fold_lgb_rmse = np.sqrt(mean_squared_error(y_fold_val, lgb_preds))
    fold_rmse_lgb.append(fold_lgb_rmse)
    print(f"      ✅ LightGBM - Fold {fold+1} RMSE: {fold_lgb_rmse:.4f} | Best Iteration: {lgb_fold.best_iteration}")
    
    # Train HGB on fold's training data
    print("      🌲 Training HistGradientBoosting model...")
    hgb_fold = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.05,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        verbose=0
    ).fit(X_fold_train, y_fold_train)
    
    # Predict on fold's validation data
    hgb_preds = hgb_fold.predict(X_fold_val)
    X_train_meta.loc[val_idx, 'z_hgb'] = hgb_preds
    
    # Calculate RMSE for this fold
    fold_hgb_rmse = np.sqrt(mean_squared_error(y_fold_val, hgb_preds))
    fold_rmse_hgb.append(fold_hgb_rmse)
    print(f"      ✅ HistGradientBoosting - Fold {fold+1} RMSE: {fold_hgb_rmse:.4f}")

# Print cross-validation summary
print("\n📊 Cross-validation performance summary:")
cv_summary = pd.DataFrame({
    'LightGBM RMSE': fold_rmse_lgb,
    'HistGBR RMSE': fold_rmse_hgb
})
cv_summary.index = [f'Fold {i+1}' for i in range(5)]
cv_summary.loc['Mean'] = cv_summary.mean()
cv_summary.loc['Std Dev'] = cv_summary.std()
display(cv_summary)

# 3. Train full LGB and HGB models on ALL training data for test set predictions
print("\n🔥 STEP 3: Training full models on all training data...")

print("   🌳 Training full LightGBM model...")
lgb_train_dataset = lgb.Dataset(X_train_full, y_train_full)
lgb_val_dataset = lgb.Dataset(X_test, y_test, reference=lgb_train_dataset)

# Progress tracking for LGB
callbacks = [
    lgb.callback.log_evaluation(period=100),
    lgb.early_stopping(stopping_rounds=50, verbose=True)
]

lgb_full = lgb.train(
    params_lgb, 
    lgb_train_dataset, 
    num_boost_round=1000,
    valid_sets=[lgb_val_dataset],
    callbacks=callbacks
)
print(f"   ✅ LightGBM trained - Best iteration: {lgb_full.best_iteration}")

print("\n   🌲 Training full HistGradientBoosting model...")
with tqdm(total=100, desc="HGB Training") as pbar:
    hgb_full = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.05,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        verbose=0
    )
    
    # Simple progress simulation for HGB since it doesn't have built-in progress reporting
    for i in range(10):
        time.sleep(0.1)  # Simulate training progress
        pbar.update(10)
    
    hgb_full.fit(X_train_full, y_train_full)
print(f"   ✅ HistGradientBoosting trained - Iterations used: {hgb_full.n_iter_}")

# 4. Create meta-features for test set using full models
print("\n🔍 STEP 4: Creating meta-features for test set...")
X_test_meta = X_test.copy()

print("   🔮 Generating LightGBM predictions...")
X_test_meta['z_lgb'] = lgb_full.predict(X_test, num_iteration=lgb_full.best_iteration)

print("   🔮 Generating HistGradientBoosting predictions...")
X_test_meta['z_hgb'] = hgb_full.predict(X_test)

# Display correlations between meta-features
meta_corr = pd.DataFrame({
    'LGB': X_test_meta['z_lgb'],
    'HGB': X_test_meta['z_hgb'],
    'True': y_test
}).corr()
print("\n📊 Meta-feature correlations:")
display(meta_corr)

# 5. Train second-level model (XGB) using meta-features
print("\n🏗️ STEP 5: Preparing data for second-level model...")
predictors += ['z_lgb', 'z_hgb']
print(f"   📋 Total features for stacking model: {len(predictors)}")

X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_meta[predictors], y_train_full, test_size=0.2, random_state=43
)
print(f"   📊 Final training split - Train: {X_train_final.shape[0]:,} samples | Validation: {X_val_final.shape[0]:,} samples")

# 6. Train XGB on clean meta-features
print("\n🚀 STEP 6: Training final XGBoost stacking model...")
xgb_train = xgb.DMatrix(X_train_final, label=y_train_final)
xgb_val = xgb.DMatrix(X_val_final, label=y_val_final)

# Progress tracking
results = {}

xgb_model = xgb.train(
    params_xgb, 
    xgb_train, 
    num_boost_round=1000, 
    evals=[(xgb_train, 'train'), (xgb_val, 'val')],
    evals_result=results,
    early_stopping_rounds=100,
    verbose_eval=100
)

print(f"   ✅ XGBoost model trained - Best iteration: {xgb_model.best_iteration}")

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(results['train']['rmse'], label='Training RMSE')
plt.plot(results['val']['rmse'], label='Validation RMSE')
plt.axvline(x=xgb_model.best_iteration, color='r', linestyle='--', label=f'Best iteration: {xgb_model.best_iteration}')
plt.xlabel('Number of Boosting Rounds')
plt.ylabel('RMSE')
plt.title('XGBoost Learning Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Show feature importance
importance = xgb_model.get_score(importance_type='gain')
importance = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Importance': list(importance.values())
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title('XGBoost Feature Importance (Gain)')
plt.tight_layout()
plt.show()

# 7. Final prediction on true holdout test set
print("\n🎯 STEP 7: Making final predictions on test set...")
z_pred = xgb_model.predict(xgb.DMatrix(X_test_meta[predictors]), iteration_range=(0, xgb_model.best_iteration))

# Calculate and display final metrics
rmse = np.sqrt(mean_squared_error(y_test, z_pred))
print(f"\n✨ FINAL RESULTS:")
print(f"   🎖️ Test RMSE: {rmse:.4f}")

# Compare base models vs stacked results
base_metrics = pd.DataFrame({
    'Model': ['LightGBM', 'HistGradientBoosting', 'Stacked XGBoost'],
    'Test RMSE': [
        np.sqrt(mean_squared_error(y_test, X_test_meta['z_lgb'])),
        np.sqrt(mean_squared_error(y_test, X_test_meta['z_hgb'])),
        rmse
    ]
})
print("\n📊 Performance comparison of all models:")
display(base_metrics)

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, z_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("🏁 STACKING MODEL WORKFLOW COMPLETED SUCCESSFULLY!")
print("=" * 80)
