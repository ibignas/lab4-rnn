# ============================================================
#  REGRESSION LAB — California Housing Prices (Kaggle CSV)
#  Target: median_house_value
#  Dataset: https://www.kaggle.com/datasets/camnugent/california-housing-prices
# ============================================================

# ── 0. IMPORTS ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, os, warnings
warnings.filterwarnings('ignore')

sns.set_style('darkgrid')
os.makedirs('models',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)


# ── 1. DATA ACQUISITION & EXPLORATION ────────────────────────────────────────
df = pd.read_csv('housing.csv')          # place housing.csv in the same folder

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape          : {df.shape}")
print(f"Columns        : {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nDescriptive statistics:\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nocean_proximity value counts:\n{df['ocean_proximity'].value_counts()}")


# ── 2. DATA PREPROCESSING ────────────────────────────────────────────────────

# 2a. Remove rows where target is NaN
df.dropna(subset=['median_house_value'], inplace=True)

# 2b. Impute missing numeric values with column median
for col in df.select_dtypes(include='number').columns:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Imputed '{col}' with median = {median_val:.2f}")

# 2c. Remove extreme outliers (top/bottom 1%)
q_low  = df['median_house_value'].quantile(0.01)
q_high = df['median_house_value'].quantile(0.99)
df = df[(df['median_house_value'] >= q_low) & (df['median_house_value'] <= q_high)]

# 2d. Feature engineering — derived features that improve model accuracy
df['rooms_per_household']    = df['total_rooms']    / df['households']
df['bedrooms_per_room']      = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population']   / df['households']
df['income_per_room']        = df['median_income']  / (df['rooms_per_household'] + 1e-6)

# 2e. Encode categorical variable: ocean_proximity (one-hot for better accuracy)
df = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean', drop_first=False)
ocean_cols = [c for c in df.columns if c.startswith('ocean_')]
# Convert boolean dummies to int
for col in ocean_cols:
    df[col] = df[col].astype(int)

print(f"\nClean dataset shape: {df.shape}")
print(f"Remaining NaNs     : {df.isnull().sum().sum()}")

# 2f. Define features and target
features = ['longitude', 'latitude', 'housing_median_age',
            'total_rooms', 'total_bedrooms', 'population', 'households',
            'median_income',
            'rooms_per_household', 'bedrooms_per_room',
            'population_per_household', 'income_per_room'] + ocean_cols
X = df[features].copy()
y = df['median_house_value']

# Final NaN safety check
X.fillna(X.median(), inplace=True)
assert X.isnull().sum().sum() == 0, "NaNs still present in X!"
print(f"Features used: {len(features)}")
print("All NaNs resolved. Ready for modelling.")

# 2g. Train / test split  (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2h. Feature scaling
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain size : {X_train_sc.shape[0]:,}")
print(f"Test  size : {X_test_sc.shape[0]:,}")


# ── 3. DATA VISUALISATION ─────────────────────────────────────────────────────

# 3a. Feature distributions (core features only)
core_cols = ['median_house_value', 'median_income', 'housing_median_age',
             'total_rooms', 'total_bedrooms', 'population',
             'households', 'rooms_per_household', 'bedrooms_per_room']
fig, axes = plt.subplots(3, 3, figsize=(16, 11))
fig.suptitle('Feature Distributions – California Housing Dataset', fontsize=14)
for ax, col in zip(axes.flatten(), core_cols):
    ax.hist(df[col], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_title(col.replace('_', ' ').title(), fontsize=9)
    ax.set_xlabel(col, fontsize=8); ax.set_ylabel('Count', fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig('outputs/01_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 01_distributions.png")

# 3b. Boxplots – outlier detection
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Boxplots – Outlier Detection', fontsize=13)
for ax, col in zip(axes, ['median_house_value', 'median_income',
                           'housing_median_age', 'rooms_per_household']):
    ax.boxplot(df[col], patch_artist=True,
               boxprops=dict(facecolor='lightblue'),
               medianprops=dict(color='red', linewidth=2),
               flierprops=dict(marker='o', markersize=2, alpha=0.3))
    ax.set_title(col.replace('_', ' ').title(), fontsize=10)
    ax.set_ylabel('Value'); ax.xaxis.set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig('outputs/02_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 02_boxplots.png")

# 3c. Correlation heatmap (core numeric features only for readability)
corr_cols = ['median_house_value', 'median_income', 'housing_median_age',
             'total_rooms', 'total_bedrooms', 'population', 'households',
             'rooms_per_household', 'bedrooms_per_room',
             'population_per_household', 'longitude', 'latitude']
fig, ax = plt.subplots(figsize=(13, 10))
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            annot_kws={'size': 8})
ax.set_title('Correlation Matrix – Multicollinearity Check', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig('outputs/03_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 03_correlation.png")
# NOTE: total_rooms, total_bedrooms, households, population are highly
#       correlated (r ~0.9) → multicollinearity inflates Linear Regression
#       coefficient variance. Tree-based models are unaffected.


# ── 4. REGRESSION MODELS ─────────────────────────────────────────────────────

# 4a. Linear Regression
print("\n[1/5] Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
joblib.dump(lr, 'models/linear_regression.pkl')
print("      Done. Saved.")

# 4b. Polynomial Regression (degree = 2)
print("[2/5] Training Polynomial Regression (degree=2)...")
poly_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('lr',   LinearRegression())
])
poly_pipe.fit(X_train_sc, y_train)
y_pred_poly = poly_pipe.predict(X_test_sc)
joblib.dump(poly_pipe, 'models/polynomial_regression.pkl')
print("      Done. Saved.")

# 4c. Decision Tree — deeper for accuracy
print("[3/5] Training Decision Tree...")
dt = DecisionTreeRegressor(
    max_depth=12,           # deeper than before (was 8)
    min_samples_split=10,   # tighter splits (was 20)
    min_samples_leaf=5,     # smaller leaves (was 10)
    random_state=42
)
dt.fit(X_train_sc, y_train)
y_pred_dt = dt.predict(X_test_sc)
joblib.dump(dt, 'models/decision_tree.pkl')
print("      Done. Saved.")

# 4d. Random Forest — more trees and deeper for accuracy (~2-3 min)
print("[4/5] Training Random Forest (300 trees, depth=15) — this takes ~2-3 min...")
rf = RandomForestRegressor(
    n_estimators=300,       # more trees (was 100)
    max_depth=15,           # deeper (was 12)
    min_samples_split=5,    # tighter splits (was 10)
    min_samples_leaf=3,     # smaller leaves
    max_features='sqrt',    # standard best practice
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
rf.fit(X_train_sc, y_train)
y_pred_rf = rf.predict(X_test_sc)
joblib.dump(rf, 'models/random_forest.pkl')
print("      Done. Saved.")

# 4e. RNN (Deep MLP) — larger network, more epochs (~1 min)
print("[5/5] Training RNN / Deep MLP...")
rnn_model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32),  # deeper network (was 128,64,32)
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    batch_size=256,         # smaller batches = more gradient steps (was 512)
    max_iter=200,           # more epochs allowed (was 100)
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,    # more patience (was 10)
    random_state=42,
    verbose=True
)
rnn_model.fit(X_train_sc, y_train)
y_pred_rnn = rnn_model.predict(X_test_sc)
joblib.dump(rnn_model, 'models/rnn_mlp.pkl')
joblib.dump(scaler,    'models/scaler.pkl')
print("      Done. Saved.")


# ── 5. EVALUATION METRICS ─────────────────────────────────────────────────────
def get_metrics(name, y_true, y_pred):
    return {
        'Model'    : name,
        'MAE ($)'  : round(mean_absolute_error(y_true, y_pred), 2),
        'MSE'      : round(mean_squared_error(y_true, y_pred), 2),
        'R² Score' : round(r2_score(y_true, y_pred), 4)
    }

results = pd.DataFrame([
    get_metrics('Linear Regression',     y_test, y_pred_lr),
    get_metrics('Polynomial Regression', y_test, y_pred_poly),
    get_metrics('Decision Tree',         y_test, y_pred_dt),
    get_metrics('Random Forest',         y_test, y_pred_rf),
    get_metrics('RNN (Deep MLP)',        y_test, y_pred_rnn),
])

print("\n" + "=" * 65)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 65)
print(results.to_string(index=False))
results.to_csv('outputs/model_results.csv', index=False)
print("\nSaved: model_results.csv")


# ── 6. VISUALISATION PLOTS ────────────────────────────────────────────────────
preds = {
    'Linear Reg.'  : y_pred_lr,
    'Poly Reg.'    : y_pred_poly,
    'Decision Tree': y_pred_dt,
    'Random Forest': y_pred_rf,
    'RNN (MLP)'    : y_pred_rnn
}

# 6a. Predictions vs Actual
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle('Predicted vs Actual House Value – All Models', fontsize=13)
y_min, y_max = y_test.min(), y_test.max()
for ax, (name, yp) in zip(axes, preds.items()):
    ax.scatter(y_test / 1e3, yp / 1e3, alpha=0.2, s=8, color='steelblue')
    ax.plot([y_min/1e3, y_max/1e3], [y_min/1e3, y_max/1e3],
            'r--', linewidth=1.5, label='Perfect fit')
    ax.set_title(f'{name}\nR²={r2_score(y_test, yp):.3f}', fontsize=10)
    ax.set_xlabel('Actual ($k)'); ax.set_ylabel('Predicted ($k)')
plt.tight_layout()
plt.savefig('outputs/04_predictions_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 04_predictions_vs_actual.png")

# 6b. Residuals distribution
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle('Residuals Distribution – All Models', fontsize=13)
for ax, (name, yp) in zip(axes, preds.items()):
    residuals = (y_test - yp) / 1e3
    ax.hist(residuals, bins=50, color='mediumpurple', edgecolor='white', alpha=0.85)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Residual ($k)'); ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('outputs/05_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 05_residuals.png")

# 6c. Learning curves (Linear Regression + Decision Tree — fast models)
print("\nGenerating learning curves (~1 min)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Learning Curves – Linear Regression & Decision Tree', fontsize=13)
for ax, (model, name) in zip(axes, [
        (LinearRegression(), 'Linear Regression'),
        (DecisionTreeRegressor(max_depth=12, random_state=42), 'Decision Tree')]):
    sizes, tr_sc, val_sc = learning_curve(
        model, X_train_sc, y_train,
        cv=5,                               # full 5-fold CV (was 3)
        scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 8),  # 8 size points (was 5)
        n_jobs=-1
    )
    ax.plot(sizes, tr_sc.mean(axis=1),  'o-', color='steelblue', label='Train R²')
    ax.fill_between(sizes,
                    tr_sc.mean(axis=1) - tr_sc.std(axis=1),
                    tr_sc.mean(axis=1) + tr_sc.std(axis=1),
                    alpha=0.15, color='steelblue')
    ax.plot(sizes, val_sc.mean(axis=1), 'o-', color='coral', label='Val R²')
    ax.fill_between(sizes,
                    val_sc.mean(axis=1) - val_sc.std(axis=1),
                    val_sc.mean(axis=1) + val_sc.std(axis=1),
                    alpha=0.15, color='coral')
    ax.set_title(name); ax.set_xlabel('Training Size'); ax.set_ylabel('R² Score')
    ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('outputs/06_learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 06_learning_curves.png")

# 6d. RNN loss curve
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(rnn_model.loss_curve_, color='steelblue', linewidth=2, label='Train Loss (MSE)')
ax.set_title('RNN (Deep MLP) – Training Loss per Epoch')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('outputs/07_rnn_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 07_rnn_loss_curve.png")

# 6e. Feature importance – Random Forest (top 15 features)
importances = pd.Series(rf.feature_importances_, index=features).sort_values().tail(15)
fig, ax = plt.subplots(figsize=(9, 6))
importances.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Top 15 Feature Importances – Random Forest', fontsize=12)
ax.set_xlabel('Importance Score')
for i, v in enumerate(importances):
    ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/08_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 08_feature_importance.png")

print("\n" + "=" * 55)
print("ALL DONE — check the outputs/ folder for all plots")
print("           and models/ folder for saved models.")
print("=" * 55)