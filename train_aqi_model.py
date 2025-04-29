# # train_aqi_model_enhanced.py

# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime, timedelta
# from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from lightgbm import LGBMRegressor

# # 1) Load & index data
# df = pd.read_csv("weather_and_aqi_last3m.csv", parse_dates=["datetime"])
# df.set_index("datetime", inplace=True)

# # 2) Clean missing values
# for col in ["prcp", "snow", "tsun", "wpgt"]:
#     if col in df.columns:
#         df[col].fillna(0, inplace=True)

# # Forward-fill remaining then zero-fill
# df.ffill(inplace=True)
# df.fillna(0, inplace=True)

# # 3) Time features (cyclical)
# df["hour"] = df.index.hour
# df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
# df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
# df["dow"] = df.index.dayofweek
# df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
# df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

# # 4) Wind-direction as exogenous driver
# df["wdir_rad"] = np.deg2rad(df["wdir"])
# df["wdir_sin"] = np.sin(df["wdir_rad"])
# df["wdir_cos"] = np.cos(df["wdir_rad"])

# # 5) Traffic proxy feature if available
# if "traffic_flow" in df.columns:
#     df["traffic_flow"] = df["traffic_flow"].ffill().fillna(0)
#     traffic_feat = ["traffic_flow"]
# else:
#     traffic_feat = []

# # 6) Lagged AQI features
# lags = [1, 3, 6, 12]
# for l in lags:
#     df[f"aqi_lag{l}"] = df["aqi"].shift(l)

# # 7) Rolling statistics for AQI: mean, min, max, variance
# windows = [24, 72]
# for w in windows:
#     rol = df["aqi"].rolling(window=w, min_periods=w)
#     df[f"aqi_roll{w}_mean"] = rol.mean().shift(1)
#     df[f"aqi_roll{w}_min"] = rol.min().shift(1)
#     df[f"aqi_roll{w}_max"] = rol.max().shift(1)
#     df[f"aqi_roll{w}_var"] = rol.var().shift(1)

# # 8) Multi-horizon targets (24h, 48h, 72h average next)
# horizons = [24, 48, 72]
# for h in horizons:
#     df[f"target_{h}h"] = df["aqi"].rolling(window=h, min_periods=h).mean().shift(-h)

# # 9) Drop rows with missing features or targets
# target_cols = [f"target_{h}h" for h in horizons]
# feature_checks = [f"aqi_lag{l}" for l in lags] + [
#     f"aqi_roll{w}_{stat}"
#     for w in windows
#     for stat in ("mean", "min", "max", "var")
# ]
# all_drop = [c for c in target_cols + feature_checks if c in df.columns]
# df.dropna(subset=all_drop, inplace=True)

# # 10) Assemble feature & target matrices
# weather_feats = [
#     "temp", "dwpt", "rhum", "prcp", "snow", "wspd", "wpgt",
#     "pres", "tsun", "coco", "co", "no", "no2", "o3", "so2",
#     "pm2_5", "pm10", "nh3"
# ]
# time_feats = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
# wind_feats = ["wdir_sin", "wdir_cos"]
# lag_feats = [f"aqi_lag{l}" for l in lags]
# roll_feats = []
# for w in windows:
#     roll_feats += [
#         f"aqi_roll{w}_mean", f"aqi_roll{w}_min",
#         f"aqi_roll{w}_max", f"aqi_roll{w}_var"
#     ]
# feature_cols = weather_feats + time_feats + wind_feats + traffic_feat + lag_feats + roll_feats
# feature_cols = [c for c in feature_cols if c in df.columns]

# X = df[feature_cols]
# y = df[target_cols]

# # 11) Persistence baseline (t-1)
# split_idx = int(len(df) * 0.8)
# y_test = y.iloc[split_idx:]
# baseline = df["aqi_lag1"].iloc[split_idx:].values.reshape(-1, 1)
# baseline_preds = np.repeat(baseline, len(horizons), axis=1)

# print("Baseline (t-1) results:")
# for i, h in enumerate(horizons):
#     y_true = y_test.iloc[:, i]
#     y_pred = baseline_preds[:, i]
#     print(f" {h}h -> MAE {mean_absolute_error(y_true, y_pred):.2f}, "
#           f"RMSE {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}, "
#           f"R2 {r2_score(y_true, y_pred):.3f}")

# # 12) Chronological split
# X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
# y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# # 13) Model & hyperparameter tuning
# base = LGBMRegressor(random_state=42, n_jobs=-1)
# model = MultiOutputRegressor(base)
# param_dist = {
#     "estimator__n_estimators": [100, 200],
#     "estimator__num_leaves": [31, 64],
#     "estimator__learning_rate": [0.05, 0.1],
#     "estimator__max_depth": [5, 10, -1],
# }
# tscv = TimeSeriesSplit(n_splits=4)
# search = RandomizedSearchCV(
#     model, param_dist, n_iter=10, cv=tscv,
#     scoring="neg_mean_absolute_error",
#     n_jobs=-1, verbose=1, random_state=42
# )

# print("Starting hyperparameter tuning…")
# search.fit(X_train, y_train)
# best = search.best_estimator_
# print("Best params:", search.best_params_)

# # 14) Evaluate tuned model
# print("\nTuned model results:")
# preds = best.predict(X_test)
# for i, h in enumerate(horizons):
#     y_t = y_test.iloc[:, i]
#     y_p = preds[:, i]
#     print(f" {h}h -> MAE {mean_absolute_error(y_t, y_p):.2f}, "
#           f"RMSE {np.sqrt(mean_squared_error(y_t, y_p)):.2f}, "
#           f"R2 {r2_score(y_t, y_p):.3f}")

# # 15) Save the final model
# joblib.dump(best, "aqi_model_enhanced.pkl")
# print("Model saved as aqi_model_enhanced.pkl")



########################## new approch for improvemnet 




"""
train_aqi_model_advanced.py
—————————————
Advanced AQI Forecasting: Hyper-tuned LSTM + TCN + LightGBM Ensemble with Augmentation

Features:
• Data augmentation (jitter + mixup) for sequence models
• Hyperparameter tuning of LSTM via Keras-Tuner
• Temporal Convolutional Network (TCN) branch
• LightGBM feature-based model
• Ensemble predictions by aligning time indices

Requirements:
    pandas numpy scikit-learn lightgbm tensorflow keras-tuner keras-tcn joblib

Usage:
    python train_aqi_model_advanced.py
"""
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

import keras_tuner as kt
from tcn import TCN

# 1) Load data (12m preferred, fallback to 3m)
fn = None
for candidate in ['weather_and_aqi_last12m.csv', 'weather_and_aqi_last3m.csv']:
    if os.path.exists(candidate):
        fn = candidate
        break
if fn is None:
    raise FileNotFoundError('No CSV found')

# parse & index by your actual "dt" column
df = pd.read_csv(
    fn,
    parse_dates=['dt'],
    index_col='dt'
)

# --- NEW: ensure it’s a true DatetimeIndex ---
df.index = pd.to_datetime(df.index)

# 2) Feature engineering
# avoid chained-assignment: assign back
for col in ['prcp', 'snow', 'tsun', 'wpgt']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# forward-fill then zero-fill
df.ffill(inplace=True)
df.fillna(0, inplace=True)

# build a Series for dt accessor
idx = df.index.to_series()

# Cyclical time features
df['hour_sin'] = np.sin(2 * np.pi * idx.dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * idx.dt.hour / 24)
df['dow_sin']  = np.sin(2 * np.pi * idx.dt.dayofweek / 7)
df['dow_cos']  = np.cos(2 * np.pi * idx.dt.dayofweek / 7)

# Wind-direction features
if 'wdir' in df.columns:
    df['wdir_rad'] = np.deg2rad(df['wdir'])
    df['wdir_sin'] = np.sin(df['wdir_rad'])
    df['wdir_cos'] = np.cos(df['wdir_rad'])

# AQI lags & rolling stats
for l in [1, 3, 6, 12]:
    df[f'aqi_lag{l}'] = df['aqi'].shift(l)
for w in [24, 72]:
    rol = df['aqi'].rolling(window=w, min_periods=w)
    df[f'aqi_roll{w}_mean'] = rol.mean().shift(1)
    df[f'aqi_roll{w}_min']  = rol.min().shift(1)
    df[f'aqi_roll{w}_max']  = rol.max().shift(1)
    df[f'aqi_roll{w}_var']  = rol.var().shift(1)

# 3) Create targets (24h, 48h, 72h avg AQI)
horizons = [24, 48, 72]
for h in horizons:
    df[f'target_{h}h'] = (
        df['aqi']
          .rolling(window=h, min_periods=h)
          .mean()
          .shift(-h)
    )
df.dropna(subset=[f'target_{h}h' for h in horizons], inplace=True)

# 4) LightGBM feature-based model
feat_cols = [
    'prcp', 'snow', 'wspd', 'pres', 'tsun',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
] + [f'aqi_lag{l}' for l in [1, 3, 6, 12]]

Xf = df[feat_cols]
yf = df[[f'target_{h}h' for h in horizons]]

split_idx = int(0.8 * len(df))
Xf_tr, Xf_te = Xf.iloc[:split_idx], Xf.iloc[split_idx:]
y_tr,  y_te  = yf.iloc[:split_idx], yf.iloc[split_idx:]

lg = lgb.LGBMRegressor(random_state=42)
b_lg = MultiOutputRegressor(lg).fit(Xf_tr, y_tr)
p_lg = pd.DataFrame(
    b_lg.predict(Xf_te),
    index=Xf_te.index,
    columns=y_te.columns
)

# 5) Sequence data for LSTM & TCN
seq_feats = ['aqi', 'wspd', 'pres', 'hour_sin', 'hour_cos']
if 'wdir_sin' in df.columns:
    seq_feats += ['wdir_sin', 'wdir_cos']

PAST = 72
max_h = max(horizons)
df_seq = df.dropna(subset=seq_feats)

timestamps, Xs, ys = [], [], []
for i in range(PAST, len(df_seq) - max_h + 1):
    Xs.append(df_seq[seq_feats].iloc[i-PAST:i].values)
    ys.append([df_seq['aqi'].iloc[i:i+h].mean() for h in horizons])
    timestamps.append(df_seq.index[i])

X_seq = np.array(Xs)
y_seq = np.array(ys)

split_seq = int(0.8 * len(X_seq))
train_idx = timestamps[:split_seq]
test_idx  = timestamps[split_seq:]

Xtr_s, Xte_s = X_seq[:split_seq], X_seq[split_seq:]
ytr_s, yte_s = y_seq[:split_seq], y_seq[split_seq:]

# Augmentation functions
def jitter(X, sigma=0.01):
    return X + np.random.normal(0, sigma, X.shape)

def mixup(X, y, alpha=0.4):
    idx = np.random.permutation(len(X))
    lam = np.random.beta(alpha, alpha, len(X))[:, None, None]
    X2, y2 = X[idx], y[idx]
    lam_y = lam.squeeze()[:, None]
    X_mix = X * lam + X2 * (1 - lam)
    y_mix = y * lam_y + y2 * (1 - lam_y)
    return X_mix, y_mix

# 6) Scale & augment training sequences
ns, nt, nf = Xtr_s.shape
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr_s.reshape(-1, nf)).reshape(ns, nt, nf)
Xte_s = scaler.transform(Xte_s.reshape(-1, nf)).reshape(len(Xte_s), nt, nf)
joblib.dump(scaler, 'seq_scaler.pkl')

X_jit = jitter(Xtr_s, sigma=0.005)
y_jit = ytr_s.copy()
X_mix, y_mix = mixup(Xtr_s, ytr_s, alpha=0.4)
X_aug = np.vstack([Xtr_s, X_jit, X_mix])
y_aug = np.vstack([ytr_s, y_jit, y_mix])

# 7) Hyperparameter-tuned LSTM via Keras-Tuner
def build_lstm(hp):
    m = models.Sequential()
    n_layers = hp.Int('layers', 1, 3)
    for i in range(n_layers):
        units = hp.Int(f'units_{i}', 32, 128, step=32)
        m.add(layers.LSTM(units, return_sequences=(i < n_layers-1)))
    m.add(layers.Dense(len(horizons)))
    m.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('lr', 1e-3, 1e-1, sampling='log')
        ),
        loss='mse'
    )
    return m

tuner = kt.RandomSearch(
    build_lstm,
    objective='val_loss',
    max_trials=5,
    directory='tuner',
    project_name='aqi'
)
tuner.search(
    X_aug, y_aug,
    epochs=30,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5)],
    verbose=1
)
best_lstm = tuner.get_best_models(1)[0]

# 8) Train TCN on augmented data
inp = layers.Input(shape=(nt, nf))
out = TCN(64)(inp)
out = layers.Dense(len(horizons))(out)
model_tcn = models.Model(inp, out)
model_tcn.compile(optimizer='adam', loss='mse')
model_tcn.fit(
    X_aug, y_aug,
    validation_split=0.1,
    epochs=30,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5)],
    verbose=1
)

# 9) Ensemble predictions
p_lstm_df = pd.DataFrame(
    best_lstm.predict(Xte_s),
    index=test_idx,
    columns=[f'target_{h}h' for h in horizons]
)
p_tcn_df = pd.DataFrame(
    model_tcn.predict(Xte_s),
    index=test_idx,
    columns=[f'target_{h}h' for h in horizons]
)
common = p_lg.index.intersection(p_lstm_df.index).intersection(p_tcn_df.index)
p_ens = (p_lg.loc[common] + p_lstm_df.loc[common] + p_tcn_df.loc[common]) / 3
y_true = y_te.loc[common]

# 10) Evaluate
print('Augmented Ensemble Results:')
for h in horizons:
    mae  = mean_absolute_error(y_true[f'target_{h}h'], p_ens[f'target_{h}h'])
    rmse = np.sqrt(mean_squared_error(y_true[f'target_{h}h'], p_ens[f'target_{h}h']))
    r2   = r2_score(y_true[f'target_{h}h'], p_ens[f'target_{h}h'])
    print(f"{h}h -> MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

# 11) Save all models
best_lstm.save('lstm_tuned.h5')
model_tcn.save('tcn_model.h5')
joblib.dump(b_lg, 'lightgbm_model.pkl')
print('✅ Saved all models.')
