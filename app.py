from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tcn import TCN
import os

app = Flask(__name__, static_folder='static')

DATA_CSV = 'weather_and_aqi_last12m.csv'
df = pd.read_csv(DATA_CSV, parse_dates=['dt'], index_col='dt')
df.index = pd.to_datetime(df.index)

lg_model = joblib.load('lightgbm_model.pkl')
seq_scaler = joblib.load('seq_scaler.pkl')

if os.environ.get("GITHUB_ACTIONS") != "true":
    lstm_model = load_model('lstm_tuned.h5', compile=False)
    with custom_object_scope({'TCN': TCN}):
        tcn_model = load_model('tcn_model.h5', compile=False)
else:
    lstm_model = None
    tcn_model = None

PAST = 72
HORIZONS = [24, 48, 72]
SEQ_FEATURES = ['aqi', 'wspd', 'pres', 'hour_sin', 'hour_cos', 'wdir_sin', 'wdir_cos']


def add_time_features(data):
    idx = data.index.to_series()
    data['hour_sin'] = np.sin(2 * np.pi * idx.dt.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * idx.dt.hour / 24)
    data['dow_sin'] = np.sin(2 * np.pi * idx.dt.dayofweek / 7)
    data['dow_cos'] = np.cos(2 * np.pi * idx.dt.dayofweek / 7)
    if 'wdir' in data.columns:
        rad = np.deg2rad(data['wdir'])
        data['wdir_sin'] = np.sin(rad)
        data['wdir_cos'] = np.cos(rad)
    else:
        data['wdir_sin'] = 0.0
        data['wdir_cos'] = 0.0
    return data


def prepare_sequence_input(data):
    recent = add_time_features(data.copy()).iloc[-PAST:]
    arr = recent[SEQ_FEATURES].values
    flat = arr.reshape(-1, arr.shape[-1])
    scaled = seq_scaler.transform(flat)
    return scaled.reshape(1, PAST, arr.shape[-1])


@app.route('/', methods=['GET'])
def index():
    preview = df.reset_index().tail(100)
    columns = preview.columns.tolist()
    records = preview.to_dict(orient='records')
    return render_template('index.html', columns=columns, records=records, HORIZONS=HORIZONS)


@app.route("/predict", methods=["POST"])
def predict():
    selected_horizons = request.form.getlist("horizon")
    selected_horizons = [int(h) for h in selected_horizons]

    latest = df.tail(1).copy()

    feat_cols = [
        'prcp', 'snow', 'wspd', 'pres', 'tsun',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'aqi_lag1', 'aqi_lag3', 'aqi_lag6', 'aqi_lag12'
    ]

    for col in feat_cols:
        if col not in latest.columns:
            latest[col] = 0

    latest = latest[feat_cols]

    p_lg = lg_model.predict(latest).flatten()

    forecast_results = {}
    for idx, h in enumerate([24, 48, 72]):
        if h in selected_horizons:
            forecast_results[f"{h}h Forecast"] = round(float(p_lg[idx]), 1)

    preview = df.reset_index().tail(100)
    columns = preview.columns.tolist()
    records = preview.to_dict(orient='records')

    return render_template(
        "index.html",
        forecast_results=forecast_results,
        columns=columns,
        records=records,
        HORIZONS=HORIZONS
    )


@app.route('/custom_predict', methods=['POST'])
def custom_predict():
    temp = float(request.form['temp'])
    wspd = float(request.form['wspd'])
    pres = float(request.form['pres'])

    custom_df = pd.DataFrame([{
        'prcp': 0,
        'snow': 0,
        'wspd': wspd,
        'pres': pres,
        'tsun': 0,
        'hour_sin': 0,
        'hour_cos': 1,
        'dow_sin': 0,
        'dow_cos': 1,
        'aqi_lag1': 50,
        'aqi_lag3': 50,
        'aqi_lag6': 50,
        'aqi_lag12': 50
    }])

    feat_cols = [
        'prcp', 'snow', 'wspd', 'pres', 'tsun',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'aqi_lag1', 'aqi_lag3', 'aqi_lag6', 'aqi_lag12'
    ]

    custom_df = custom_df[feat_cols]

    prediction = lg_model.predict(custom_df)

    custom_results = {
        "24h AQI": round(float(prediction[0][0]), 1),
        "48h AQI": round(float(prediction[0][1]), 1),
        "72h AQI": round(float(prediction[0][2]), 1)
    }

    preview = df.reset_index().tail(100)
    columns = preview.columns.tolist()
    records = preview.to_dict(orient='records')

    return render_template(
        "index.html",
        custom_results=custom_results,
        columns=columns,
        records=records,
        HORIZONS=HORIZONS
    )


if __name__ == '__main__':
    app.run(debug=True)
