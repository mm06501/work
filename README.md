# AQI 3-Day Forecast System 🌎

This project is a complete AI-driven Air Quality Index (AQI) Forecasting System that predicts the AQI for the next 3 days using advanced machine learning models.  
It includes a **fancy Flask frontend** where users can either:

- Predict based on **latest real-time weather data** 🛰️
- Predict by **entering custom temperature, wind speed, and pressure** inputs manually 🧪

---

## 📜 Project Overview

- **Backend:** Flask (Python)
- **Machine Learning Models:** LightGBM + LSTM + TCN (Ensemble)
- **Frontend:** Bootstrap 5 (Fancy Modern UI)
- **Dataset:** Weather and Air Quality historical dataset (12 months)

---

## 🚀 My Approach and Core Logic

I followed a **careful pipeline** for maximum accuracy:

- Collected weather and AQI data.
- Engineered time-based features like `hour_sin`, `hour_cos`, `dow_sin`, and `dow_cos`.
- Built lag features for AQI trends (1h, 3h, 6h, 12h lags).
- Applied advanced augmentation techniques (jitter + mixup) to enrich sequence data.
- Trained models using LightGBM for tabular features and deep learning (LSTM + TCN) for sequences.
- Created a **smart ensemble** by averaging predictions from all models for better robustness.

---

## 🧠 Core Technologies & Libraries Used

| Area | Tools/Technologies |
|:-----|:-------------------|
| Machine Learning | LightGBM, TensorFlow, Keras |
| Deep Learning | LSTM, Temporal Convolutional Network (TCN) |
| Web Framework | Flask |
| Frontend Styling | Bootstrap 5 |
| Data Handling | Pandas, NumPy |
| Scaling | StandardScaler (scikit-learn) |

---

## 🛠️ How the System Works

1. **Fetch latest weather data** from CSV (`weather_and_aqi_last12m.csv`).
2. **Preprocess** data — create time features, lag features, and fill missing values.
3. **Predict AQI** using trained models:
   - If using real-time latest data ➔ Ensemble prediction (LGBM + LSTM + TCN).
   - If using custom manual input ➔ LightGBM direct prediction.
4. **Display output** beautifully using Bootstrap frontend.

---

## 📈 Frontend Experience

- Displays the latest 100 records from dataset in an attractive responsive table 📊
- Two main options:
  - 📈 Forecast based on **Live Latest Data**
  - 🧪 Predict AQI based on **Custom User Input**
- Result is **displayed instantly below the form** without refreshing the page
- Responsive and optimized for **desktop and mobile** devices

---

## 📋 Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/mm06501/work.git
    cd work
    ```

2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask app:
    ```bash
    python app.py
    ```

4. Open browser and visit:
    ```
    http://127.0.0.1:5000
    ```

---

## 📸 Project Preview

![App Screenshot](static/images/preview.png)

---

## ✨ Future Improvements

- Add city selection for forecasting.
- Integrate real-time API instead of static CSV.
- Deploy live on a cloud server using Docker + GitHub Actions.

---

## 🤝 Credits

- Designed and developed by [mm06501](https://github.com/mm06501).
- Special thanks to open-source Python libraries ❤️.
