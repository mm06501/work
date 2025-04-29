"""
fetch_weather_and_aqi.py
—————————————
• Rolling 3‐month window: current month + previous 2 months
• Weather   : Meteostat hourly (open)
• Air Quality: OpenWeather air_pollution/history
• Merge on datetime ±30 min → CSV

Setup:
    pip install pandas requests python-dateutil meteostat tqdm pytz
"""

import pandas as pd
import requests
from datetime import datetime, date, time, timezone
from dateutil.relativedelta import relativedelta
from meteostat import Hourly, Point
from tqdm import tqdm

# ───── CONFIG ───── #
LAT, LON   = 33.6844, 73.0479               
TZ         = "Asia/Karachi"
API_KEY    = "bb501c0016562abfc0fee02e63871d4f"  
OUT_FILE   = "weather_and_aqi_last3m.csv"
# ──────────────────── #

# 1) Compute 3‐month window (calendar months)
today      = datetime.now(timezone.utc).date()       # e.g. 2025-04-28
start_date = (today.replace(day=1) - relativedelta(months=2))
end_date   = today
print(f"Fetching weather+AQI from {start_date} to {end_date} "
      f"({(end_date-start_date).days+1} days)")

# 2) Fetch hourly weather via Meteostat
#    (returns UTC timestamps, convert to Asia/Karachi)
station = Point(LAT, LON)
start_dt = datetime.combine(start_date, time.min)   # naive UTC
end_dt   = datetime.combine(end_date,   time.max)
wx = Hourly(station, start_dt, end_dt).fetch().reset_index()
wx.rename(columns={"time":"datetime"}, inplace=True)
wx["datetime"] = (
    wx["datetime"].dt.tz_localize("UTC")
                 .dt.tz_convert(TZ)
)

# 3) Fetch AQI & pollutant concentrations via OpenWeather
start_unix = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
end_unix   = int(end_dt.replace(tzinfo=timezone.utc).timestamp())
aq_url = (
    f"https://api.openweathermap.org/data/2.5/air_pollution/history"
    f"?lat={LAT}&lon={LON}"
    f"&start={start_unix}&end={end_unix}"
    f"&appid={API_KEY}"
)
print("🔗  fetching AQI & pollutants…")
resp = requests.get(aq_url, timeout=30)
resp.raise_for_status()
j = resp.json()

if "list" not in j or not j["list"]:
    raise RuntimeError("No AQI data returned – check your API key & dates.")

aq_df = pd.json_normalize(j["list"])
# dt → datetime with tz
aq_df["datetime"] = (
    pd.to_datetime(aq_df["dt"], unit="s", utc=True)
      .dt.tz_convert(TZ)
)
# flatten components & AQI
rename_map = {
    "main.aqi": "aqi",
    "components.co": "co",
    "components.no2": "no2",
    "components.no":  "no",
    "components.o3":  "o3",
    "components.so2": "so2",
    "components.pm2_5": "pm2_5",
    "components.pm10":  "pm10",
    "components.nh3":   "nh3",
}
aq_df.rename(columns=rename_map, inplace=True)

# 4) Merge on nearest timestamps (±30 min)
merged = pd.merge_asof(
    wx.sort_values("datetime"),
    aq_df.sort_values("datetime"),
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta("30m")
)

# 5) Save to CSV
merged.to_csv(OUT_FILE, index=False)
print(f"✅  saved {len(merged)} rows → {OUT_FILE}")
