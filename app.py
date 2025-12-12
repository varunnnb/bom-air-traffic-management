# app.py
"""
Streamlit dashboard — timestamp-selection mode (NO timezone conversion).
- User picks an exact timestamp from rolling_next_hour_predictions.csv (no timezone conversion)
- The app selects rolling rows for that timestamp, finds master flights in the same hour,
  enriches with aircraft DB (if provided), computes per-flight financials and shows:
    fuel_cost_inr, crew_cost_inr, other_ops_cost_inr, num_passengers_fallback,
    compensation_cost_inr, mtow_tonnes, expected_delay_hours, parking_hours_after_free,
    parking_cost_inr, delay_cost_inr, total_cost_inr
"""

import os
from datetime import datetime
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------
# Defaults (edit paths if needed)
MASTER_PATH_DEFAULT = "master_dataset_2019.csv"
ROLLING_PATH_DEFAULT = "rolling_next_hour_predictions.csv"
AIRCRAFT_DB_PATH_DEFAULT = "aircraftDatabase-2020-11.csv"
# ---------------------------

st.set_page_config(layout="wide", page_title="Flight Financials by Selected Hour")
st.title("Select-Timestamp Flight Delay Financial Dashboard")

# Sidebar: file paths and selection controls
st.sidebar.header("Files & settings")
master_path = st.sidebar.text_input("Master dataset CSV path", MASTER_PATH_DEFAULT)
rolling_path = st.sidebar.text_input("Rolling predictions CSV path", ROLLING_PATH_DEFAULT)
aircraft_db_path = st.sidebar.text_input("Aircraft DB CSV path", AIRCRAFT_DB_PATH_DEFAULT)

st.sidebar.markdown("### Pick timestamp from rolling predictions (exact hour)")
st.sidebar.caption("Timestamps are used *as-is* from the rolling CSV (no timezone conversion).")

# ---------------------------
# small defaults for model/MTOW/financials (you can tweak)
MODEL_SEAT_MAP = {
    "A320": 150, "A321": 185, "B737": 160, "B738": 160, "B737-800": 160,
    "B777": 300, "B787": 242, "A330": 250, "ATR72": 70, "CRJ": 50, "E190": 100
}
MTOW_MODEL_MAP = {
    "A320": 77.0, "A321": 97.0, "B737": 80.0, "B738": 79.0, "B737-800": 79.0,
    "B777": 351.0, "B787": 243.0, "A330": 242.0, "ATR72": 22.0, "CRJ": 39.0, "E190": 52.0,
}
DEFAULT_PASSENGERS = 150
DEFAULT_MTOW = 70.0

COST_PARAMS = {
    "fuel_per_minute_inr": 3000.0,
    "crew_per_minute_inr": 800.0,
    "other_ops_per_minute_inr": 700.0,
    "compensation_per_pax_inr": 2000.0,
    "compensation_delay_threshold_min": 60.0,
    "parking_rate_inr_per_mt_per_hour": 50.0,
    "free_parking_hours": 2.0,
    "default_passengers": DEFAULT_PASSENGERS,
    "default_mtow_tonnes": DEFAULT_MTOW,
}

# ---------------------------
# Helpers
def try_read_csv_as_str(path):
    if not os.path.exists(path):
        return None
    try:
        # read as strings to avoid dtype surprises
        return pd.read_csv(path, low_memory=False, dtype=str)
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return None

def parse_rolling_ts(series):
    # parse using the expected format 'DD-MM-YYYY HH:MM' but keep naive datetimes (no tz conversions)
    def _p(v):
        if pd.isna(v) or v == "":
            return pd.NaT
        s = str(v).strip()
        for fmt in ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        try:
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return pd.NaT
    return series.map(_p)

def parse_master_ts(series):
    def _p(v):
        if pd.isna(v) or v == "":
            return pd.NaT
        s = str(v).strip()
        for fmt in ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        if s.isdigit():
            try:
                n = int(s)
                if n > 1e12:
                    return pd.to_datetime(n, unit="ms")
                return pd.to_datetime(n, unit="s")
            except Exception:
                pass
        try:
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return pd.NaT
    return series.map(_p)

def normalize_icao24(df, col="icao24"):
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    if col not in df.columns:
        df[col] = ""
        return df
    df[col] = df[col].fillna("").astype(str).str.strip().str.lower().replace("nan", "")
    df[col] = df[col].str.replace(r"^0x", "", regex=True)
    return df

def normalize_callsign(df, col="callsign"):
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    if col not in df.columns:
        df[col] = ""
        return df
    df[col] = df[col].fillna("").astype(str).str.strip().str.upper().replace("NAN","")
    return df

def infer_mtow(model, typecode):
    for x in (model, typecode):
        if pd.isna(x) or x == "":
            continue
        k = str(x).upper()
        for km, mv in MTOW_MODEL_MAP.items():
            if km in k:
                return mv
    return DEFAULT_MTOW

def estimate_passengers_from_row(row):
    if "num_passengers" in row and pd.notna(row["num_passengers"]) and row["num_passengers"] != "":
        try:
            return int(float(row["num_passengers"]))
        except Exception:
            pass
    if "seatconfiguration" in row and pd.notna(row["seatconfiguration"]) and row["seatconfiguration"] != "":
        s = str(row["seatconfiguration"])
        nums = re.findall(r"\d{2,3}", s)
        if nums:
            return int(max(nums, key=int))
    for fld in ("model", "typecode"):
        if fld in row and pd.notna(row.get(fld)) and row.get(fld) != "":
            key = str(row.get(fld)).upper()
            for m, sc in MODEL_SEAT_MAP.items():
                if m in key:
                    return sc
    return COST_PARAMS["default_passengers"]

def compute_financials(df):
    df = df.copy()
    df["expected_delay_mins"] = df["predicted_delay_mins"].fillna(0.0).astype(float).clip(lower=0.0, upper=360.0)
    cp = COST_PARAMS
    df["fuel_cost_inr"] = df["expected_delay_mins"] * cp["fuel_per_minute_inr"]
    df["crew_cost_inr"] = df["expected_delay_mins"] * cp["crew_per_minute_inr"]
    df["other_ops_cost_inr"] = df["expected_delay_mins"] * cp["other_ops_per_minute_inr"]
    df["num_passengers_fallback"] = pd.to_numeric(df.get("num_passengers", cp["default_passengers"]), errors="coerce").fillna(cp["default_passengers"])
    df["compensation_cost_inr"] = 0.0
    mask = df["expected_delay_mins"] > cp["compensation_delay_threshold_min"]
    df.loc[mask, "compensation_cost_inr"] = df.loc[mask, "num_passengers_fallback"] * cp["compensation_per_pax_inr"]
    df["expected_delay_hours"] = df["expected_delay_mins"] / 60.0
    df["parking_hours_after_free"] = (df["expected_delay_hours"] - cp["free_parking_hours"]).clip(lower=0.0)
    df["parking_cost_inr"] = df["parking_hours_after_free"] * df["mtow_tonnes"] * cp["parking_rate_inr_per_mt_per_hour"]
    df["delay_cost_inr"] = df["fuel_cost_inr"] + df["crew_cost_inr"] + df["other_ops_cost_inr"] + df["compensation_cost_inr"]
    df["total_cost_inr"] = df["delay_cost_inr"] + df["parking_cost_inr"]
    return df

# ---------------------------
# Load CSVs
master_df = try_read_csv_as_str(master_path)
rolling_df = try_read_csv_as_str(rolling_path)
aircraft_db = try_read_csv_as_str(aircraft_db_path)

missing = []
if master_df is None:
    missing.append(("master", master_path))
if rolling_df is None:
    missing.append(("rolling", rolling_path))
if missing:
    st.error("One or more required CSVs could not be loaded: " + ", ".join([p for _, p in missing]))
    st.stop()

if aircraft_db is None:
    st.warning("Aircraft DB not found — aircraft enrichment will be skipped.")

# parse timestamps (no tz conversion)
if "timestamp" not in rolling_df.columns:
    st.error("rolling CSV must have 'timestamp' column")
    st.stop()

rolling_df["timestamp_parsed"] = parse_rolling_ts(rolling_df["timestamp"])
master_df["timestamp_parsed"] = parse_master_ts(master_df["timestamp"])

# normalize join keys
rolling_df = normalize_icao24(rolling_df, "icao24")
master_df = normalize_icao24(master_df, "icao24")
if aircraft_db is not None:
    aircraft_db = normalize_icao24(aircraft_db, "icao24")
rolling_df = normalize_callsign(rolling_df, "callsign")
master_df = normalize_callsign(master_df, "callsign")
if aircraft_db is not None and "operatorcallsign" in aircraft_db.columns:
    aircraft_db["operatorcallsign"] = aircraft_db["operatorcallsign"].fillna("").astype(str).str.strip().str.upper()

# ---------------------------
# Sidebar timestamp selection built from rolling CSV values (no conversion)
rolling_non_null = rolling_df[rolling_df["timestamp_parsed"].notna()].copy()
if rolling_non_null.empty:
    st.error("No parseable timestamps found in rolling CSV. Ensure format 'DD-MM-YYYY HH:MM' or similar.")
    st.stop()

rolling_non_null["date_only"] = rolling_non_null["timestamp_parsed"].dt.date
available_dates = sorted(rolling_non_null["date_only"].unique())
selected_date = st.sidebar.selectbox("Choose a date", available_dates)

day_data = rolling_non_null[rolling_non_null["date_only"] == selected_date].copy()
available_hours = sorted(day_data["timestamp_parsed"].unique())
selected_ts = st.sidebar.selectbox("Choose exact hour (timestamp)", available_hours)

# Show selected hour rolling rows (no plot)
hour_data = day_data[day_data["timestamp_parsed"] == selected_ts]
st.subheader("Selected Hour Rolling Rows")
st.write(hour_data[["timestamp","y_true","y_pred","true_category","pred_category"]])

# ---------------------------
# Select rolling_window using exact timestamp first, fallback to hour-floor
rolling_window = rolling_df[rolling_df["timestamp_parsed"] == selected_ts].copy()
st.write(f"Rolling rows selected for timestamp {selected_ts}: {len(rolling_window)}")
if rolling_window.empty:
    st.info("No exact-match rolling row found for selected timestamp — falling back to same-hour rows.")
    rolling_window = rolling_df[rolling_df["timestamp_parsed"].dt.floor("h") == selected_ts.replace(minute=0, second=0, microsecond=0)].copy()
    st.write(f"Rolling fallback rows (same hour): {len(rolling_window)}")

rolling_window["y_pred"] = pd.to_numeric(rolling_window.get("y_pred", 0), errors="coerce").fillna(0.0)
avg_pred_global = rolling_window["y_pred"].mean() if not rolling_window.empty else 0.0
agg_by_icao = rolling_window[rolling_window["icao24"].astype(bool)].groupby("icao24", dropna=True)["y_pred"].mean().reset_index().rename(columns={"y_pred":"y_pred_mean_icao"})
agg_by_callsign = rolling_window[rolling_window["callsign"].astype(bool)].groupby("callsign", dropna=True)["y_pred"].mean().reset_index().rename(columns={"y_pred":"y_pred_mean_callsign"})

# ---------------------------
# Find flights in master that correspond to selected_ts (exact parsed match -> fallback hour-floor)
flights_next = master_df[master_df["timestamp_parsed"] == selected_ts].copy()
if flights_next.empty:
    try:
        hour_key = selected_ts.replace(minute=0, second=0, microsecond=0)
        flights_next = master_df[master_df["timestamp_parsed"].dt.floor("h") == hour_key].copy()
    except Exception:
        flights_next = pd.DataFrame()

st.write(f"Flights found in master for selected timestamp (exact or hour-floor): {len(flights_next)}")

# Normalize keys on flights_next and merge predicted values
flights_next = normalize_icao24(flights_next, "icao24")
flights_next = normalize_callsign(flights_next, "callsign")

if not agg_by_icao.empty:
    flights_next = flights_next.merge(agg_by_icao, how="left", on="icao24")
else:
    flights_next["y_pred_mean_icao"] = np.nan

if not agg_by_callsign.empty:
    flights_next = flights_next.merge(agg_by_callsign, how="left", on="callsign")
else:
    flights_next["y_pred_mean_callsign"] = np.nan

flights_next["predicted_delay_mins"] = flights_next["y_pred_mean_icao"].fillna(flights_next["y_pred_mean_callsign"]).fillna(avg_pred_global).fillna(0.0)

# Attach aircraft DB info if available and has icao24
if aircraft_db is not None and "icao24" in aircraft_db.columns:
    aircraft_db = normalize_icao24(aircraft_db, "icao24")
    aircraft_subset = aircraft_db[["icao24","model","typecode","seatconfiguration","manufacturername","operator"]].drop_duplicates(subset=["icao24"])
    flights_next = flights_next.merge(aircraft_subset, how="left", on="icao24")
else:
    flights_next["model"] = flights_next.get("model", pd.NA)
    flights_next["typecode"] = flights_next.get("typecode", pd.NA)
    flights_next["seatconfiguration"] = flights_next.get("seatconfiguration", pd.NA)

# Estimate passengers and MTOW
if "num_passengers" not in flights_next.columns:
    flights_next["num_passengers"] = pd.NA
flights_next["num_passengers"] = flights_next.apply(estimate_passengers_from_row, axis=1)
flights_next["mtow_tonnes"] = flights_next.apply(lambda r: infer_mtow(r.get("model"), r.get("typecode")), axis=1)

# Compute financials
flights_fin = flights_next.copy()
flights_fin["predicted_delay_mins"] = pd.to_numeric(flights_fin["predicted_delay_mins"], errors="coerce").fillna(0.0)
flights_fin["mtow_tonnes"] = pd.to_numeric(flights_fin["mtow_tonnes"], errors="coerce").fillna(COST_PARAMS["default_mtow_tonnes"])
flights_fin = compute_financials(flights_fin)

# KPIs
total_cost = flights_fin["total_cost_inr"].sum() if not flights_fin.empty else 0.0
avg_delay = flights_fin["predicted_delay_mins"].mean() if not flights_fin.empty else 0.0
num_flights = len(flights_fin)
total_comp = flights_fin["compensation_cost_inr"].sum() if not flights_fin.empty else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Flights (selected hour)", f"{num_flights}")
k2.metric("Avg predicted delay (mins)", f"{avg_delay:.1f}")
k3.metric("Total expected cost (INR)", f"₹{total_cost:,.0f}")
k4.metric("Total expected compensation (INR)", f"₹{total_comp:,.0f}")

# ---------------------------
# Airline/operator aggregation + airline filter dropdown (NEW)
# Build a normalized airline_id column if not present
if "airline_id" not in flights_fin.columns:
    flights_fin["airline_id"] = flights_fin.get("operator").fillna(flights_fin.get("callsign").str[:3].fillna("UNK"))

# Compute overall aggregation (used to populate dropdown)
agg_air = flights_fin.groupby("airline_id").agg(
    flights_count=("icao24","count"),
    avg_delay_mins=("predicted_delay_mins","mean"),
    total_cost_inr=("total_cost_inr","sum"),
    total_comp=("compensation_cost_inr","sum")
).reset_index().sort_values("total_cost_inr", ascending=False)

# Sidebar dropdown to select an airline (or All)
airline_options = ["All"] + agg_air["airline_id"].tolist()
selected_airline = st.sidebar.selectbox("Filter by airline / operator", airline_options, index=0)

# Filter flights_fin to selected airline (or keep all)
if selected_airline != "All":
    flights_filtered = flights_fin[flights_fin["airline_id"] == selected_airline].copy()
    agg_air_display = agg_air[agg_air["airline_id"] == selected_airline].copy()
else:
    flights_filtered = flights_fin.copy()
    agg_air_display = agg_air.copy()

# Recompute KPIs for the filtered view
total_cost_filt = flights_filtered["total_cost_inr"].sum() if not flights_filtered.empty else 0.0
avg_delay_filt = flights_filtered["predicted_delay_mins"].mean() if not flights_filtered.empty else 0.0
num_flights_filt = len(flights_filtered)
total_comp_filt = flights_filtered["compensation_cost_inr"].sum() if not flights_filtered.empty else 0.0

# Show KPIs (replace or supplement the previous KPIs)
st.subheader("KPIs (filtered)")
kf1, kf2, kf3, kf4 = st.columns(4)
kf1.metric("Flights (selected)", f"{num_flights_filt}")
kf2.metric("Avg predicted delay (mins)", f"{avg_delay_filt:.1f}")
kf3.metric("Total expected cost (INR)", f"₹{total_cost_filt:,.0f}")
kf4.metric("Total expected compensation (INR)", f"₹{total_comp_filt:,.0f}")

# Airline / Operator summary (filtered)
st.subheader("Airline / Operator summary (filtered)")
left, right = st.columns([2,3])
with left:
    st.dataframe(agg_air_display.reset_index(drop=True), height=300)
with right:
    if not agg_air_display.empty:
        fig = px.bar(agg_air_display.head(10), x="airline_id", y="total_cost_inr",
                     title="Top airlines by expected cost (filtered)", labels={"total_cost_inr":"Total cost (INR)"})
        st.plotly_chart(fig, use_container_width=True)

# Flight-level detail table (filtered)
st.subheader("Flights (detailed) for selection")
detailed_cols = [
    "fuel_cost_inr","crew_cost_inr","other_ops_cost_inr","num_passengers_fallback",
    "compensation_cost_inr","mtow_tonnes","expected_delay_hours","parking_hours_after_free",
    "parking_cost_inr","delay_cost_inr","total_cost_inr"
]
id_cols = [c for c in ["timestamp","timestamp_parsed","callsign","icao24","operator","model","typecode","estdepartureairport","estarrivalairport","airline_id"] if c in flights_filtered.columns]
available_cols = [c for c in detailed_cols if c in flights_filtered.columns]
# combined columns to show
show_cols = id_cols + available_cols
if show_cols:
    st.dataframe(flights_filtered[show_cols].sort_values("total_cost_inr", ascending=False).reset_index(drop=True), height=500)
else:
    st.write("No detailed columns available to display.")

# Download filtered CSV
csv_bytes_filtered = flights_filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered results (CSV)", csv_bytes_filtered, file_name=f"selected_airline_{selected_airline}_financials.csv", mime="text/csv")

st.write("Notes: No timezone conversions are performed. The selected timestamp is taken exactly from the rolling CSV and used as the canonical key.")
