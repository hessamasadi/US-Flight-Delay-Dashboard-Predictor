import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import gdown
import os

st.set_page_config(layout="wide", page_title="Flight Delay Dashboard")

# ------------------------------------------------------------------
# 1. Define your Google Drive file IDs (from your share links)
# ------------------------------------------------------------------
FILE_IDS = {
    'flights_dashboard_ready.csv': '1nRh-fpz6y1iKgkAJdVg_sagKwDw6Pi6X',
    'airports_filtered.csv': '1IIiIL0cWj1aJSNi42l1D5GvJoBBCcWmN',
    'delay_regressor.pkl': '1KkYvOOKegz6_nckdrTLY0NQe1E-3cLzM',
    'delay_classifier.pkl': '1LUNDFtOjJMTYk2tvkQW2KtaaotWv9QYX',
    'label_encoders.pkl': '13famhi3vbswMqzwaLeeTFSeVnbVLucy_',
    'valid_routes.csv': '1LOhz9kycmrviSuqLtWPoDV1CKDyE-60O'
}

# ------------------------------------------------------------------
# 2. Helper function to download any file from Google Drive
#    Uses gdown with fuzzy=True to bypass the virus scan warning
# ------------------------------------------------------------------
def download_file_from_drive(file_id, output_path):
    """Download a file from Google Drive using gdown."""
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"Downloading {os.path.basename(output_path)} ... (first time only)"):
            gdown.download(url, output_path, quiet=False, fuzzy=True)
    else:
        st.write(f"✓ Using cached {os.path.basename(output_path)}")

# ------------------------------------------------------------------
# 3. Load all data and models with caching
# ------------------------------------------------------------------
@st.cache_resource
def load_flights_data():
    flights_path = "/tmp/flights_dashboard_ready.csv"
    download_file_from_drive(FILE_IDS['flights_dashboard_ready.csv'], flights_path)
    flights = pd.read_csv(flights_path)
    flights['FL_DATE'] = pd.to_datetime(flights['FL_DATE'])
    return flights

@st.cache_resource
def load_airports():
    airports_path = "/tmp/airports_filtered.csv"
    download_file_from_drive(FILE_IDS['airports_filtered.csv'], airports_path)
    return pd.read_csv(airports_path)

@st.cache_resource
def load_valid_routes():
    routes_path = "/tmp/valid_routes.csv"
    download_file_from_drive(FILE_IDS['valid_routes.csv'], routes_path)
    return pd.read_csv(routes_path)

@st.cache_resource
def load_ml_models():
    reg_path = "/tmp/delay_regressor.pkl"
    clf_path = "/tmp/delay_classifier.pkl"
    enc_path = "/tmp/label_encoders.pkl"
    
    download_file_from_drive(FILE_IDS['delay_regressor.pkl'], reg_path)
    download_file_from_drive(FILE_IDS['delay_classifier.pkl'], clf_path)
    download_file_from_drive(FILE_IDS['label_encoders.pkl'], enc_path)
    
    reg = joblib.load(reg_path)
    clf = joblib.load(clf_path)
    encoders = joblib.load(enc_path)
    
    return reg, clf, encoders

# ------------------------------------------------------------------
# 4. Load everything (progress bar, then cache)
# ------------------------------------------------------------------
st.title("✈️ US Flight Delay Dashboard & Predictor")
st.info("📡 Loading data from Google Drive. First load may take 2-3 minutes. Please wait...")

progress_text = st.empty()
progress_bar = st.progress(0)

progress_text.text("Loading flight data...")
flights = load_flights_data()
progress_bar.progress(20)

progress_text.text("Loading airport data...")
airports = load_airports()
progress_bar.progress(40)

progress_text.text("Loading route data...")
valid_routes_df = load_valid_routes()
progress_bar.progress(60)

progress_text.text("Loading ML models...")
reg_model, clf_model, encoders = load_ml_models()
progress_bar.progress(80)

progress_text.text("Preparing data for dashboard...")

# ------------------------------------------------------------------
# 5. Pre‑aggregate for performance (using the flights DataFrame)
#    This runs only once, then caches the results
# ------------------------------------------------------------------
@st.cache_resource
def pre_aggregate(flights, airports):
    # Filter to contiguous US
    contiguous_states = ['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
    airports_continental = airports[airports['state'].isin(contiguous_states)].copy()
    
    # Daily airport activity
    origin_counts = flights.groupby(['FL_DATE', 'ORIGIN']).size().reset_index(name='flight_count')
    origin_counts.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    dest_counts = flights.groupby(['FL_DATE', 'DEST']).size().reset_index(name='flight_count')
    dest_counts.rename(columns={'DEST': 'AIRPORT_CODE'}, inplace=True)
    all_activity = pd.concat([origin_counts, dest_counts])
    daily_airport_activity = all_activity.groupby(['FL_DATE', 'AIRPORT_CODE'])['flight_count'].sum().reset_index()
    
    # Daily delays by airport
    daily_dep_delays = flights.groupby(['FL_DATE', 'ORIGIN']).agg(avg_dep_delay=('DEP_DELAY', 'mean')).reset_index()
    daily_dep_delays.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    
    # Airline + airport aggregate
    airline_airport_agg = flights.groupby(['AIRLINE', 'ORIGIN']).agg(
        avg_delay=('DEP_DELAY', 'mean'),
        flight_count=('DEP_DELAY', 'count')
    ).reset_index()
    airline_airport_agg.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    
    # Daily airline + airport
    daily_airline_airport = flights.groupby(['FL_DATE', 'AIRLINE', 'ORIGIN']).agg(
        avg_delay=('DEP_DELAY', 'mean'),
        flight_count=('DEP_DELAY', 'count')
    ).reset_index()
    daily_airline_airport.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    
    # Airport summary (full, pre‑computed)
    airport_summary = flights.groupby('ORIGIN').agg(
        total_flights=('DEP_DELAY', 'count'),
        avg_delay=('DEP_DELAY', 'mean'),
        median_delay=('DEP_DELAY', 'median'),
        max_delay=('DEP_DELAY', 'max'),
        std_delay=('DEP_DELAY', 'std')
    ).reset_index()
    airport_summary.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    
    # Date range and airline list
    date_range = flights[['FL_DATE']].drop_duplicates().sort_values('FL_DATE')
    airlines_list = sorted(flights['AIRLINE'].unique())
    
    return (daily_airport_activity, daily_dep_delays, daily_airline_airport,
            airline_airport_agg, airport_summary, airports_continental,
            date_range, airlines_list)

# Run pre‑aggregation and clear progress indicators
(daily_airport_activity, daily_dep_delays, daily_airline_airport,
 airline_airport_agg, airport_summary, airports_continental,
 date_range, airlines_list) = pre_aggregate(flights, airports)

progress_bar.progress(100)
progress_text.empty()
progress_bar.empty()

st.success("✅ All data loaded and aggregated successfully!")

# ------------------------------------------------------------------
# 6. Sidebar: Global Date Filter (applied to all tabs)
# ------------------------------------------------------------------
st.sidebar.header("Global Filters")
min_date = date_range['FL_DATE'].min()
max_date = date_range['FL_DATE'].max()

start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

start_datetime = pd.Timestamp(start_date)
end_datetime = pd.Timestamp(end_date)

# Filter the pre‑aggregated data by the selected date range
filtered_activity = daily_airport_activity[
    (daily_airport_activity['FL_DATE'] >= start_datetime) &
    (daily_airport_activity['FL_DATE'] <= end_datetime)
]

filtered_dep_delays = daily_dep_delays[
    (daily_dep_delays['FL_DATE'] >= start_datetime) &
    (daily_dep_delays['FL_DATE'] <= end_datetime)
]

filtered_daily_airline = daily_airline_airport[
    (daily_airline_airport['FL_DATE'] >= start_datetime) &
    (daily_airline_airport['FL_DATE'] <= end_datetime)
]

# Totals for the selected period
total_flights_per_airport = filtered_activity.groupby('AIRPORT_CODE')['flight_count'].sum().reset_index()
total_flights_per_airport.columns = ['AIRPORT_CODE', 'total_flights']

avg_dep_delay_per_airport = filtered_dep_delays.groupby('AIRPORT_CODE')['avg_dep_delay'].mean().reset_index()

airport_metrics = airports_continental.merge(total_flights_per_airport, on='AIRPORT_CODE', how='inner')
airport_metrics = airport_metrics.merge(avg_dep_delay_per_airport, on='AIRPORT_CODE', how='left')
airport_metrics['avg_dep_delay'] = airport_metrics['avg_dep_delay'].round(1).fillna(0)

total_flights_period = filtered_activity['flight_count'].sum()

# Airline + airport for the selected period
airline_airport_filtered_global = filtered_daily_airline.groupby(['AIRLINE', 'AIRPORT_CODE']).agg(
    avg_delay=('avg_delay', 'mean'),
    flight_count=('flight_count', 'sum')
).reset_index()

# Airport percentages
airport_pct = filtered_dep_delays.groupby('AIRPORT_CODE').agg(
    pct_late=('avg_dep_delay', lambda x: (x > 0).mean() * 100),
    pct_early=('avg_dep_delay', lambda x: (x < 0).mean() * 100),
    pct_on_time=('avg_dep_delay', lambda x: (x == 0).mean() * 100)
).reset_index()

airport_summary_filtered = airport_metrics.merge(airport_pct, on='AIRPORT_CODE', how='left')
airport_summary_filtered['pct_late'] = airport_summary_filtered['pct_late'].fillna(0).round(1)
airport_summary_filtered['pct_early'] = airport_summary_filtered['pct_early'].fillna(0).round(1)
airport_summary_filtered['pct_on_time'] = airport_summary_filtered['pct_on_time'].fillna(0).round(1)

# Add max_delay and std_delay from the full summary (static)
airport_summary_extra = airport_summary[['AIRPORT_CODE', 'max_delay', 'std_delay']]
airport_summary_filtered = airport_summary_filtered.merge(airport_summary_extra, on='AIRPORT_CODE', how='left')
airport_summary_filtered['max_delay'] = airport_summary_filtered['max_delay'].fillna(0)
airport_summary_filtered['std_delay'] = airport_summary_filtered['std_delay'].fillna(0)

valid_origins = sorted(valid_routes_df['ORIGIN'].unique())

# ------------------------------------------------------------------
# 7. Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "✈️ Airline Analysis", "🏢 Airport Analysis", "🤖 Delay Predictor"])

# ---------------------------- TAB 1: MAP ---------------------------------
with tab1:
    st.subheader("🗺️ Airport Departure Delay Map (Continental US)")
    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Date Range", f"{start_date} to {end_date}")
    col2.metric("🛫 Total Flights", f"{total_flights_period:,}")
    col3.metric("📍 Active Airports", len(airport_metrics))
    
    heatmap_enabled = st.toggle("🔥 Heatmap Mode", value=False)
    map_center = [39.8283, -98.5795]
    m = folium.Map(location=map_center, zoom_start=4, tiles='CartoDB positron')
    
    if heatmap_enabled:
        heat_data = airport_metrics[['latitude', 'longitude', 'total_flights']].values.tolist()
        HeatMap(heat_data, min_opacity=0.3, radius=15).add_to(m)
    else:
        for _, row in airport_metrics.iterrows():
            d = row['avg_dep_delay']
            color = 'green' if d < 0 else ('blue' if d < 15 else ('orange' if d < 45 else 'red'))
            radius = max(4, min(20, row['total_flights'] / 8000))
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                popup=f"{row['name']} ({row['AIRPORT_CODE']})<br>{row['city']}, {row['state']}<br>{int(row['total_flights']):,} flights<br>{d} min avg delay",
                tooltip=f"{row['AIRPORT_CODE']}: {d} min",
                color=color, fill=True, fill_color=color, fill_opacity=0.7
            ).add_to(m)
    st_folium(m, width=900, height=500)

# ---------------------------- TAB 2: AIRLINE ANALYSIS ---------------------
with tab2:
    st.subheader("✈️ Airline Performance Analysis")
    selected_airline = st.selectbox("Select Airline:", options=airlines_list)
    airline_data = airline_airport_filtered_global[airline_airport_filtered_global['AIRLINE'] == selected_airline].copy()
    airline_data = airline_data.merge(airports_continental[['AIRPORT_CODE', 'name', 'city', 'state']], on='AIRPORT_CODE', how='inner')
    airline_data = airline_data[airline_data['flight_count'] >= 50]
    
    if len(airline_data) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.write("🔴 Worst Airports")
            worst = airline_data.nlargest(10, 'avg_delay')[['AIRPORT_CODE', 'name', 'city', 'avg_delay', 'flight_count']]
            st.dataframe(worst)
        with col2:
            st.write("🟢 Best Airports")
            best = airline_data.nsmallest(10, 'avg_delay')[['AIRPORT_CODE', 'name', 'city', 'avg_delay', 'flight_count']]
            st.dataframe(best)
        fig = px.bar(airline_data, x='AIRPORT_CODE', y='avg_delay', color='avg_delay',
                     title=f"Average Delay by Airport - {selected_airline}", height=500)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data for {selected_airline} in selected date range.")

# ---------------------------- TAB 3: AIRPORT ANALYSIS ---------------------
with tab3:
    st.subheader("🏢 Airport Performance Analysis")
    available = sorted(airport_summary_filtered['AIRPORT_CODE'].unique())
    selected = st.multiselect("Select airports to compare:", available, default=['ATL', 'DFW', 'DEN'])
    if selected:
        comp = airport_summary_filtered[airport_summary_filtered['AIRPORT_CODE'].isin(selected)]
        st.dataframe(comp[['AIRPORT_CODE', 'name', 'city', 'total_flights', 'avg_delay', 'pct_late']])
        fig = px.bar(comp, x='AIRPORT_CODE', y='avg_delay', color='pct_late', title="Airport Comparison")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------- TAB 4: DELAY PREDICTOR ---------------------
with tab4:
    st.subheader("🤖 Flight Delay Predictor")
    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("Origin:", valid_origins[:50])
        dests = valid_routes_df[valid_routes_df['ORIGIN'] == origin]['DEST'].unique()
        destination = st.selectbox("Destination:", sorted(dests)[:50])
        airline = st.selectbox("Airline:", sorted(encoders['AIRLINE'].classes_))
    with col2:
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow = st.selectbox("Day of Week:", range(7), format_func=lambda x: days[x])
        hour = st.slider("Departure Hour:", 0, 23, 12)
        month = st.slider("Month:", 1, 12, 6)
    
    if st.button("🔮 Predict Delay", type="primary"):
        try:
            feat = np.array([[
                encoders['ORIGIN'].transform([origin])[0],
                encoders['DEST'].transform([destination])[0],
                encoders['AIRLINE'].transform([airline])[0],
                hour, dow, month
            ]])
            delay = reg_model.predict(feat)[0]
            prob = clf_model.predict_proba(feat)[0][1]
            st.success(f"**Expected Departure Delay:** {delay:.0f} minutes")
            st.info(f"**Probability of being late (>15 min):** {prob*100:.0f}%")
            if delay < 0:
                st.success("🟢 Expected to depart EARLY")
            elif delay < 15:
                st.success("🟢 Expected ON TIME")
            elif delay < 45:
                st.warning("🟡 Expected MINOR delay")
            else:
                st.error("🔴 Expected SIGNIFICANT delay")
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Data source: US Flight Delays 2019-2023 | Continental US only | Data & models loaded from Google Drive using gdown")
