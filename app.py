import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from datetime import datetime
import plotly.express as px
import joblib
import numpy as np
import requests
from io import BytesIO
import gc

st.set_page_config(layout="wide", page_title="Flight Delay Dashboard")

# Hugging Face URLs (using your compressed/aggregated files)
BASE_URL = "https://huggingface.co/hessamedin/flight-delay-models/resolve/main"

# Use daily_flight_aggregated.csv (compressed version) instead of huge CSV
FLIGHTS_URL = f"{BASE_URL}/daily_flight_aggregated.csv"
AIRPORTS_URL = f"{BASE_URL}/airports_filtered.csv"
VALID_ROUTES_URL = f"{BASE_URL}/valid_routes.csv"
REG_MODEL_URL = f"{BASE_URL}/delay_regressor_compressed.pkl"
CLF_MODEL_URL = f"{BASE_URL}/delay_classifier_compressed.pkl"
ENCODERS_URL = f"{BASE_URL}/label_encoders.pkl"

# Memory-optimized data loading
@st.cache_data(ttl=86400)
def load_flights():
    # Read aggregated CSV (much smaller than original)
    df = pd.read_csv(FLIGHTS_URL)
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    
    # Downcast numeric columns to save memory
    for col in ['total_flights', 'avg_dep_delay', 'avg_arr_delay', 'median_dep_delay', 'pct_late', 'pct_early', 'pct_on_time']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

@st.cache_data(ttl=86400)
def load_airports():
    df = pd.read_csv(AIRPORTS_URL)
    return df[['AIRPORT_CODE', 'name', 'city', 'state', 'latitude', 'longitude']]

@st.cache_data(ttl=86400)
def load_valid_routes():
    return pd.read_csv(VALID_ROUTES_URL)

@st.cache_resource(ttl=86400)
def load_models():
    reg = joblib.load(BytesIO(requests.get(REG_MODEL_URL, timeout=180).content))
    clf = joblib.load(BytesIO(requests.get(CLF_MODEL_URL, timeout=180).content))
    encoders = joblib.load(BytesIO(requests.get(ENCODERS_URL, timeout=180).content))
    return reg, clf, encoders

# Load everything
st.title("✈️ US Flight Delay Dashboard & Predictor")

with st.spinner("Loading data (first time takes 1-2 minutes)..."):
    flights = load_flights()
    airports = load_airports()
    valid_routes_df = load_valid_routes()
    reg_model, clf_model, encoders = load_models()
    gc.collect()

st.success("✅ All data and models ready!")

# Filter to continental US
contiguous_states = ['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA',
                     'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV',
                     'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD',
                     'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
airports_continental = airports[airports['state'].isin(contiguous_states)].copy()

# Sidebar filters
st.sidebar.header("Global Filters")
min_date = flights['FL_DATE'].min()
max_date = flights['FL_DATE'].max()
start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

# Filter flights by date range
filtered_flights = flights[(flights['FL_DATE'] >= start_ts) & (flights['FL_DATE'] <= end_ts)]

total_flights = filtered_flights['total_flights'].sum() if 'total_flights' in filtered_flights.columns else len(filtered_flights)
active_airports = filtered_flights['ORIGIN'].nunique()

# Pre-aggregate airport metrics
@st.cache_data
def get_airport_metrics(_filtered_flights, _airports):
    # Aggregate flights and delays per airport
    airport_data = _filtered_flights.groupby('ORIGIN').agg(
        total_flights=('total_flights', 'sum'),
        avg_delay=('avg_dep_delay', 'mean')
    ).reset_index()
    airport_data.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    
    metrics = airport_data.merge(_airports, on='AIRPORT_CODE', how='inner')
    metrics['avg_delay'] = metrics['avg_delay'].round(1).fillna(0)
    return metrics

airport_metrics = get_airport_metrics(filtered_flights, airports_continental)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "✈️ Airline Analysis", "🏢 Airport Analysis", "🤖 Delay Predictor"])

# ---------------------------- TAB 1 : MAP ---------------------------------
with tab1:
    st.subheader("🗺️ Airport Departure Delay Map (Continental US)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Date Range", f"{start_date} to {end_date}")
    col2.metric("🛫 Total Flights", f"{int(total_flights):,}")
    col3.metric("📍 Active Airports", active_airports)

    heatmap_enabled = st.toggle("🔥 Heatmap Mode", value=False)
    map_center = [39.8283, -98.5795]
    m = folium.Map(location=map_center, zoom_start=4, tiles='CartoDB positron')

    if heatmap_enabled:
        heat_data = airport_metrics[['latitude', 'longitude', 'total_flights']].values.tolist()
        HeatMap(heat_data, min_opacity=0.3, radius=15).add_to(m)
    else:
        for _, row in airport_metrics.iterrows():
            d = row['avg_delay']
            if d < 0:
                color = 'green'
            elif d < 15:
                color = 'blue'
            elif d < 45:
                color = 'orange'
            else:
                color = 'red'
            radius = max(4, min(20, row['total_flights'] / 5000))
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                popup=f"{row['name']} ({row['AIRPORT_CODE']})<br>{int(row['total_flights']):,} flights<br>{d} min avg delay",
                tooltip=f"{row['AIRPORT_CODE']}: {d} min",
                color=color, fill=True, fill_color=color, fill_opacity=0.7
            ).add_to(m)

    st_folium(m, width=900, height=500)

# ---------------------------- TAB 2 : AIRLINE ANALYSIS --------------------
with tab2:
    st.subheader("✈️ Airline Performance Analysis")
    airlines_list = sorted(filtered_flights['AIRLINE'].unique())
    selected_airline = st.selectbox("Select Airline:", airlines_list)

    airline_data = filtered_flights[filtered_flights['AIRLINE'] == selected_airline].groupby('ORIGIN').agg(
        avg_delay=('avg_dep_delay', 'mean'),
        flight_count=('total_flights', 'sum')
    ).reset_index()
    airline_data = airline_data.merge(airports_continental[['AIRPORT_CODE', 'name', 'city', 'state']],
                                      left_on='ORIGIN', right_on='AIRPORT_CODE', how='inner')
    airline_data = airline_data[airline_data['flight_count'] >= 50]

    if len(airline_data) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.write("🔴 Worst Airports")
            worst = airline_data.nlargest(10, 'avg_delay')[['AIRPORT_CODE', 'name', 'city', 'avg_delay', 'flight_count']]
            st.dataframe(worst, use_container_width=True)
        with col2:
            st.write("🟢 Best Airports")
            best = airline_data.nsmallest(10, 'avg_delay')[['AIRPORT_CODE', 'name', 'city', 'avg_delay', 'flight_count']]
            st.dataframe(best, use_container_width=True)

        fig = px.bar(airline_data, x='AIRPORT_CODE', y='avg_delay', color='avg_delay',
                     title=f"Average Delay by Airport – {selected_airline}", height=500)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data for {selected_airline} in the selected date range.")

# ---------------------------- TAB 3 : AIRPORT ANALYSIS --------------------
with tab3:
    st.subheader("🏢 Airport Performance Analysis")
    available = sorted(airport_metrics['AIRPORT_CODE'].unique())
    selected = st.multiselect("Select airports to compare:", available, default=['ATL', 'DFW', 'DEN'])

    if selected:
        comp_data = airport_metrics[airport_metrics['AIRPORT_CODE'].isin(selected)].copy()
        
        # Calculate additional metrics
        for airport in selected:
            airport_flights = filtered_flights[filtered_flights['ORIGIN'] == airport]
            pct_late = (airport_flights['avg_dep_delay'] > 15).mean() * 100 if len(airport_flights) > 0 else 0
            comp_data.loc[comp_data['AIRPORT_CODE'] == airport, 'pct_late'] = round(pct_late, 1)

        comp_data['pct_late'] = comp_data['pct_late'].fillna(0).round(1)

        st.dataframe(comp_data[['AIRPORT_CODE', 'name', 'city', 'total_flights', 'avg_delay', 'pct_late']], use_container_width=True)
        
        fig = px.bar(comp_data, x='AIRPORT_CODE', y='avg_delay', color='pct_late',
                     title="Airport Comparison – Average Delay & % Late", height=500)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------- TAB 4 : DELAY PREDICTOR ---------------------
with tab4:
    st.subheader("🤖 Flight Delay Predictor")
    st.markdown("Predicts departure delay based on historical patterns for your route, airline, and time.")

    valid_origins = sorted(valid_routes_df['ORIGIN'].unique())
    col1, col2 = st.columns(2)

    with col1:
        origin = st.selectbox("Origin Airport:", valid_origins[:50])
        dest_options = valid_routes_df[valid_routes_df['ORIGIN'] == origin]['DEST'].unique()
        destination = st.selectbox("Destination Airport:", sorted(dest_options)[:50])
        airline = st.selectbox("Airline:", sorted(encoders['AIRLINE'].classes_))

    with col2:
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_of_week = st.selectbox("Day of Week:", range(7), format_func=lambda x: days[x])
        hour = st.slider("Departure Hour (24h):", 0, 23, 12)
        month = st.slider("Month:", 1, 12, 6)

    if st.button("🔮 Predict Delay", type="primary", use_container_width=True):
        try:
            features = np.array([[
                encoders['ORIGIN'].transform([origin])[0],
                encoders['DEST'].transform([destination])[0],
                encoders['AIRLINE'].transform([airline])[0],
                hour, day_of_week, month
            ]])
            pred_delay = reg_model.predict(features)[0]
            prob_late = clf_model.predict_proba(features)[0][1]

            st.success(f"**Expected departure delay:** {pred_delay:.0f} minutes")
            st.info(f"**Probability of being late (>15 min):** {prob_late*100:.0f}%")

            if pred_delay < 0:
                st.success("🟢 Expected to depart EARLY")
            elif pred_delay < 15:
                st.success("🟢 Expected ON TIME")
            elif pred_delay < 45:
                st.warning("🟡 Expected MINOR delay (15–45 min)")
            else:
                st.error("🔴 Expected SIGNIFICANT delay (>45 min)")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Try another route – the model may not have seen this origin–destination pair.")

# ------------------------------------------------------------------
st.markdown("---")
st.caption("Data source: US Flight Delays 2019–2023 (Kaggle) | Hosted on Hugging Face | Optimized for memory & performance")
