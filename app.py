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
import requests
from io import BytesIO

st.set_page_config(layout="wide", page_title="Flight Delay Dashboard")

# ------------------------------------------------------------------
# 1. Direct Hugging Face URLs (raw file access)
# ------------------------------------------------------------------
BASE_URL = "https://huggingface.co/hessamedin/flight-delay-models/resolve/main"

CSV_URL = f"{BASE_URL}/flights_dashboard_ready.csv"
AIRPORTS_URL = f"{BASE_URL}/airports_filtered.csv"
VALID_ROUTES_URL = f"{BASE_URL}/valid_routes.csv"
REG_MODEL_URL = f"{BASE_URL}/delay_regressor_compressed.pkl"
CLF_MODEL_URL = f"{BASE_URL}/delay_classifier_compressed.pkl"
ENCODERS_URL = f"{BASE_URL}/label_encoders.pkl"

# ------------------------------------------------------------------
# 2. Cached data loaders (first load ~2‑3 min, then instant)
# ------------------------------------------------------------------
@st.cache_data
def load_flights():
    df = pd.read_csv(CSV_URL)
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    return df

@st.cache_data
def load_airports():
    return pd.read_csv(AIRPORTS_URL)

@st.cache_data
def load_valid_routes():
    return pd.read_csv(VALID_ROUTES_URL)

@st.cache_resource
def load_models():
    reg = joblib.load(BytesIO(requests.get(REG_MODEL_URL).content))
    clf = joblib.load(BytesIO(requests.get(CLF_MODEL_URL).content))
    encoders = joblib.load(BytesIO(requests.get(ENCODERS_URL).content))
    return reg, clf, encoders

# ------------------------------------------------------------------
# 3. Load everything once when the app starts
# ------------------------------------------------------------------
st.title("✈️ US Flight Delay Dashboard & Predictor")
st.info("📡 Loading data from Hugging Face. First load may take 2‑3 minutes. Please wait...")

with st.spinner("Loading flights, airports, routes and models..."):
    flights = load_flights()
    airports = load_airports()
    valid_routes_df = load_valid_routes()
    reg_model, clf_model, encoders = load_models()

st.success("✅ All data and models ready!")

# ------------------------------------------------------------------
# 4. Filter airports to contiguous US (for cleaner map)
# ------------------------------------------------------------------
contiguous_states = [
    'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA',
    'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV',
    'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD',
    'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]
airports_continental = airports[airports['state'].isin(contiguous_states)].copy()

# ------------------------------------------------------------------
# 5. Sidebar: global date range filter (applies to all charts)
# ------------------------------------------------------------------
st.sidebar.header("Global Filters")
min_date = flights['FL_DATE'].min()
max_date = flights['FL_DATE'].max()
start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

filtered_flights = flights[(flights['FL_DATE'] >= start_ts) & (flights['FL_DATE'] <= end_ts)]

total_flights = len(filtered_flights)
active_airports = filtered_flights['ORIGIN'].nunique()

# ------------------------------------------------------------------
# 6. Pre‑aggregate for map & airline analysis (cached per filter)
# ------------------------------------------------------------------
@st.cache_data
def get_airport_metrics(_filtered_flights, airports_cont):
    # Flight counts and average delay per airport
    airport_counts = _filtered_flights.groupby('ORIGIN').size().reset_index(name='total_flights')
    airport_delays = _filtered_flights.groupby('ORIGIN')['DEP_DELAY'].mean().reset_index(name='avg_delay')
    
    metrics = airport_counts.merge(airport_delays, on='ORIGIN', how='left')
    metrics.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    metrics = metrics.merge(airports_cont, on='AIRPORT_CODE', how='inner')
    metrics['avg_delay'] = metrics['avg_delay'].round(1).fillna(0)
    return metrics

airport_metrics = get_airport_metrics(filtered_flights, airports_continental)

# ------------------------------------------------------------------
# 7. Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "✈️ Airline Analysis", "🏢 Airport Analysis", "🤖 Delay Predictor"])

# ---------------------------- TAB 1 : MAP ---------------------------------
with tab1:
    st.subheader("🗺️ Airport Departure Delay Map (Continental US)")
    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Date Range", f"{start_date} to {end_date}")
    col2.metric("🛫 Total Flights", f"{total_flights:,}")
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
            radius = max(4, min(20, row['total_flights'] / 1000))
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
        avg_delay=('DEP_DELAY', 'mean'),
        flight_count=('DEP_DELAY', 'count')
    ).reset_index()
    airline_data = airline_data.merge(airports_continental[['AIRPORT_CODE', 'name', 'city', 'state']],
                                      left_on='ORIGIN', right_on='AIRPORT_CODE', how='inner')
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
        # compute % late on the fly
        late_pct = filtered_flights[filtered_flights['ORIGIN'].isin(selected)].groupby('ORIGIN')['DEP_DELAY'].apply(
            lambda x: (x > 15).mean() * 100).reset_index(name='pct_late')
        comp_data = comp_data.merge(late_pct, left_on='AIRPORT_CODE', right_on='ORIGIN', how='left')
        comp_data['pct_late'] = comp_data['pct_late'].fillna(0).round(1)

        st.dataframe(comp_data[['AIRPORT_CODE', 'name', 'city', 'total_flights', 'avg_delay', 'pct_late']])
        fig = px.bar(comp_data, x='AIRPORT_CODE', y='avg_delay', color='pct_late',
                     title="Airport Comparison – Average Delay & % Late")
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
st.caption("Data source: US Flight Delays 2019–2023 (Kaggle) | Hosted on Hugging Face | Models: Random Forest with class balancing (64% recall for late flights)")
