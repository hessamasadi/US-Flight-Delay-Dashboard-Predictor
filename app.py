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
import time

st.set_page_config(layout="wide", page_title="Flight Delay Dashboard")

BASE_URL = "https://huggingface.co/hessamedin/flight-delay-models/resolve/main"
FLIGHTS_URL = f"{BASE_URL}/flights_dashboard_ready.parquet"
AIRPORTS_URL = f"{BASE_URL}/airports_filtered.csv"
VALID_ROUTES_URL = f"{BASE_URL}/valid_routes.csv"
REG_MODEL_URL = f"{BASE_URL}/delay_regressor_compressed.pkl"
CLF_MODEL_URL = f"{BASE_URL}/delay_classifier_compressed.pkl"
ENCODERS_URL = f"{BASE_URL}/label_encoders.pkl"

@st.cache_data(ttl=86400)
def load_flights():
    columns = ['FL_DATE', 'ORIGIN', 'DEST', 'AIRLINE', 'DEP_DELAY', 'ARR_DELAY', 'ELAPSED_TIME', 'DISTANCE']
    df = pd.read_parquet(FLIGHTS_URL, columns=columns)
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    for col in ['DEP_DELAY', 'ARR_DELAY', 'ELAPSED_TIME', 'DISTANCE']:
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

st.title("✈️ US Flight Delay Dashboard")

progress_bar = st.progress(0, text="Initializing...")
status_text = st.empty()

status_text.text("Step 1/4: Loading flight data...")
progress_bar.progress(10, text="Loading flights...")
flights = load_flights()
progress_bar.progress(25, text="Flights loaded ✓")
time.sleep(0.2)

status_text.text("Step 2/4: Loading airport data...")
progress_bar.progress(35, text="Loading airports...")
airports = load_airports()
progress_bar.progress(50, text="Airports loaded ✓")
time.sleep(0.2)

status_text.text("Step 3/4: Loading route data...")
progress_bar.progress(60, text="Loading routes...")
valid_routes_df = load_valid_routes()
progress_bar.progress(75, text="Routes loaded ✓")
time.sleep(0.2)

status_text.text("Step 4/4: Loading ML models...")
progress_bar.progress(85, text="Loading models...")
reg_model, clf_model, encoders = load_models()
progress_bar.progress(100, text="Complete! ✓")
time.sleep(0.3)

progress_bar.empty()
status_text.empty()
st.success("✅ All data loaded successfully!")

gc.collect()

contiguous_states = ['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA',
                     'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV',
                     'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD',
                     'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
airports = airports[airports['state'].isin(contiguous_states)]

with st.sidebar:
    st.header("Filters")
    min_date = flights['FL_DATE'].min()
    max_date = flights['FL_DATE'].max()
    start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

filtered_flights = flights[(flights['FL_DATE'] >= start_ts) & (flights['FL_DATE'] <= end_ts)]
total_flights = len(filtered_flights)
active_airports = filtered_flights['ORIGIN'].nunique()

@st.cache_data
def get_airport_metrics(_filtered_flights, _airports):
    counts = _filtered_flights.groupby('ORIGIN').size().reset_index(name='total_flights')
    delays = _filtered_flights.groupby('ORIGIN')['DEP_DELAY'].mean().reset_index(name='avg_delay')
    metrics = counts.merge(delays, on='ORIGIN', how='left')
    metrics.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    metrics = metrics.merge(_airports, on='AIRPORT_CODE', how='inner')
    metrics['avg_delay'] = metrics['avg_delay'].round(1).fillna(0)
    return metrics

airport_metrics = get_airport_metrics(filtered_flights, airports)

tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map", "✈️ Airlines", "🏢 Airports", "🤖 Predictor"])

with tab1:
    st.subheader("Airport Departure Delay Map")
    c1, c2, c3 = st.columns(3)
    c1.metric("Date Range", f"{start_date} to {end_date}")
    c2.metric("Flights", f"{total_flights:,}")
    c3.metric("Airports", active_airports)
    
    heatmap = st.toggle("Heatmap Mode", value=False)
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles='CartoDB positron')
    
    if heatmap:
        heat_data = airport_metrics[['latitude', 'longitude', 'total_flights']].values.tolist()
        HeatMap(heat_data, min_opacity=0.3, radius=15).add_to(m)
    else:
        for _, row in airport_metrics.iterrows():
            d = row['avg_delay']
            if d < 0: color = 'green'
            elif d < 15: color = 'blue'
            elif d < 45: color = 'orange'
            else: color = 'red'
            radius = max(4, min(20, row['total_flights'] / 1000))
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                popup=f"{row['name']} ({row['AIRPORT_CODE']})<br>{int(row['total_flights']):,} flights<br>{d} min",
                color=color, fill=True, fill_color=color
            ).add_to(m)
    
    st_folium(m, width=900, height=500)

with tab2:
    st.subheader("Airline Performance")
    airlines = sorted(filtered_flights['AIRLINE'].unique())
    selected = st.selectbox("Select Airline:", airlines)
    
    data = filtered_flights[filtered_flights['AIRLINE'] == selected].groupby('ORIGIN').agg(
        delay=('DEP_DELAY', 'mean'),
        flights=('DEP_DELAY', 'count')
    ).reset_index()
    data = data.merge(airports[['AIRPORT_CODE', 'name']], left_on='ORIGIN', right_on='AIRPORT_CODE')
    data = data[data['flights'] >= 50]
    
    if len(data) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.write("🔴 Worst Airports")
            st.dataframe(data.nlargest(10, 'delay')[['AIRPORT_CODE', 'name', 'delay', 'flights']])
        with col2:
            st.write("🟢 Best Airports")
            st.dataframe(data.nsmallest(10, 'delay')[['AIRPORT_CODE', 'name', 'delay', 'flights']])
        
        fig = px.bar(data, x='AIRPORT_CODE', y='delay', title=f"{selected} - Delay by Airport")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Airport Comparison")
    available = sorted(airport_metrics['AIRPORT_CODE'].unique())
    selected = st.multiselect("Select Airports:", available, default=['ATL', 'DFW', 'DEN'])
    if selected:
        comp = airport_metrics[airport_metrics['AIRPORT_CODE'].isin(selected)]
        st.dataframe(comp[['AIRPORT_CODE', 'name', 'city', 'total_flights', 'avg_delay']])
        
        fig = px.bar(comp, x='AIRPORT_CODE', y='avg_delay', title="Airport Delay Comparison")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Delay Predictor")
    origins = sorted(valid_routes_df['ORIGIN'].unique())
    origin = st.selectbox("Origin:", origins[:50])
    dests = valid_routes_df[valid_routes_df['ORIGIN'] == origin]['DEST'].unique()
    dest = st.selectbox("Destination:", sorted(dests)[:50])
    airline = st.selectbox("Airline:", sorted(encoders['AIRLINE'].classes_))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow = st.selectbox("Day of Week:", range(7), format_func=lambda x: days[x])
    hour = st.slider("Hour:", 0, 23, 12)
    month = st.slider("Month:", 1, 12, 6)
    
    if st.button("Predict Delay"):
        try:
            feat = np.array([[
                encoders['ORIGIN'].transform([origin])[0],
                encoders['DEST'].transform([dest])[0],
                encoders['AIRLINE'].transform([airline])[0],
                hour, dow, month
            ]])
            delay = reg_model.predict(feat)[0]
            prob = clf_model.predict_proba(feat)[0][1]
            st.success(f"Expected delay: {delay:.0f} minutes")
            st.info(f"Probability of being late (>15 min): {prob*100:.0f}%")
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.caption("Data source: US Flight Delays 2019–2023 | Optimized for performance")
