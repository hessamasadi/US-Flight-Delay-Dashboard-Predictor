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

# Google Drive file IDs
FILE_IDS = {
    'flights_dashboard_ready.csv': '1nRh-fpz6y1iKgkAJdVg_sagKwDw6Pi6X',
    'airports_filtered.csv': '1IIiIL0cWj1aJSNi42l1D5GvJoBBCcWmN',
    'delay_regressor.pkl': '1KkYvOOKegz6_nckdrTLY0NQe1E-3cLzM',
    'delay_classifier.pkl': '1LUNDFtOjJMTYk2tvkQW2KtaaotWv9QYX',
    'label_encoders.pkl': '13famhi3vbswMqzwaLeeTFSeVnbVLucy_',
    'valid_routes.csv': '1LOhz9kycmrviSuqLtWPoDV1CKDyE-60O'
}

@st.cache_data
def download_file_gdown(filename, file_id):
    """Download file from Google Drive using gdown (handles large files)"""
    url = f"https://drive.google.com/uc?id={file_id}"
    output = f"/tmp/{filename}"
    
    # Check if file already exists in /tmp
    if os.path.exists(output):
        st.write(f"  ✓ Using cached {filename}")
    else:
        st.write(f"  📥 Downloading {filename} from Google Drive...")
        # gdown.download returns the output path
        gdown.download(url, output, quiet=False, fuzzy=True)
        
        # Verify file was downloaded
        if os.path.getsize(output) < 1000:
            st.error(f"  ❌ {filename} download failed - file too small")
            raise ValueError(f"Download failed for {filename}")
    
    # Load based on file type
    if filename.endswith('.csv'):
        df = pd.read_csv(output)
        st.write(f"  ✓ {filename}: {len(df):,} rows, {len(df.columns)} columns")
        return df
    else:
        obj = joblib.load(output)
        st.write(f"  ✓ {filename} loaded")
        return obj

@st.cache_data
def load_aggregated_data():
    st.write("🔄 Loading and aggregating data...")
    
    flights = download_file_gdown('flights_dashboard_ready.csv', FILE_IDS['flights_dashboard_ready.csv'])
    
    # Verify data
    if len(flights) == 0:
        st.error("Downloaded file is empty. Please check Google Drive permissions.")
        st.stop()
    
    flights['FL_DATE'] = pd.to_datetime(flights['FL_DATE'])
    airports = download_file_gdown('airports_filtered.csv', FILE_IDS['airports_filtered.csv'])
    
    st.write("  Aggregating flight counts...")
    origin_counts = flights.groupby(['FL_DATE', 'ORIGIN']).size().reset_index(name='flight_count')
    origin_counts.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    
    dest_counts = flights.groupby(['FL_DATE', 'DEST']).size().reset_index(name='flight_count')
    dest_counts.rename(columns={'DEST': 'AIRPORT_CODE'}, inplace=True)
    
    all_activity = pd.concat([origin_counts, dest_counts])
    daily_airport_activity = all_activity.groupby(['FL_DATE', 'AIRPORT_CODE'])['flight_count'].sum().reset_index()
    
    st.write("  Aggregating delays...")
    daily_dep_delays = flights.groupby(['FL_DATE', 'ORIGIN']).agg(
        avg_dep_delay=('DEP_DELAY', 'mean')
    ).reset_index()
    daily_dep_delays.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    
    airline_airport_agg = flights.groupby(['AIRLINE', 'ORIGIN']).agg(
        avg_delay=('DEP_DELAY', 'mean'),
        flight_count=('DEP_DELAY', 'count')
    ).reset_index()
    airline_airport_agg.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    
    daily_airline_airport = flights.groupby(['FL_DATE', 'AIRLINE', 'ORIGIN']).agg(
        avg_delay=('DEP_DELAY', 'mean'),
        flight_count=('DEP_DELAY', 'count')
    ).reset_index()
    daily_airline_airport.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    
    airport_summary = flights.groupby('ORIGIN').agg(
        total_flights=('DEP_DELAY', 'count'),
        avg_delay=('DEP_DELAY', 'mean'),
        median_delay=('DEP_DELAY', 'median'),
        max_delay=('DEP_DELAY', 'max'),
        std_delay=('DEP_DELAY', 'std')
    ).reset_index()
    airport_summary.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    
    date_range = flights[['FL_DATE']].drop_duplicates().sort_values('FL_DATE')
    airlines_list = sorted(flights['AIRLINE'].unique())
    
    st.write("✅ Data aggregation complete!")
    
    return (daily_airport_activity, daily_dep_delays, daily_airline_airport, 
            airline_airport_agg, airport_summary, airports, date_range, airlines_list)

@st.cache_resource
def load_ml_models():
    st.write("🔄 Loading ML models...")
    reg = download_file_gdown('delay_regressor.pkl', FILE_IDS['delay_regressor.pkl'])
    clf = download_file_gdown('delay_classifier.pkl', FILE_IDS['delay_classifier.pkl'])
    encoders = download_file_gdown('label_encoders.pkl', FILE_IDS['label_encoders.pkl'])
    st.write("✅ ML models loaded!")
    return reg, clf, encoders

@st.cache_data
def load_valid_routes():
    return download_file_gdown('valid_routes.csv', FILE_IDS['valid_routes.csv'])

# Main app
st.title("✈️ US Flight Delay Dashboard & Predictor")
st.info("📡 Loading data from Google Drive. First load takes 2-3 minutes. Please wait...")

try:
    # Load all data
    with st.spinner("Loading flight data (2-3 minutes on first visit)..."):
        (daily_airport_activity, daily_dep_delays, daily_airline_airport, 
         airline_airport_agg, airport_summary, airports, date_range, airlines_list) = load_aggregated_data()
    
    # Filter airports to contiguous US
    contiguous_states = ['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
    airports_continental = airports[airports['state'].isin(contiguous_states)].copy()
    
    st.sidebar.header("Global Filters")
    min_date = date_range['FL_DATE'].min()
    max_date = date_range['FL_DATE'].max()
    
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    start_datetime = pd.Timestamp(start_date)
    end_datetime = pd.Timestamp(end_date)
    
    # Filter data
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
    
    total_flights_per_airport = filtered_activity.groupby('AIRPORT_CODE')['flight_count'].sum().reset_index()
    total_flights_per_airport.columns = ['AIRPORT_CODE', 'total_flights']
    
    avg_dep_delay_per_airport = filtered_dep_delays.groupby('AIRPORT_CODE')['avg_dep_delay'].mean().reset_index()
    
    airport_metrics = airports_continental.merge(total_flights_per_airport, on='AIRPORT_CODE', how='inner')
    airport_metrics = airport_metrics.merge(avg_dep_delay_per_airport, on='AIRPORT_CODE', how='left')
    airport_metrics['avg_dep_delay'] = airport_metrics['avg_dep_delay'].round(1).fillna(0)
    
    total_flights_period = filtered_activity['flight_count'].sum()
    
    airline_airport_filtered_global = filtered_daily_airline.groupby(['AIRLINE', 'AIRPORT_CODE']).agg(
        avg_delay=('avg_delay', 'mean'),
        flight_count=('flight_count', 'sum')
    ).reset_index()
    
    airport_summary_date_filtered = filtered_dep_delays.groupby('AIRPORT_CODE').agg(
        avg_delay=('avg_dep_delay', 'mean'),
        total_flights=('AIRPORT_CODE', 'count')
    ).reset_index()
    
    airport_pct = filtered_dep_delays.groupby('AIRPORT_CODE').agg(
        pct_late=('avg_dep_delay', lambda x: (x > 0).mean() * 100),
        pct_early=('avg_dep_delay', lambda x: (x < 0).mean() * 100),
        pct_on_time=('avg_dep_delay', lambda x: (x == 0).mean() * 100)
    ).reset_index()
    
    airport_summary_filtered = airport_summary_date_filtered.merge(
        airports_continental[['AIRPORT_CODE', 'name', 'city', 'state', 'latitude', 'longitude']], 
        on='AIRPORT_CODE', 
        how='inner'
    )
    airport_summary_filtered = airport_summary_filtered.merge(airport_pct, on='AIRPORT_CODE', how='left')
    airport_summary_filtered['pct_late'] = airport_summary_filtered['pct_late'].fillna(0).round(1)
    airport_summary_filtered['pct_early'] = airport_summary_filtered['pct_early'].fillna(0).round(1)
    airport_summary_filtered['pct_on_time'] = airport_summary_filtered['pct_on_time'].fillna(0).round(1)
    
    airport_summary_extra = airport_summary[['AIRPORT_CODE', 'max_delay', 'std_delay']]
    airport_summary_filtered = airport_summary_filtered.merge(airport_summary_extra, on='AIRPORT_CODE', how='left')
    airport_summary_filtered['max_delay'] = airport_summary_filtered['max_delay'].fillna(0)
    airport_summary_filtered['std_delay'] = airport_summary_filtered['std_delay'].fillna(0)
    
    valid_routes_df = load_valid_routes()
    valid_origins = sorted(valid_routes_df['ORIGIN'].unique())
    
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "✈️ Airline Analysis", "🏢 Airport Analysis", "🤖 Delay Predictor"])
    
    # ==================== TAB 1: MAP VIEW ====================
    with tab1:
        st.subheader("🗺️ Airport Departure Delay Map")
        
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
                delay_value = row['avg_dep_delay']
                
                if delay_value < 0:
                    color = 'green'
                elif delay_value < 15:
                    color = 'blue'
                elif delay_value < 45:
                    color = 'orange'
                else:
                    color = 'red'
                
                marker_radius = max(4, min(20, row['total_flights'] / 8000))
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=marker_radius,
                    popup=f"{row['name']} ({row['AIRPORT_CODE']})<br>{row['city']}, {row['state']}<br>{int(row['total_flights']):,} flights<br>{delay_value} min avg delay",
                    tooltip=f"{row['AIRPORT_CODE']}: {delay_value} min",
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7
                ).add_to(m)
        
        st_folium(m, width=900, height=500)
    
    # ==================== TAB 2: AIRLINE ANALYSIS ====================
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
    
    # ==================== TAB 3: AIRPORT ANALYSIS ====================
    with tab3:
        st.subheader("🏢 Airport Performance Analysis")
        
        available_airports = sorted(airport_summary_filtered['AIRPORT_CODE'].unique())
        selected_airports = st.multiselect("Select airports:", available_airports, default=['ATL', 'DFW', 'DEN'])
        
        if selected_airports:
            comparison_data = airport_summary_filtered[airport_summary_filtered['AIRPORT_CODE'].isin(selected_airports)]
            st.dataframe(comparison_data[['AIRPORT_CODE', 'name', 'city', 'total_flights', 'avg_delay', 'pct_late']])
            
            fig = px.bar(comparison_data, x='AIRPORT_CODE', y='avg_delay', color='pct_late', title="Airport Comparison")
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 4: DELAY PREDICTOR ====================
    with tab4:
        st.subheader("🤖 Flight Delay Predictor")
        
        try:
            reg_model, clf_model, encoders = load_ml_models()
            
            col1, col2 = st.columns(2)
            
            with col1:
                origin = st.selectbox("Origin:", valid_origins[:50])
                valid_dests = valid_routes_df[valid_routes_df['ORIGIN'] == origin]['DEST'].unique()
                destination = st.selectbox("Destination:", sorted(valid_dests)[:50])
                airline = st.selectbox("Airline:", sorted(encoders['AIRLINE'].classes_))
            
            with col2:
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_of_week = st.selectbox("Day of Week:", range(7), format_func=lambda x: days[x])
                hour = st.slider("Departure Hour:", 0, 23, 12)
                month = st.slider("Month:", 1, 12, 6)
            
            if st.button("🔮 Predict Delay", type="primary"):
                features = np.array([[
                    encoders['ORIGIN'].transform([origin])[0],
                    encoders['DEST'].transform([destination])[0],
                    encoders['AIRLINE'].transform([airline])[0],
                    hour, day_of_week, month
                ]])
                
                pred_delay = reg_model.predict(features)[0]
                pred_late_prob = clf_model.predict_proba(features)[0][1]
                
                st.success(f"**Expected Departure Delay:** {pred_delay:.0f} minutes")
                st.info(f"**Probability of being late (>15 min):** {pred_late_prob*100:.0f}%")
                
                if pred_delay < 0:
                    st.balloons()
                    st.success("🟢 Expected to depart EARLY")
                elif pred_delay < 15:
                    st.success("🟢 Expected ON TIME")
                elif pred_delay < 45:
                    st.warning("🟡 Expected MINOR delay")
                else:
                    st.error("🔴 Expected SIGNIFICANT delay")
                    
        except Exception as e:
            st.error(f"Models not ready: {str(e)}")

except Exception as e:
    st.error(f"Failed to load: {str(e)}")
