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

st.set_page_config(layout="wide", page_title="Flight Delay Dashboard")

@st.cache_data
def load_aggregated_data():
    flights = pd.read_csv(r'C:\Users\HessaM\Desktop\dataset\flights_dashboard_ready.csv')
    flights['FL_DATE'] = pd.to_datetime(flights['FL_DATE'])
    
    airports = pd.read_csv(r'C:\Users\HessaM\Desktop\dataset\airports_filtered.csv')
    
    origin_counts = flights.groupby(['FL_DATE', 'ORIGIN']).size().reset_index(name='flight_count')
    origin_counts.rename(columns={'ORIGIN': 'AIRPORT_CODE'}, inplace=True)
    
    dest_counts = flights.groupby(['FL_DATE', 'DEST']).size().reset_index(name='flight_count')
    dest_counts.rename(columns={'DEST': 'AIRPORT_CODE'}, inplace=True)
    
    all_activity = pd.concat([origin_counts, dest_counts])
    daily_airport_activity = all_activity.groupby(['FL_DATE', 'AIRPORT_CODE'])['flight_count'].sum().reset_index()
    
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
    
    return (daily_airport_activity, daily_dep_delays, daily_airline_airport, 
            airline_airport_agg, airport_summary, airports, date_range, airlines_list)

@st.cache_resource
def load_ml_models():
    reg = joblib.load(r'C:\Users\HessaM\Desktop\dataset\delay_regressor.pkl')
    clf = joblib.load(r'C:\Users\HessaM\Desktop\dataset\delay_classifier.pkl')
    encoders = joblib.load(r'C:\Users\HessaM\Desktop\dataset\label_encoders.pkl')
    return reg, clf, encoders

(daily_airport_activity, daily_dep_delays, daily_airline_airport, 
 airline_airport_agg, airport_summary, airports, date_range, airlines_list) = load_aggregated_data()

contiguous_states = ['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
airports_continental = airports[airports['state'].isin(contiguous_states)].copy()

st.sidebar.header("Global Filters")
min_date = date_range['FL_DATE'].min()
max_date = date_range['FL_DATE'].max()

start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

start_datetime = pd.Timestamp(start_date)
end_datetime = pd.Timestamp(end_date)

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

valid_routes_df = pd.read_csv(r'C:\Users\HessaM\Desktop\dataset\valid_routes.csv')
valid_origins = sorted(valid_routes_df['ORIGIN'].unique())

tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "✈️ Airline Analysis", "🏢 Airport Analysis", "🤖 Delay Predictor"])

with tab1:
    st.title("✈️ US Flight Delay Dashboard - Map View")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Date Range", f"{start_date} to {end_date}")
    col2.metric("🛫 Total Flights", f"{total_flights_period:,}")
    col3.metric("📍 Active Airports", len(airport_metrics))
    
    heatmap_enabled = st.checkbox("🔥 Heatmap Mode (show flight density instead of markers)", value=False)
    
    map_center = [39.8283, -98.5795]
    m = folium.Map(location=map_center, zoom_start=4, tiles='CartoDB positron')
    
    if heatmap_enabled:
        heat_data = airport_metrics[['latitude', 'longitude', 'total_flights']].values.tolist()
        HeatMap(heat_data, min_opacity=0.3, max_zoom=6, radius=15, blur=10).add_to(m)
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
            
            popup_html = f"""
            <b>{row['name']} ({row['AIRPORT_CODE']})</b><br>
            <hr>
            <b>📍 City:</b> {row['city']}, {row['state']}<br>
            <b>🛫 Total Flights:</b> {int(row['total_flights']):,}<br>
            <b>⏱️ Avg Departure Delay:</b> {delay_value} minutes
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=marker_radius,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row['AIRPORT_CODE']}: {delay_value} min avg delay",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=2
            ).add_to(m)
    
    st.subheader("🗺️ Airport Departure Delay Map (Continental US)")
    st_folium(m, width=900, height=500)
    
    st.subheader("📊 Airline Delay Breakdown by Airport")
    selected_airport_code = st.selectbox(
        "Select an airport to see airline delay breakdown:",
        options=sorted(airport_metrics['AIRPORT_CODE'].unique()),
        format_func=lambda x: f"{x} - {airport_metrics[airport_metrics['AIRPORT_CODE']==x]['name'].iloc[0]}"
    )
    
    if selected_airport_code:
        airport_airline_data = airline_airport_filtered_global[
            airline_airport_filtered_global['AIRPORT_CODE'] == selected_airport_code
        ].copy()
        
        if len(airport_airline_data) > 0:
            airport_airline_data = airport_airline_data.sort_values('avg_delay', ascending=False)
            fig = px.bar(
                airport_airline_data,
                x='AIRLINE',
                y='avg_delay',
                color='avg_delay',
                color_continuous_scale='RdYlGn_r',
                title=f"Average Departure Delay by Airline at {selected_airport_code} ({start_date} to {end_date})",
                labels={'avg_delay': 'Avg Delay (minutes)', 'AIRLINE': 'Airline'},
                text='flight_count',
                height=500
            )
            fig.update_traces(texttemplate='%{text} flights', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data available for {selected_airport_code} in selected date range")

with tab2:
    st.title("✈️ Airline Performance Analysis")
    st.caption(f"Date range: {start_date} to {end_date}")
    
    selected_airline = st.selectbox("Select Airline:", options=airlines_list, key="airline_select")
    
    airline_data = airline_airport_filtered_global[airline_airport_filtered_global['AIRLINE'] == selected_airline].copy()
    
    airline_data = airline_data.merge(
        airports_continental[['AIRPORT_CODE', 'name', 'city', 'state']], 
        on='AIRPORT_CODE', 
        how='inner'
    )
    
    airline_data = airline_data[airline_data['flight_count'] >= 50]
    
    if len(airline_data) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔴 Worst Airports")
            worst = airline_data.nlargest(10, 'avg_delay')[['AIRPORT_CODE', 'name', 'city', 'avg_delay', 'flight_count']]
            worst['avg_delay'] = worst['avg_delay'].round(1)
            st.dataframe(worst, use_container_width=True)
        
        with col2:
            st.subheader("🟢 Best Airports")
            best = airline_data.nsmallest(10, 'avg_delay')[['AIRPORT_CODE', 'name', 'city', 'avg_delay', 'flight_count']]
            best['avg_delay'] = best['avg_delay'].round(1)
            st.dataframe(best, use_container_width=True)
        
        fig = px.bar(
            airline_data.sort_values('avg_delay', ascending=False),
            x='AIRPORT_CODE',
            y='avg_delay',
            color='avg_delay',
            color_continuous_scale='RdYlGn_r',
            title=f"Average Delay by Airport for {selected_airline}",
            labels={'avg_delay': 'Avg Delay (minutes)', 'AIRPORT_CODE': 'Airport'},
            hover_data={'name': True, 'city': True, 'flight_count': True},
            height=500
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        total_flights_airline = airline_data['flight_count'].sum()
        avg_delay_airline = airline_data['avg_delay'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Flights", f"{total_flights_airline:,}")
        col2.metric("Average Delay (All Airports)", f"{avg_delay_airline:.1f} min")
        col3.metric("Airports Served", len(airline_data))
    else:
        st.warning(f"No data available for {selected_airline} in selected date range.")

with tab3:
    st.title("🏢 Airport Performance Analysis")
    st.caption(f"Date range: {start_date} to {end_date}")
    st.markdown("Compare airports by flight volume, delay statistics, and operational metrics.")
    
    available_airports = sorted(airport_summary_filtered['AIRPORT_CODE'].unique())
    default_airports = ['ATL', 'DFW', 'DEN', 'ORD', 'LAX']
    selected_airports = st.multiselect(
        "Select airports to compare:",
        options=available_airports,
        default=[a for a in default_airports if a in available_airports][:5]
    )
    
    compare_metric = st.selectbox(
        "Select metric to compare:",
        options=['total_flights', 'avg_delay', 'pct_late', 'pct_early', 'pct_on_time', 'max_delay', 'std_delay'],
        format_func=lambda x: {
            'total_flights': 'Total Flights',
            'avg_delay': 'Average Delay (minutes)',
            'pct_late': 'Percentage Late (>0 min)',
            'pct_early': 'Percentage Early (<0 min)',
            'pct_on_time': 'Percentage On Time (0 min)',
            'max_delay': 'Maximum Delay (minutes)',
            'std_delay': 'Delay Standard Deviation'
        }.get(x, x)
    )
    
    if len(selected_airports) > 0:
        comparison_data = airport_summary_filtered[airport_summary_filtered['AIRPORT_CODE'].isin(selected_airports)].copy()
        
        fig1 = px.bar(
            comparison_data.sort_values(compare_metric, ascending=False),
            x='AIRPORT_CODE',
            y=compare_metric,
            color=compare_metric,
            color_continuous_scale='RdYlGn_r' if compare_metric != 'avg_delay' else 'RdYlGn_r',
            title=f"Airport Comparison: {compare_metric.replace('_', ' ').title()}",
            labels={compare_metric: compare_metric.replace('_', ' ').title(), 'AIRPORT_CODE': 'Airport'},
            hover_data={'name': True, 'city': True, 'state': True},
            text=compare_metric if compare_metric != 'total_flights' else None
        )
        if compare_metric == 'total_flights':
            fig1.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("📋 Detailed Airport Statistics")
        display_cols = ['AIRPORT_CODE', 'name', 'city', 'state', 'total_flights', 'avg_delay', 
                        'pct_late', 'pct_early', 'pct_on_time', 'max_delay', 'std_delay']
        display_df = comparison_data[display_cols].copy()
        display_df['avg_delay'] = display_df['avg_delay'].round(1)
        display_df['max_delay'] = display_df['max_delay'].round(0).astype(int)
        display_df['std_delay'] = display_df['std_delay'].round(1)
        
        st.dataframe(display_df, use_container_width=True)
        
        st.subheader("📈 Flight Volume vs Average Delay")
        fig2 = px.scatter(
            comparison_data,
            x='total_flights',
            y='avg_delay',
            text='AIRPORT_CODE',
            size='total_flights',
            color='pct_late',
            color_continuous_scale='RdYlGn_r',
            title="Relationship Between Flight Volume and Average Delay",
            labels={'total_flights': 'Total Flights', 'avg_delay': 'Average Delay (minutes)', 'pct_late': 'Percent Late'},
            hover_data={'name': True, 'city': True, 'state': True},
            height=500
        )
        fig2.update_traces(textposition='top center')
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("🏆 Top Airlines at Selected Airports")
        for airport in selected_airports[:3]:
            airport_airlines = airline_airport_filtered_global[airline_airport_filtered_global['AIRPORT_CODE'] == airport].copy()
            airport_airlines = airport_airlines.nlargest(8, 'flight_count')
            
            if len(airport_airlines) > 0:
                airport_name = comparison_data[comparison_data['AIRPORT_CODE'] == airport]['name'].iloc[0] if len(comparison_data[comparison_data['AIRPORT_CODE'] == airport]) > 0 else airport
                st.markdown(f"**{airport} - {airport_name}**")
                fig4 = px.bar(
                    airport_airlines,
                    x='AIRLINE',
                    y='flight_count',
                    color='avg_delay',
                    color_continuous_scale='RdYlGn_r',
                    title=f"Top Airlines at {airport}",
                    labels={'flight_count': 'Number of Flights', 'AIRLINE': 'Airline', 'avg_delay': 'Avg Delay (min)'},
                    height=350
                )
                st.plotly_chart(fig4, use_container_width=True)
        
        st.download_button(
            label="📥 Download Airport Comparison Data (CSV)",
            data=display_df.to_csv(index=False),
            file_name=f"airport_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Please select at least one airport from the dropdown above to start the analysis.")

with tab4:
    st.title("🤖 Flight Delay Predictor")
    st.markdown("Predicts departure delay based on historical patterns for your selected route, airline, and departure time.")
    
    try:
        reg_model, clf_model, encoders = load_ml_models()
        models_loaded = True
    except Exception as e:
        st.error(f"Models not found. Please run ml_train_balanced.py first.\nError: {str(e)}")
        models_loaded = False
    
    if models_loaded:
        st.subheader("📋 Flight Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            origin = st.selectbox("Origin Airport:", options=valid_origins)
            valid_dest_for_origin = valid_routes_df[valid_routes_df['ORIGIN'] == origin]['DEST'].unique()
            valid_dest_for_origin = sorted(valid_dest_for_origin)
            destination = st.selectbox("Destination Airport:", options=valid_dest_for_origin)
            
            airline = st.selectbox("Airline:", options=sorted(encoders['AIRLINE'].classes_))
        
        with col2:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_of_week = st.selectbox(
                "Day of Week:", 
                options=list(range(7)),
                format_func=lambda x: day_names[x]
            )
            
            hour = st.slider("Departure Hour (24h format):", 0, 23, 12, help="Example: 9 = 9:00 AM, 14 = 2:00 PM")
            
            month = st.slider("Month:", 1, 12, 6, help="January = 1, December = 12")
        
        st.markdown("---")
        
        if st.button("🔮 Predict Delay", type="primary", use_container_width=True):
            try:
                origin_enc = encoders['ORIGIN'].transform([origin])[0]
                dest_enc = encoders['DEST'].transform([destination])[0]
                airline_enc = encoders['AIRLINE'].transform([airline])[0]
                
                features = np.array([[origin_enc, dest_enc, airline_enc, hour, day_of_week, month]])
                
                pred_delay = reg_model.predict(features)[0]
                pred_late_prob = clf_model.predict_proba(features)[0][1]
                
                st.subheader("📊 Prediction Results")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                
                if pred_delay < 0:
                    delay_color = "🟢"
                    delay_text = "Early"
                elif pred_delay < 10:
                    delay_color = "🟢"
                    delay_text = "On Time"
                elif pred_delay < 30:
                    delay_color = "🟡"
                    delay_text = "Minor Delay"
                elif pred_delay < 60:
                    delay_color = "🟠"
                    delay_text = "Moderate Delay"
                else:
                    delay_color = "🔴"
                    delay_text = "Severe Delay"
                
                result_col1.metric(
                    "Expected Departure Delay", 
                    f"{pred_delay:.0f} minutes",
                    delta=delay_text
                )
                
                if pred_late_prob < 0.3:
                    prob_color = "🟢 Low"
                elif pred_late_prob < 0.6:
                    prob_color = "🟡 Medium"
                else:
                    prob_color = "🔴 High"
                
                result_col2.metric(
                    "Probability of Being Late (>15 min)", 
                    f"{pred_late_prob*100:.0f}%",
                    delta=prob_color
                )
                
                avg_flight_time = None
                try:
                    flight_time_df = pd.read_csv(r'C:\Users\HessaM\Desktop\dataset\flights_dashboard_ready.csv')
                    flight_time_df = flight_time_df[
                        (flight_time_df['ORIGIN'] == origin) & 
                        (flight_time_df['DEST'] == destination)
                    ]
                    if len(flight_time_df) > 0:
                        avg_flight_time = flight_time_df['ELAPSED_TIME'].mean()
                except:
                    pass
                
                if avg_flight_time and not pd.isna(avg_flight_time):
                    total_minutes = avg_flight_time + pred_delay
                    arrival_hour = (hour + int(total_minutes // 60)) % 24
                    arrival_minute = int(total_minutes % 60)
                    
                    result_col3.metric(
                        "Estimated Flight Time", 
                        f"{avg_flight_time:.0f} minutes",
                        delta=None
                    )
                    
                    st.info(f"✈️ **Estimated arrival time:** {int(arrival_hour):02d}:{arrival_minute:02d} (based on {avg_flight_time:.0f} min average flight time + {pred_delay:.0f} min delay)")
                else:
                    result_col3.metric("Route Data", "Limited historical data", delta=None)
                
                st.markdown("---")
                st.subheader("📖 About This Prediction")
                
                season = "Winter" if month in [12,1,2] else ("Spring" if month in [3,4,5] else ("Summer" if month in [6,7,8] else "Fall"))
                
                st.markdown(f"""
                **Based on historical data for {airline} flights from {origin} to {destination}:**
                - Most significant factor: **Departure hour** (accounts for 36% of prediction weight)
                - Origin airport: 23% importance
                - Destination airport: 15% importance
                - Airline: 14% importance
                - Month ({season}): 7% importance
                - Day of week ({day_names[day_of_week]}): 5% importance
                
                > ⚠️ **Note:** This prediction is based on historical patterns only. Actual delays may vary due to weather, air traffic control, mechanical issues, or other unforeseen events.
                """)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("This route combination may not exist in the training data. Try a different origin/destination pair.")

st.markdown("---")
st.caption("Data source: US Flight Delays 2019-2023 | Continental US only | All data pre-aggregated for performance | ML model: Random Forest with class balancing (64% recall for late flights)")