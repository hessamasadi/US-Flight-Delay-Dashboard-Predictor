# US Flight Delay Dashboard & Predictor

Interactive dashboard and ML prediction tool for US domestic flights (2019–2023). Built with Python, Streamlit, and Scikit-learn.

## Dataset

- Source: 3M flight records from Kaggle
- Cleaned to 2.87M rows after removing outliers, cancellations, and invalid routes
- 18 airlines, 340 airports (continental US)

## Features

### 1. Map View
- Interactive Folium map with airport markers
- Color-coded by average departure delay (green → blue → orange → red)
- Marker size scales with flight volume
- Heatmap mode shows flight density
- Click any airport → bar chart of delay by airline

### 2. Airline Analysis
- Select any airline
- View worst and best airports (by average delay)
- Bar chart of delay across all airports
- Summary metrics (total flights, average delay, airports served)

### 3. Airport Analysis
- Multi-select up to 10 airports
- Compare metrics: flight volume, average delay, % late, % early, % on time, max delay, standard deviation
- Scatter plot: flight volume vs average delay
- Top 8 airlines at each selected airport
- Export comparison data to CSV

### 4. Delay Predictor
- Input: origin, destination, airline, day of week, departure hour, month
- Output: expected departure delay (minutes) + probability of being late (>15 min)
- Estimated arrival time based on historical flight duration
- Feature importance display (hour = 36%, origin = 23%, dest = 15%, airline = 14%)

## ML Model Performance

| Metric | Value |
|--------|-------|
| Regression MAE | 14.0 minutes |
| Classification Accuracy | 67.3% |
| Recall (Late flights) | 64.4% |
| Features | ORIGIN, DEST, AIRLINE, HOUR, DAY_OF_WEEK, MONTH |
| Algorithm | Random Forest with class balancing |

## Tech Stack

- **Dashboard:** Streamlit
- **Visualization:** Folium (maps), Plotly (charts)
- **ML:** Scikit-learn (Random Forest Regressor + Classifier)
- **Data:** Pandas, NumPy

## File Structure
