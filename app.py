from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

from sklearn.linear_model import LinearRegression

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

st.set_page_config(
    page_title="Sales Intel Pro",
    page_icon="📊",
    layout="wide", # This makes the app use the full screen width
    initial_sidebar_state="expanded"
)

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not run the rest of the app if password is wrong
    
# --- STEP 1: GENERATE SYNTHETIC DATA ---
@st.cache_data # This tells Streamlit to store the result so it doesn't re-run every time you click a button
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq='D')
    n = len(dates)
    
    # Create sales with a slight upward trend and weekend noise
    base_sales = np.linspace(200, 500, n) 
    weekend_boost = [50 if d.weekday() >= 5 else 0 for d in dates]
    noise = np.random.normal(0, 20, n)
    
    sales = base_sales + weekend_boost + noise

    # Create the DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales.round(2),
        'Category': np.random.choice(['Electronics', 'Home Office', 'Furniture'], n),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n)
    })
    return df

# Load the data into our app
df_raw = generate_data()
      
# --- STEP 2: DASHBOARD UI & FILTERING ---
st.title("🚀 Sales Performance Dashboard")
# --- STYLE UPGRADES ---
# Sidebar branding
st.sidebar.markdown("---")
st.sidebar.image("https://www.gstatic.com/images/branding/product/2x/analytics_64dp.png", width=50)
st.sidebar.markdown("### Admin Controls")

# Custom CSS for Metric Cards
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 40px;
        color: #007BFF;
    }
    .main {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
st.sidebar.header("Dashboard Filters")

# Sidebar Filter
selected_region = st.sidebar.multiselect(
    "Select Regions:",
    options=df_raw['Region'].unique(),
    default=df_raw['Region'].unique()
)

# Using POLARS for fast filtering (converting from pandas to polars)
# This is where you see the benefit for larger datasets
df_pl = pl.from_pandas(df_raw)
filtered_df = df_pl.filter(pl.col('Region').is_in(selected_region)).to_pandas()

# Show basic metrics at the top
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
col2.metric("Avg Daily Sale", f"${filtered_df['Sales'].mean():,.2f}")
col3.metric("Data Points", len(filtered_df))

# --- STEP 3: VISUALIZATIONS ---
st.subheader("Visual Analysis")

# Create two columns for side-by-side charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.write("### Sales over Time (Interactive)")
    # Using Plotly for the time series
    fig_line = px.line(
        filtered_df, 
        x='Date', 
        y='Sales', 
        color='Category',
        template='plotly_white'
    )
    st.plotly_chart(fig_line, width='stretch')

with chart_col2:
    st.write("### Sales Distribution (Static)")
    # Using Seaborn for statistical distribution
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig_sns, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x='Category', y='Sales', ax=ax, palette='Set2')
    plt.xticks(rotation=45)
    st.pyplot(fig_sns) 

# --- STEP 4: MACHINE LEARNING PREDICTION ---
# 
st.divider() # Adds a visual line
st.subheader("🔮 Future Sales Predictor")

# 1. Prepare Data for ML
# We convert dates to numbers so the model can understand "time"
filtered_df['Date_Ordinal'] = filtered_df['Date'].map(datetime.toordinal)

X = filtered_df[['Date_Ordinal']] # Input (Feature)
y = filtered_df['Sales']          # Output (Target)

# 2. Train a simple model
model = LinearRegression()
model.fit(X, y)

# 3. User Input for Prediction
days_ahead = st.slider("Predict sales how many days into the future?", 1, 30, 7)
future_date = datetime.now() + timedelta(days=days_ahead)
future_date_ordinal = future_date.toordinal()

# 4. Make the Prediction
prediction = model.predict([[future_date_ordinal]])

st.write(f"Based on current trends in the selected regions:")
st.success(f"Predicted Sales for **{future_date.strftime('%Y-%m-%d')}**: **${prediction[0]:,.2f}**")

# --- STEP 5: DATA EXPORT ---
st.divider()
st.subheader("📥 Export Filtered Data")

# Streamlit needs the data to be converted to a CSV format in a "buffer"
@st.cache_data
def convert_df(df_to_convert):
    # IMPORTANT: Cache this conversion so it doesn't happen on every click
    return df_to_convert.to_csv(index=False).encode('utf-8')

csv_data = convert_df(filtered_df)

st.download_button(
    label="Download filtered data as CSV",
    data=csv_data,
    file_name='filtered_sales_data.csv',
    mime='text/csv',
)

st.info(f"Clicking the button will export {len(filtered_df)} rows based on your sidebar filters.")
