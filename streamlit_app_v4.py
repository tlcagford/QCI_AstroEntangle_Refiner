# Custom CSS for LIGHT THEME
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1e3c72;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4b4b;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
    }
    /* Button styling */
    .stButton > button {
        background-color: #1e3c72;
        color: white;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #2c5282;
        color: white;
    }
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #1e3c72;
    }
    /* Metric styling */
    .stMetric {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 0.5rem;
    }
    /* Dataframe styling */
    .dataframe {
        background-color: #ffffff;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)
