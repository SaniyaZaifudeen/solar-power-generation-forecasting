
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Solar Power Analytics",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. PROJECT PATHS
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "solar_power_model.pkl"
DATA_PATH = BASE_DIR / "dataset" / "solar_plant1_processed.csv"

FEATURES = [
    "HOUR",
    "MINUTE",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION"
]

# ==================================================
# 3. DARK SOLAR THEME
# ==================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #07111F 0%,
            #0B1928 55%,
            #10251F 100%
        );
        color: #F1F5F9;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #091522;
        border-right: 1px solid #203346;
    }

    section[data-testid="stSidebar"] * {
        color: #E2E8F0;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(
            120deg,
            #102B37,
            #124A3A,
            #1B352D
        );
        border: 1px solid #28634F;
        border-radius: 20px;
        padding: 30px 32px;
        margin-bottom: 26px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.18);
    }

    .hero-label {
        color: #FACC15;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #FFFFFF;
        font-size: 34px;
        font-weight: 800;
        margin: 10px 0 8px 0;
        line-height: 1.25;
    }

    .hero-description {
        color: #C6DAD3;
        font-size: 15px;
        margin-bottom: 0;
    }

    /* Section headings */
    h1, h2, h3 {
        color: #F8FAFC !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            #122437,
            #102B2B
        );
        border: 1px solid #284354;
        padding: 20px;
        border-radius: 16px;
        min-height: 120px;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.12);
    }

    div[data-testid="stMetricLabel"] {
        color: #B6C8D8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #FACC15 !important;
        font-weight: 750;
    }

    /* General containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #263D4E;
        border-radius: 16px;
    }

    /* Inputs */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background-color: #102235 !important;
        color: #F8FAFC !important;
        border-color: #365064 !important;
        border-radius: 9px;
    }

    label {
        color: #D5E1EA !important;
    }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(
            90deg,
            #15803D,
            #16A34A
        );
        color: #FFFFFF;
        border: 1px solid #4ADE80;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 700;
        min-height: 44px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #22C55E;
        border-color: #86EFAC;
        color: #07111F;
        transform: translateY(-1px);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #CBD5E1;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FACC15;
    }

    /* Data tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #294052;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Informational text */
    .muted {
        color: #9FB2C4;
        font-size: 13px;
    }

    .status-pill {
        display: inline-block;
        color: #86EFAC;
        background: #123729;
        border: 1px solid #286747;
        border-radius: 30px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 700;
    }

    hr {
        border-color: #263D4E;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 4. LOAD MODEL AND DATASET
# ==================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.info(
        "Please check that models/solar_power_model.pkl exists."
    )
    st.stop()

# ==================================================
# 5. SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## ☀️ Solar Analytics")
    st.markdown(
        '<span class="status-pill">ML POWERED</span>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Power Prediction",
            "Model Analysis",
            "About Project"
        ]
    )

    st.markdown("---")
    st.markdown("### ⚡ Project Details")
    st.caption("Machine Learning")
    st.caption("Random Forest Regression")
    st.caption("Solar Plant 1 Dataset")

    st.markdown("---")
    st.markdown(
        '<p class="muted">Solar Power Forecasting</p>',
        unsafe_allow_html=True
    )

# ==================================================
# 6. HERO HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <div class="hero-label">RENEWABLE ENERGY INTELLIGENCE</div>
    <div class="hero-title">
        ☀️ Solar Power Forecasting
    </div>
    <p class="hero-description">
        Predict solar power output using weather conditions,
        environmental measurements, and machine learning.
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# 7. OVERVIEW PAGE
# ==================================================
if page == "Overview":

    st.subheader("⚡ Performance at a Glance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Selected Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "R² Score",
            "0.986",
            help="Coefficient of determination on the original test split."
        )

    with col3:
        st.metric(
            "Mean Absolute Error",
            "16.37",
            help="Average absolute prediction error in the dataset's AC power units."
        )

    st.markdown("---")

    st.subheader("🌿 How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌡️")
        st.markdown("**01 · Environmental Inputs**")
        st.write(
            "Enter temperature, irradiation, and the time of day."
        )

    with col2:
        st.markdown("### 🧠")
        st.markdown("**02 · Machine Learning**")
        st.write(
            "The trained Random Forest model processes the five input features."
        )

    with col3:
        st.markdown("### ⚡")
        st.markdown("**03 · Power Estimate**")
        st.write(
            "View the estimated AC power output for your selected conditions."
        )

    st.markdown("---")

    st.subheader("📈 Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "R² Score": [0.980, 0.986],
        "MAE": [26.30, 16.37],
        "RMSE": [55.51, 45.67]
    })

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These are the previously recorded test-split results, "
        "not a newly calculated evaluation."
    )

    st.info(
        "Choose **Power Prediction** in the sidebar to try the model."
    )

# ==================================================
# 8. POWER PREDICTION PAGE
# ==================================================
elif page == "Power Prediction":

    st.subheader("🔮 Predict Solar Power")
    st.write(
        "Adjust the five input values and generate a power estimate."
    )

    with st.container(border=True):

        st.markdown("### 🌤️ Environmental Conditions")

        with st.form("solar_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:
                hour = st.number_input(
                    "Hour",
                    min_value=0,
                    max_value=23,
                    value=12,
                    step=1,
                    help="Hour of the day, from 0 to 23."
                )

                ambient_temperature = st.number_input(
                    "Ambient Temperature (°C)",
                    value=30.0,
                    step=1.0
                )

                irradiation = st.number_input(
                    "Irradiation",
                    min_value=0.0,
                    value=0.8,
                    step=0.1,
                    help="Use the same irradiation units and range as the training dataset."
                )

            with col2:
                minute = st.number_input(
                    "Minute",
                    min_value=0,
                    max_value=59,
                    value=0,
                    step=1
                )

                module_temperature = st.number_input(
                    "Module Temperature (°C)",
                    value=40.0,
                    step=1.0
                )

            submitted = st.form_submit_button(
                "⚡ Generate Prediction",
                use_container_width=True
            )

    if submitted:

        input_data = pd.DataFrame({
            "HOUR": [hour],
            "MINUTE": [minute],
            "AMBIENT_TEMPERATURE": [ambient_temperature],
            "MODULE_TEMPERATURE": [module_temperature],
            "IRRADIATION": [irradiation]
        })

        try:
            prediction = float(model.predict(input_data)[0])

            st.session_state["latest_prediction"] = prediction

            if "prediction_history" not in st.session_state:
                st.session_state["prediction_history"] = []

            st.session_state["prediction_history"].append({
                "Hour": int(hour),
                "Minute": int(minute),
                "Ambient Temperature (°C)": ambient_temperature,
                "Module Temperature (°C)": module_temperature,
                "Irradiation": irradiation,
                "Predicted AC Power": round(prediction, 2)
            })

            st.success("Prediction generated successfully!")

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    if "latest_prediction" in st.session_state:

        st.markdown("---")
        st.subheader("⚡ Prediction Result")

        result_col1, result_col2 = st.columns([2, 1])

        with result_col1:
            st.metric(
                "Predicted AC Power",
                f"{st.session_state['latest_prediction']:,.2f}"
            )

        with result_col2:
            st.markdown("### 🌱")
            st.markdown("**Model:** Random Forest")
            st.caption("Regression prediction")

        st.caption(
            "The prediction is an estimate, not a guarantee of actual output. "
            "The displayed unit follows the AC_POWER target used in training."
        )

    if st.session_state.get("prediction_history"):

        st.markdown("---")
        st.subheader("📋 Recent Predictions")

        history_df = pd.DataFrame(
            st.session_state["prediction_history"]
        )

        st.dataframe(
            history_df.tail(10),
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction History",
            data=csv_data,
            file_name="solar_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):
            st.session_state["prediction_history"] = []
            st.session_state.pop("latest_prediction", None)
            st.rerun()

# ==================================================
# 9. MODEL ANALYSIS PAGE
# ==================================================
elif page == "Model Analysis":

    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### Linear Regression")
            st.metric("R² Score", "0.980")
            st.metric("MAE", "26.30")
            st.metric("RMSE", "55.51")

    with col2:
        with st.container(border=True):
            st.markdown("### 🌲 Random Forest")
            st.metric("R² Score", "0.986")
            st.metric("MAE", "16.37")
            st.metric("RMSE", "45.67")

    st.success(
        "Random Forest had the better recorded results "
        "and is the selected model."
    )

    st.markdown("---")

    st.subheader("🌞 Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        st.bar_chart(
            importance_df.set_index("Feature"),
            horizontal=True
        )

        st.caption(
            "Feature importance indicates how the fitted Random Forest "
            "uses features. It does not establish causation."
        )

    else:
        st.warning(
            "This model does not provide feature_importances_."
        )

    st.markdown("---")

    st.subheader("🎯 Actual vs Predicted AC Power")

    try:
        df = load_dataset()

        required_columns = FEATURES + ["AC_POWER"]
        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            st.error(
                f"Missing dataset columns: {missing_columns}"
            )
        else:
            X = df[FEATURES]
            y = df["AC_POWER"]

            _, X_test, _, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            y_pred = model.predict(X_test)

            chart_data = pd.DataFrame({
                "Actual": y_test.to_numpy(),
                "Predicted": y_pred
            })

            # Preserve the original actual-versus-predicted line chart.
            st.line_chart(
                chart_data.head(300),
                height=350
            )

            st.caption(
                "Showing the first 300 rows of the original random test split. "
                "This line chart compares values by row order; it is not a "
                "time-series forecast. A scatter plot is another useful "
                "way to evaluate actual versus predicted values."
            )

            with st.expander("View sample comparison values"):
                st.dataframe(
                    chart_data.head(20),
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as error:
        st.error(
            f"Could not load the dataset or build the chart: {error}"
        )

# ==================================================
# 10. ABOUT PROJECT PAGE
# ==================================================
elif page == "About Project":

    st.subheader("🌍 About This Project")

    with st.container(border=True):

        st.markdown("### Solar Power Generation Forecasting Using Machine Learning")

        st.write(
            "This project estimates solar power output from time and "
            "weather-related measurements using supervised machine learning."
        )

        st.markdown("#### Input Features")

        st.markdown("""
        - **HOUR:** Hour of the day
        - **MINUTE:** Minute of the hour
        - **AMBIENT_TEMPERATURE:** Surrounding air temperature
        - **MODULE_TEMPERATURE:** Solar module temperature
        - **IRRADIATION:** Solar irradiation measurement
        """)

        st.markdown("#### Target Variable")
        st.write("AC_POWER — the power output value predicted by the model.")

        st.markdown("#### Technologies Used")

        st.markdown("""
        - Python
        - Pandas
        - Scikit-learn
        - Joblib
        - Streamlit
        """)

    st.subheader("⚠️ Limitations")

    st.write("""
    - The dataset covers a limited period, so it cannot by itself establish
      long-term seasonal forecasting performance.
    - The existing performance figures come from a random train-test split.
      A chronological split would be a stronger evaluation for future-time
      forecasting.
    - Predictions depend on the training data and the input conditions.
    """)

# ==================================================
# 11. FOOTER
# ==================================================
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:10px 0 0 0;">
        <p style="color:#FACC15; font-weight:700; margin-bottom:5px;">
            ☀️ SOLAR POWER ANALYTICS
        </p>
        <p style="color:#94A3B8; font-size:12px;">
            Machine Learning · Renewable Energy · Power Forecasting
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
