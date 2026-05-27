import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1F2937;
}

.metric-card {
    background: linear-gradient(
        145deg,
        #111827,
        #1F2937
    );

    padding: 1.5rem;

    border-radius: 20px;

    border: 1px solid #374151;

    box-shadow:
        0px 0px 15px rgba(0,0,0,0.35);

    transition: all 0.3s ease-in-out;
}

.metric-card:hover {
    transform: translateY(-4px);
    border: 1px solid #4B5563;
}

.metric-title {
    color: #9CA3AF;
    font-size: 15px;
    margin-bottom: 10px;
}

.metric-value {
    color: white;
    font-size: 34px;
    font-weight: 700;
}

h1, h2, h3 {
    color: white;
}

.stMarkdown {
    color: #D1D5DB;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df


@st.cache_resource
def load_model():

    model = joblib.load(
        "models/xgb_model.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    model_columns = joblib.load(
        "models/model_columns.pkl"
    )

    return model, scaler, model_columns


df = load_data()

model, scaler, model_columns = load_model()

st.title("Customer Churn Intelligence")

st.markdown("""
Modern AI-powered analytics dashboard for
customer retention and churn prediction.
""")

st.divider()

st.sidebar.title("Filters")

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

internet_filter = st.sidebar.multiselect(
    "Internet Service",
    options=df["InternetService"].unique(),
    default=df["InternetService"].unique()
)

payment_filter = st.sidebar.multiselect(
    "Payment Method",
    options=df["PaymentMethod"].unique(),
    default=df["PaymentMethod"].unique()
)

filtered_df = df[
    (df["Contract"].isin(contract_filter))
    &
    (df["InternetService"].isin(internet_filter))
    &
    (df["PaymentMethod"].isin(payment_filter))
]

total_customers = len(filtered_df)

churn_rate = (
    filtered_df["Churn"]
    .value_counts(normalize=True)
    .get("Yes", 0) * 100
)

monthly_revenue = (
    filtered_df["MonthlyCharges"]
    .sum()
)

avg_tenure = (
    filtered_df["tenure"]
    .mean()
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">
            Total Customers
        </div>

        <div class="metric-value">
            {total_customers:,}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">
            Churn Rate
        </div>

        <div class="metric-value">
            {churn_rate:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">
            Monthly Revenue
        </div>

        <div class="metric-value">
            ${monthly_revenue:,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">
            Avg Tenure
        </div>

        <div class="metric-value">
            {avg_tenure:.1f}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

chart1, chart2 = st.columns(2)

with chart1:

    churn_contract = pd.crosstab(
        filtered_df["Contract"],
        filtered_df["Churn"],
        normalize="index"
    )

    fig_contract = px.bar(
        churn_contract,
        barmode="stack",
        title="Churn by Contract Type",
        color_discrete_sequence=[
            "#374151",
            "#EF4444"
        ]
    )

    fig_contract.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117"
    )

    st.plotly_chart(
        fig_contract,
        use_container_width=True
    )

with chart2:

    fig_tenure = px.histogram(
        filtered_df,
        x="tenure",
        nbins=30,
        title="Customer Tenure Distribution",
        color_discrete_sequence=["#3B82F6"]
    )

    fig_tenure.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117"
    )

    st.plotly_chart(
        fig_tenure,
        use_container_width=True
    )


chart3, chart4 = st.columns(2)

with chart3:

    fig_monthly = px.box(
        filtered_df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges vs Churn",
        color_discrete_map={
            "Yes": "#EF4444",
            "No": "#10B981"
        }
    )

    fig_monthly.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117"
    )

    st.plotly_chart(
        fig_monthly,
        use_container_width=True
    )

with chart4:

    churn_counts = (
        filtered_df["Churn"]
        .value_counts()
    )

    fig_pie = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        title="Churn Distribution",
        color_discrete_sequence=[
            "#10B981",
            "#EF4444"
        ]
    )

    fig_pie.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

st.divider()

st.header("AI Churn Prediction Engine")

left, right = st.columns([1,1])


with left:

    tenure_input = st.slider(
        "Customer Tenure",
        0,
        72,
        12
    )

    monthly_charges_input = st.slider(
        "Monthly Charges",
        0,
        150,
        70
    )

with right:

    contract_input = st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    internet_input = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )


input_data = pd.DataFrame({
    "tenure": [tenure_input],
    "MonthlyCharges": [monthly_charges_input],
})


input_data["Contract_One year"] = 0
input_data["Contract_Two year"] = 0

if contract_input == "One year":
    input_data["Contract_One year"] = 1

elif contract_input == "Two year":
    input_data["Contract_Two year"] = 1

input_data["InternetService_Fiber optic"] = 0
input_data["InternetService_No"] = 0

if internet_input == "Fiber optic":
    input_data["InternetService_Fiber optic"] = 1

elif internet_input == "No":
    input_data["InternetService_No"] = 1


missing_cols = (
    set(model_columns)
    - set(input_data.columns)
)

for col in missing_cols:
    input_data[col] = 0

input_data = input_data[model_columns]


input_scaled = scaler.transform(
    input_data
)

prediction_prob = model.predict_proba(
    input_scaled
)[0][1]

risk_percentage = prediction_prob * 100


gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_percentage,

    title={
        "text": "Churn Risk Score"
    },

    gauge={

        "axis": {
            "range": [0,100]
        },

        "bar": {
            "color": "#EF4444"
        },

        "steps": [

            {
                "range": [0,40],
                "color": "#10B981"
            },

            {
                "range": [40,70],
                "color": "#F59E0B"
            },

            {
                "range": [70,100],
                "color": "#EF4444"
            }
        ]
    }
))

gauge.update_layout(
    template="plotly_dark",
    height=400,
    paper_bgcolor="#0E1117"
)

st.plotly_chart(
    gauge,
    use_container_width=True
)



if prediction_prob >= 0.70:

    st.error(
        f"⚠️ High Churn Risk — {risk_percentage:.1f}%"
    )

elif prediction_prob >= 0.40:

    st.warning(
        f"⚠️ Medium Churn Risk — {risk_percentage:.1f}%"
    )

else:

    st.success(
        f"✅ Low Churn Risk — {risk_percentage:.1f}%"
    )



st.subheader("AI Retention Recommendations")

recommendations = []

if contract_input == "Month-to-month":

    recommendations.append(
        "Offer incentives for long-term contracts."
    )

if tenure_input < 12:

    recommendations.append(
        "New customers require onboarding engagement."
    )

if monthly_charges_input > 90:

    recommendations.append(
        "Evaluate pricing strategy for this customer."
    )

if internet_input == "Fiber optic":

    recommendations.append(
        "Fiber optic users may require improved support."
    )

for rec in recommendations:

    st.markdown(f"""
    - {rec}
    """)

st.divider()



st.subheader("Customer Dataset")

st.dataframe(
    filtered_df.drop(
        columns=["customerID"]
    ),

    use_container_width=True
)



st.markdown("""
<hr style="border:1px solid #1F2937">
""", unsafe_allow_html=True)

st.caption("""
Developed by Gabriel Soares
| Data Science & Analytics Project
""")