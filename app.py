import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="SmartCart AI",
    page_icon="🛒",
    layout="wide"
)

# =========================
# LOAD MODELS
# =========================

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# =========================
# CLUSTER INFORMATION
# =========================

cluster_info = {

    0: {
        "name": "High-Value Family Customers",

        "description": [
            "High income and high spending customers",
            "Strong purchasing power",
            "Family-oriented buyers",
            "Frequent store purchases"
        ],

        "marketing": [
            "Offer premium family product bundles",
            "Provide personalized recommendations",
            "Send exclusive family discount offers",
            "Promote loyalty memberships"
        ],

        "retention": [
            "VIP loyalty rewards",
            "Early access to premium products",
            "Priority customer support",
            "Exclusive festive offers"
        ]
    },

    1: {
        "name": "Low-Spending Partnered Customers",

        "description": [
            "Price-sensitive customers",
            "Low spending behavior",
            "High browsing but low purchasing",
            "Mostly discount-oriented buyers"
        ],

        "marketing": [
            "Flash sales and discount campaigns",
            "Combo offers and coupons",
            "Festival sales promotions",
            "Affordable product recommendations"
        ],

        "retention": [
            "Cashback rewards",
            "Cart reminder notifications",
            "Price-drop alerts",
            "Limited-time offers"
        ]
    },

    2: {
        "name": "Occasional Independent Customers",

        "description": [
            "Independent customers living alone",
            "Moderate engagement levels",
            "Occasional shopping behavior",
            "Inconsistent purchasing patterns"
        ],

        "marketing": [
            "Lifestyle-based recommendations",
            "Trending product suggestions",
            "Personalized marketing emails",
            "Social media targeted campaigns"
        ],

        "retention": [
            "Gamified reward systems",
            "Re-engagement notifications",
            "Limited-time engagement offers",
            "Personalized reminders"
        ]
    },

    3: {
        "name": "Premium Highly Engaged Customers",

        "description": [
            "Highest spending customers",
            "Very high campaign response rate",
            "Premium purchasing behavior",
            "Highly valuable customer segment"
        ],

        "marketing": [
            "Luxury product recommendations",
            "Exclusive memberships",
            "Premium experiences and launches",
            "AI-driven personalized recommendations"
        ],

        "retention": [
            "VIP concierge support",
            "High-tier loyalty rewards",
            "Exclusive early-access launches",
            "Premium customer benefits"
        ]
    }
}

# =========================
# TITLE
# =========================

st.title("🛒 SmartCart AI")

st.subheader(
    "AI-Powered Customer Segmentation & Marketing Intelligence System"
)

st.write(
    "Predict customer segments and generate personalized "
    "marketing and retention strategies using Machine Learning."
)

st.divider()

# =========================
# INPUT SECTION
# =========================

col1, col2 = st.columns(2)

# =========================
# CUSTOMER PROFILE
# =========================

with col1:

    st.header("Customer Profile")

    income = st.number_input(
        "Income",
        min_value=0,
        max_value=200000,
        value=50000
    )

    age = st.slider(
        "Age",
        18,
        90,
        35
    )

    children = st.slider(
        "Total Children",
        0,
        5,
        1
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Postgraduate", "Undergraduate"]
    )

    living = st.selectbox(
        "Living Status",
        ["Alone", "Partner"]
    )

    tenure = st.slider(
        "Customer Tenure (Days)",
        0,
        1000,
        350
    )

# =========================
# PURCHASE BEHAVIOUR
# =========================

with col2:

    st.header("Purchase Behaviour")

    spending = st.number_input(
        "Overall Historical Spending",
        min_value=0.0,
        max_value=2000.0,
        value=500.0
    )

    num_web = st.slider(
        "Number of Web Purchases",
        0,
        20,
        5
    )

    num_store = st.slider(
        "Number of Store Purchases",
        0,
        20,
        5
    )

    num_deals = st.slider(
        "Number of Deal Purchases",
        0,
        20,
        2
    )

    web_visits = st.slider(
        "Website Visits Per Month",
        0,
        20,
        5
    )

# =========================
# HIDDEN DEFAULT VALUES
# =========================

recency = 50
num_catalog = 2
complain = 0
response = 0

# =========================
# ENCODING
# =========================

edu_grad = 1 if education == "Graduate" else 0
edu_post = 1 if education == "Postgraduate" else 0
edu_under = 1 if education == "Undergraduate" else 0

living_alone = 1 if living == "Alone" else 0
living_partner = 1 if living == "Partner" else 0

st.divider()

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict Customer Segment"):

    input_data = pd.DataFrame({
        "Income": [income],
        "Recency": [recency],
        "NumDealsPurchases": [num_deals],
        "NumWebPurchases": [num_web],
        "NumCatalogPurchases": [num_catalog],
        "NumStorePurchases": [num_store],
        "NumWebVisitsMonth": [web_visits],
        "Complain": [complain],
        "Response": [response],
        "Age": [age],
        "customer_tenure_days": [tenure],
        "Total_Spending": [spending],
        "Total_Children": [children],
        "Education_Graduate": [edu_grad],
        "Education_Postgraduate": [edu_post],
        "Education_undergraduate": [edu_under],
        "Living_with_Alone": [living_alone],
        "Living_with_Partner": [living_partner]
    })

    # Scale data
    scaled_data = scaler.transform(input_data)

    # PCA transform
    pca_data = pca.transform(scaled_data)

    # Predict cluster
    prediction = kmeans.predict(pca_data)[0]

    # Get cluster info
    info = cluster_info[prediction]

    st.divider()

    # =========================
    # OUTPUT
    # =========================

    st.success(f"Predicted Segment: {info['name']}")

    output_col1, output_col2 = st.columns(2)

    with output_col1:

        st.subheader("📌 Customer Characteristics")

        for item in info["description"]:
            st.write(f"• {item}")

    with output_col2:

        st.subheader("📢 Personalized Marketing Strategy")

        for item in info["marketing"]:
            st.write(f"• {item}")

    st.subheader("🔒 Customer Retention Strategy")

    for item in info["retention"]:
        st.write(f"• {item}")

    st.divider()

    st.info(
        "Prediction generated using KMeans clustering, "
        "feature scaling, PCA dimensionality reduction, "
        "and customer behavior analytics."
    )