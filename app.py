
import streamlit as st
import pandas as pd
import joblib
from datetime import date

st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="centered"
)

# Ladda den tränade modellen
@st.cache_resource
def load_model():
    return joblib.load("hotel_booking_model.joblib")

bundle = load_model()
model = bundle["model"]

# Hämta kategorier som modellen känner igen
preprocessor = model.named_steps["preprocessor"]
cat_columns = preprocessor.transformers_[1][2]
encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

categories = {
    col: list(values)
    for col, values in zip(cat_columns, encoder.categories_)
}

def select_category(label, feature):
    options = categories[feature]
    default = bundle["categorical_defaults"][feature]
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index)

st.title("Hotel Cancellation Predictor")
st.write(
    "Uppskatta sannolikheten för att en hotellbokning avbokas "
    "med hjälp av en tränad Random Forest-modell."
)

st.info("Demonstration baserad på hotellbokningar från Portugal 2015–2017.")

with st.form("booking_form"):

    st.subheader("Bokningsinformation")

    arrival = st.date_input(
        "Planerat ankomstdatum",
        value=date(2016, 7, 15),
        min_value=date(2015, 7, 1),
        max_value=date(2017, 8, 31)
    )

    col1, col2 = st.columns(2)

    with col1:
        hotel = select_category("Hotelltyp", "hotel")
        lead_time = st.number_input(
            "Lead time (dagar)", min_value=0, max_value=737, value=90
        )
        total_guests = st.number_input(
            "Antal gäster", min_value=1, max_value=20, value=2
        )
        previous_cancellations = st.number_input(
            "Tidigare avbokningar", min_value=0, value=0
        )

    with col2:
        country = select_category("Land", "country")
        customer_type = select_category("Kundtyp", "customer_type")
        market_segment = select_category("Marknadssegment", "market_segment")
        total_nights = st.number_input(
            "Antal nätter", min_value=1, max_value=69, value=3
        )

    submitted = st.form_submit_button(
        "Beräkna avbokningsrisk",
        use_container_width=True
    )

if submitted:

    # Utgå från standardvärden för övriga features
    booking = {
        **bundle["numeric_defaults"],
        **bundle["categorical_defaults"]
    }

    months = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    # Uppdatera värden från formuläret
    booking.update({
        "hotel": hotel,
        "lead_time": lead_time,
        "total_guests": total_guests,
        "total_nights": total_nights,
        "previous_cancellations": previous_cancellations,
        "country": country,
        "customer_type": customer_type,
        "market_segment": market_segment,
        "arrival_date_year": arrival.year,
        "arrival_date_month": months[arrival.month - 1],
        "arrival_date_week_number": arrival.isocalendar().week,
        "arrival_date_day_of_month": arrival.day
    })

    # Samma features och ordning som vid träningen
    input_data = pd.DataFrame(
        [booking],
        columns=bundle["feature_columns"]
    )

    # Modellens förutsägelse
    probability = model.predict_proba(input_data)[0, 1]
    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("Resultat")

    st.metric(
        "Uppskattad sannolikhet för avbokning",
        f"{probability:.1%}"
    )

    if prediction == 1:
        st.warning("Modellen förutsäger att bokningen avbokas.")
    else:
        st.success("Modellen förutsäger att bokningen inte avbokas.")

    st.caption(
        "Övriga features använder standardvärden från träningsdata. "
        "Resultatet är en uppskattning, inte en garanti eller "
        "en kalibrerad sannolikhet."
    )
