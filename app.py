"""
Sleep Health Predictor — Streamlit Web Application
====================================================
Predicts sleep disorders (Sleep Apnea / Insomnia / Healthy) from
lifestyle and health attributes using a pre-trained XGBoost pipeline.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sleep Health Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = "sleep_disorder_model.pkl"

# Label map that was used during training:  Sleep Apnea=0, Insomnia=1, Healthy=2
LABEL_MAP = {0: "Sleep Apnea", 1: "Insomnia", 2: "Healthy"}

RESULT_CONFIG = {
    "Healthy": {
        "color": "#2e7d32",
        "bg": "#e8f5e9",
        "border": "#a5d6a7",
        "message": (
            "Great news! Your lifestyle indicators suggest you are **not "
            "currently affected by a sleep disorder**. Keep maintaining "
            "healthy sleep habits, regular exercise, and balanced stress levels."
        ),
    },
    "Insomnia": {
        "color": "#e65100",
        "bg": "#fff3e0",
        "border": "#ffcc80",
        "message": (
            "Your profile shows markers associated with **Insomnia** — "
            "difficulty falling or staying asleep. Consider consulting a "
            "healthcare professional and reviewing your sleep hygiene, "
            "stress management, and daily routine."
        ),
    },
    "Sleep Apnea": {
        "color": "#6a1b9a",
        "bg": "#f3e5f5",
        "border": "#ce93d8",
        "message": (
            "Your profile shows markers associated with **Sleep Apnea** — "
            "a condition where breathing repeatedly stops during sleep. "
            "Please consult a healthcare professional for a proper diagnosis "
            "and potential treatment options."
        ),
    },
}

OCCUPATIONS = [
    "Accountant",
    "Doctor",
    "Engineer",
    "Lawyer",
    "Manager",
    "Nurse",
    "Sales Representative",
    "Salesperson",
    "Scientist",
    "Software Engineer",
    "Teacher",
]

BMI_CATEGORIES = ["Normal", "Overweight", "Obese"]


# ── Model loader (cached so it is only read once per session) ────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load the pre-trained scikit-learn / XGBoost pipeline from disk."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(
            f" Model file `{MODEL_PATH}` not found.  "
            "Make sure it is in the same directory as `app.py`."
        )
        st.stop()


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/sleeping-in-bed.png",
            width=80,
        )
        st.title("Sleep Health Predictor")
        st.markdown("---")

        st.markdown(
            """
            ### About
            This application uses a **machine-learning model** trained on
            lifestyle and health data to predict the likelihood of a sleep
            disorder.

            ### How to use
            1. Fill in your personal details in the main panel.
            2. Click **Predict Sleep Health**.
            3. Review your personalised result and confidence scores.

            ### Prediction Classes
            | Class | Description |
            |-------|-------------|
            | Healthy | No sleep disorder detected |
            | Insomnia | Difficulty falling/staying asleep |
            | Sleep Apnea | Repeated breathing interruptions |

            ### Disclaimer
            > This tool is for **informational purposes only** and does not
            > constitute medical advice. Always consult a qualified healthcare
            > professional for diagnosis and treatment.
            """
        )

        st.markdown("---")
        st.markdown(
            """
            ### Model Details
            - **Algorithm:** XGBoost Classifier
            - **Tuning:** 5-fold GridSearchCV
            - **Metric:** Weighted F1-Score
            - **Dataset:** Sleep Health & Lifestyle (374 records)
            """
        )

        st.markdown("---")
        st.caption("Built with [Streamlit](https://streamlit.io)")


# ── Input form ───────────────────────────────────────────────────────────────
def render_input_form():
    """Render the user-input form and return a dict of raw values."""
    st.header("Enter Your Health & Lifestyle Details")
    st.markdown(
        "Complete all fields below for the most accurate prediction. "
        "Hover over the **ℹ** icons for field-level guidance."
    )

    col1, col2, col3 = st.columns(3)

    # ── Column 1 : Demographics ──────────────────────────────────────────────
    with col1:
        st.subheader("Demographics")

        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Select your biological sex.",
        )

        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
            help="Your current age in years. The training data covers ages 27–59.",
        )

        occupation = st.selectbox(
            "Occupation",
            options=OCCUPATIONS,
            help="Select the occupation that best matches yours.",
        )

        bmi_category = st.selectbox(
            "BMI Category",
            options=BMI_CATEGORIES,
            help=(
                "Normal: BMI 18.5–24.9 | "
                "Overweight: BMI 25–29.9 | "
                "Obese: BMI ≥ 30"
            ),
        )

    # ── Column 2 : Sleep & Activity ──────────────────────────────────────────
    with col2:
        st.subheader("Sleep & Activity")

        sleep_duration = st.slider(
            "Sleep Duration (hours/night)",
            min_value=4.0,
            max_value=12.0,
            value=7.0,
            step=0.1,
            format="%.1f h",
            help="Average number of hours you sleep per night.",
        )

        quality_of_sleep = st.slider(
            "Quality of Sleep (1–10)",
            min_value=1,
            max_value=10,
            value=7,
            help="Self-rated sleep quality. 1 = very poor, 10 = excellent.",
        )

        physical_activity_level = st.slider(
            "Physical Activity Level (min/day)",
            min_value=0,
            max_value=120,
            value=45,
            step=5,
            help="Average minutes of moderate-to-vigorous physical activity per day.",
        )

        daily_steps = st.number_input(
            "Daily Steps",
            min_value=0,
            max_value=30000,
            value=7000,
            step=500,
            help="Average number of steps walked per day.",
        )

    # ── Column 3 : Vitals ────────────────────────────────────────────────────
    with col3:
        st.subheader("Health Vitals")

        stress_level = st.slider(
            "Stress Level (1–10)",
            min_value=1,
            max_value=10,
            value=5,
            help="Self-rated daily stress level. 1 = very low, 10 = extremely high.",
        )

        heart_rate = st.number_input(
            "Resting Heart Rate (bpm)",
            min_value=40,
            max_value=150,
            value=72,
            step=1,
            help="Your resting heart rate in beats per minute.",
        )

        systolic = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=80,
            max_value=200,
            value=120,
            step=1,
            help="The upper number in a blood-pressure reading (e.g. **120**/80).",
        )

        dystolic = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=40,
            max_value=130,
            value=80,
            step=1,
            help="The lower number in a blood-pressure reading (e.g. 120/**80**).",
        )

    return {
        "gender": gender,
        "age": age,
        "occupation": occupation,
        "sleep_duration": sleep_duration,
        "quality_of_sleep": quality_of_sleep,
        "physical_activity_level": physical_activity_level,
        "stress_level": stress_level,
        "BMI_category": bmi_category,
        "heart_rate": heart_rate,
        "daily_steps": daily_steps,
        "systolic": systolic,
        "dystolic": dystolic,
    }


# ── Input validation ──────────────────────────────────────────────────────────
def validate_inputs(inputs: dict) -> list[str]:
    """Return a list of validation error messages (empty = all good)."""
    errors = []

    if inputs["systolic"] <= inputs["dystolic"]:
        errors.append(
            "Systolic blood pressure must be **greater** than diastolic blood pressure."
        )

    if inputs["sleep_duration"] < 1:
        errors.append("Sleep duration must be at least 1 hour.")

    if inputs["heart_rate"] < 40:
        errors.append("Resting heart rate seems too low. Please verify the value.")

    return errors


# ── Prediction result display ─────────────────────────────────────────────────
def render_result(prediction_label: str, probabilities: np.ndarray):
    """Display the prediction result and probability breakdown."""
    cfg = RESULT_CONFIG[prediction_label]

    # ── Main result card ─────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background-color: {cfg['bg']};
            border: 2px solid {cfg['border']};
            border-radius: 12px;
            padding: 24px 28px;
            margin-top: 16px;
        ">
            <h2 style="color: {cfg['color']}; margin: 0 0 8px 0;">
                {cfg['emoji']} Prediction: {prediction_label}
            </h2>
            <p style="color: #333; font-size: 1rem; margin: 0;">
                {cfg['message']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Confidence breakdown ─────────────────────────────────────────────────
    st.subheader("Prediction Confidence")
    st.caption(
        "The bars below show how confident the model is for each class. "
        "Higher is more likely."
    )

    class_order = ["Healthy", "Insomnia", "Sleep Apnea"]
    # probabilities order from model: 0=Sleep Apnea, 1=Insomnia, 2=Healthy
    prob_dict = {
        "Sleep Apnea": float(probabilities[0]),
        "Insomnia": float(probabilities[1]),
        "Healthy": float(probabilities[2]),
    }

    for label in class_order:
        prob = prob_dict[label]
        bar_color = RESULT_CONFIG[label]["color"]
        st.markdown(
            f"""
            <div style="margin-bottom: 12px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="font-weight:600;">{RESULT_CONFIG[label]['emoji']} {label}</span>
                    <span style="font-weight:700; color:{bar_color};">{prob * 100:.1f}%</span>
                </div>
                <div style="background:#e0e0e0; border-radius:8px; height:18px; overflow:hidden;">
                    <div style="
                        width:{prob * 100:.1f}%;
                        background:{bar_color};
                        height:100%;
                        border-radius:8px;
                        transition: width 0.4s ease;
                    "></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Input summary ─────────────────────────────────────────────────────────
    with st.expander("View your submitted inputs"):
        st.markdown("These are the values used for prediction:")
        # reconstruct from session state
        inputs = st.session_state.get("last_inputs", {})
        if inputs:
            summary_df = pd.DataFrame(
                [
                    {"Feature": k.replace("_", " ").title(), "Value": str(v)}
                    for k, v in inputs.items()
                ]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ── Main application ──────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Hero banner ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1a237e 0%, #4527a0 100%);
            border-radius: 14px;
            padding: 32px 36px;
            margin-bottom: 28px;
        ">
            <h1 style="color: white; margin: 0 0 8px 0; font-size: 2.2rem;">
                Sleep Health Predictor
            </h1>
            <p style="color: #c5cae9; font-size: 1.05rem; margin: 0;">
                Enter your lifestyle and health details to receive an AI-powered
                assessment of your sleep health — powered by an XGBoost classifier
                trained on real-world data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()

    inputs = render_input_form()

    st.markdown("---")

    # ── Predict button ────────────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button(
            "Predict Sleep Health",
            type="primary",
            use_container_width=True,
        )

    if predict_clicked:
        errors = validate_inputs(inputs)

        if errors:
            for err in errors:
                st.error(f"{err}")
        else:
            # Build a single-row DataFrame with explicit dtypes so that
            # sklearn's ColumnTransformer always receives plain numpy-backed
            # columns (object for strings, float64/int64 for numerics).
            # Relying on pd.DataFrame([dict]) on newer pandas + PyArrow
            # backends can produce StringDtype columns that confuse the
            # np.issubdtype check inside the pickled ColumnTransformer.
            input_df = pd.DataFrame(
    {
        "gender": [inputs["gender"]],
        "age": [inputs["age"]],
        "occupation": [inputs["occupation"]],
        "sleep_duration": [inputs["sleep_duration"]],
        "quality_of_sleep": [inputs["quality_of_sleep"]],
        "physical_activity_level": [inputs["physical_activity_level"]],
        "stress_level": [inputs["stress_level"]],
        "BMI_category": [inputs["BMI_category"]],
        "heart_rate": [inputs["heart_rate"]],
        "daily_steps": [inputs["daily_steps"]],
        "systolic": [inputs["systolic"]],
        "dystolic": [inputs["dystolic"]],
    }
)

            with st.spinner("Analysing your data…"):
                prediction_idx = int(model.predict(input_df)[0])
                probabilities = model.predict_proba(input_df)[0]

            prediction_label = LABEL_MAP[prediction_idx]

            # Store for the expander
            st.session_state["last_inputs"] = inputs

            render_result(prediction_label, probabilities)


if __name__ == "__main__":
    main()
