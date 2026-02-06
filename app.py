import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("Student Performance Predictor")
st.write("AI-based system to predict PASS/FAIL outcome with confidence.")

model = joblib.load("models/best_model.pkl")
results_df = joblib.load("models/model_results.pkl")

# ---------------- MODEL RESULTS ----------------
st.subheader("Model Comparison")
st.dataframe(results_df, use_container_width=True)

# ---------------- FEATURE IMPORTANCE ----------------
if hasattr(model, "feature_importances_"):
    st.subheader("Key Factors Affecting Performance")

    feature_names = [
        'StudyHours', 'AttendancePercent', 'PreviousScore', 'SleepHours',
        'AssignmentsCompleted', 'InternetAccess', 'ParentalEducationLevel',
        'FamilyIncomeLevel', 'ExtracurricularParticipation', 'ScreenTimeHours',
        'MotivationLevel', 'TeacherSupportRating', 'PeerStudyHours',
        'HealthRating'
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))

# ---------------- INPUT SECTION ----------------
st.sidebar.header("Student Information")

StudyHours = st.sidebar.slider("Study Hours", 0.0, 12.0, 5.0)
AttendancePercent = st.sidebar.slider("Attendance %", 0.0, 100.0, 75.0)
PreviousScore = st.sidebar.slider("Previous Score", 0.0, 100.0, 60.0)
SleepHours = st.sidebar.slider("Sleep Hours", 0.0, 10.0, 6.0)
AssignmentsCompleted = st.sidebar.slider("Assignments Completed", 0, 10, 3)

internet_option = st.sidebar.selectbox("Internet Access", ["No", "Yes"])
InternetAccess = 1 if internet_option == "Yes" else 0

ParentalEducationLevel = st.sidebar.slider("Parental Education Level", 1, 5, 3)
FamilyIncomeLevel = st.sidebar.slider("Family Income Level", 1, 5, 3)

extra_option = st.sidebar.selectbox("Extracurricular Participation", ["No", "Yes"])
ExtracurricularParticipation = 1 if extra_option == "Yes" else 0

ScreenTimeHours = st.sidebar.slider("Screen Time Hours", 0.0, 10.0, 4.0)
MotivationLevel = st.sidebar.slider("Motivation Level", 1, 5, 3)
TeacherSupportRating = st.sidebar.slider("Teacher Support Rating", 1, 5, 3)
PeerStudyHours = st.sidebar.slider("Peer Study Hours", 0.0, 6.0, 2.0)
HealthRating = st.sidebar.slider("Health Rating", 1, 5, 3)

input_data = pd.DataFrame([[ 
    StudyHours, AttendancePercent, PreviousScore, SleepHours,
    AssignmentsCompleted, InternetAccess, ParentalEducationLevel,
    FamilyIncomeLevel, ExtracurricularParticipation, ScreenTimeHours,
    MotivationLevel, TeacherSupportRating, PeerStudyHours, HealthRating
]], columns=[
    'StudyHours', 'AttendancePercent', 'PreviousScore', 'SleepHours',
    'AssignmentsCompleted', 'InternetAccess', 'ParentalEducationLevel',
    'FamilyIncomeLevel', 'ExtracurricularParticipation', 'ScreenTimeHours',
    'MotivationLevel', 'TeacherSupportRating', 'PeerStudyHours',
    'HealthRating'
])

# ---------------- PREDICTION ----------------
if st.button("Predict Performance"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    pass_confidence = probability[1] * 100
    fail_confidence = probability[0] * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("PASS")
        else:
            st.error("FAIL")

    with col2:
        st.metric("Confidence", f"{max(pass_confidence, fail_confidence):.2f}%")

    st.write("Pass Probability:", f"{pass_confidence:.2f}%")
    st.write("Fail Probability:", f"{fail_confidence:.2f}%")


