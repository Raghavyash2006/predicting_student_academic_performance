# import pandas as pd
# import joblib
# import os

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# print("Loading dataset...")

# # LOAD DATA
# df = pd.read_csv("data/student_performance.csv")

# # CREATE PASS/FAIL TARGET
# df["Result"] = df["FinalScore"].apply(lambda x: 1 if x >= 60 else 0)


# # FEATURES AND TARGET
# X = df.drop(["FinalScore", "Result"], axis=1)
# y = df["Result"]

# # TRAIN TEST SPLIT
# X_train, X_test, y_train, y_test, = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )



# # MODELS
# models = {
#     "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=2000),
#     "RandomForest": RandomForestClassifier(class_weight="balanced"),
#     "GradientBoosting": GradientBoostingClassifier(),
# }


# results = []

# best_score = 0
# best_model = None

# print("Training models...")

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)

#     acc = accuracy_score(y_test, preds)
#     results.append([name, acc])

#     if acc > best_score:
#         best_score = acc
#         best_model = model

# results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

# print(results_df)

# # SAVE MODELS
# os.makedirs("models", exist_ok=True)

# joblib.dump(best_model, "models/best_model.pkl")
# joblib.dump(results_df, "models/model_results.pkl")

# print("Training complete. Best model saved.")
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f6f8fb;
}

.title {
    font-size: 42px;
    font-weight: bold;
    color: #1f4e79;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.result-pass {
    background-color: #e8f5e9;
    padding: 20px;
    border-radius: 10px;
    font-size: 24px;
    color: #2e7d32;
    text-align: center;
}

.result-fail {
    background-color: #fdecea;
    padding: 20px;
    border-radius: 10px;
    font-size: 24px;
    color: #c62828;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">Student Performance Predictor</div>', unsafe_allow_html=True)
st.write("AI system to predict PASS/FAIL outcome with confidence.")

model = joblib.load("models/best_model.pkl")
results_df = joblib.load("models/model_results.pkl")

# ---------------- MODEL TABLE ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Comparison")
st.dataframe(results_df, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FEATURE IMPORTANCE ----------------
if hasattr(model, "feature_importances_"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR INPUT ----------------
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

    col1, col2 = st.columns([2,1])

    with col1:
        if prediction == 1:
            st.markdown('<div class="result-pass">PASS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-fail">FAIL</div>', unsafe_allow_html=True)

    with col2:
        st.metric("Confidence", f"{max(pass_confidence, fail_confidence):.2f}%")

    st.write("Pass Probability:", f"{pass_confidence:.2f}%")
    st.write("Fail Probability:", f"{fail_confidence:.2f}%")

