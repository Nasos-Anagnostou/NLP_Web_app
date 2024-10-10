import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from custom_funct import add_bg_from_url, empty_line, init_db, load_model, predict_intent, save_to_db
import sqlite3
from st_pages import add_page_title, get_nav_from_toml


# Set page config with multiple pages
st.set_page_config(page_title="Intent Classification", page_icon="🔮", layout="centered",
                   initial_sidebar_state="expanded", menu_items=None)

# Add background
add_bg_from_url()

# set the homepage style
title = '<p style="font-family:Arial Black; color:white; font-size: 300%; text-align: center;">Intent Classification for Booking Appointments</p>'
st.markdown(title, unsafe_allow_html=True)
# Add some space
empty_line(2)

# App description
st.write("Enter a sentence below, and the system will predict its intent.")
st.write("All predictions are saved and can be viewed in the history.")
empty_line(2)

# Load Model and Vectorizer
clf, vectorizer = load_model()

# Initialize Database Connection
conn = init_db()
c = conn.cursor()

# Initialize session state for user input and feedback
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "feedback_given" not in st.session_state:
    st.session_state["feedback_given"] = False

# Split layout into two columns
col1, col2 = st.columns([2, 2])

# Show input and prediction form
st.session_state["user_input"] = col1.text_input("🔮 Enter a sentence to predict its intent:", st.session_state["user_input"])

# "Predict Intent" button remains in the first column
if col2.button("🔮 Predict Intent"):
    user_input = st.session_state["user_input"]
    if user_input.strip() == "" or len(user_input.split()) < 2:
        st.warning("Please enter a meaningful sentence with at least two words.")
    else:
        with st.spinner('Predicting...'):
            try:
                prediction, confidence = predict_intent(user_input, clf, vectorizer)
                if prediction is not None:
                    st.success(f"**Predicted Intent:** {prediction} (Confidence: {confidence:.2f})")
                    save_to_db(c, user_input, prediction, confidence)

                    # Feedback mechanism with radio buttons in the same column
                    feedback = st.radio("Is this prediction correct?", ("Yes", "No"))
                    if feedback == "No":
                        st.session_state["feedback_given"] = False  # Reset state
                        correct_intent = col2.text_input("Please provide the correct intent:")
                        if col2.button("Submit Feedback"):
                            st.session_state["feedback_given"] = True  # Mark feedback as given
                            st.success("Thank you for your feedback!")
                    elif feedback == "Yes":
                        st.session_state["feedback_given"] = True  # No need for feedback submission
                        st.success("Thank you for your confirmation!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Move "Show History" button to align with input in the first column
if col1.button("📜 Show Prediction History"):
    switch_page("history")  # Navigating to the history page

# Move "Give me an example!" button below "Predict Intent" as a dropdown (selectbox)
example_options = ["Book an appointment", "Cancel my meeting", "Reschedule my consultation"]
selected_example = col1.selectbox("Select an example sentence:", ["Choose an example"] + example_options)

# If a sentence is selected, set it as the text input without rerun
if selected_example != "Choose an example":
    st.session_state["user_input"] = selected_example
    st.experimental_set_query_params(user_input=selected_example)
