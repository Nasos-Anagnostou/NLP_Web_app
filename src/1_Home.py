import streamlit as st  # Streamlit for UI components
from streamlit_extras.switch_page_button import switch_page  # For navigation between pages
from custom_funct import add_bg_from_url, empty_line, init_db, load_model, predict_intent, save_to_db  # Custom utility functions
import sqlite3  # SQLite for database connection
from st_pages import add_page_title, get_nav_from_toml  # For page title and navigation management (optional usage)

# Configure Streamlit page settings
st.set_page_config(
    page_title="Intent Classification",  # Title of the page in the browser tab
    page_icon="🔮",  # Icon shown in the browser tab
    layout="centered",  # Page layout, centered in the browser
    initial_sidebar_state="expanded",  # Sidebar starts expanded
    menu_items=None  # Default menu items
)

# Add custom background using a function from custom_funct.py
add_bg_from_url()

# Set the homepage title with custom styling
title = '<p style="font-family:Arial Black; color:white; font-size: 300%; text-align: center;">Intent Classification for Booking Appointments</p>'
st.markdown(title, unsafe_allow_html=True)  # Display styled title in the center of the page

# Add empty lines for spacing
empty_line(2)

# Brief description of the app functionality
st.write("Enter a sentence below, and the system will predict its intent.")
st.write("All predictions are saved and can be viewed in the history.")
empty_line(1)

# Load the pre-trained model and vectorizer from custom_funct.py
clf, vectorizer = load_model()

# Initialize the SQLite database and cursor for data interaction
conn = init_db()
c = conn.cursor()

# Initialize session state variables to store user input and feedback
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""  # For storing the text input from the user
if "feedback_given" not in st.session_state:
    st.session_state["feedback_given"] = False  # Tracks whether feedback has been provided
if "last_selected_example" not in st.session_state:
    st.session_state["last_selected_example"] = "Choose an example"  # Tracks the last example selected

# Create two columns for text input and example sentence selection
col1, col2 = st.columns([2, 1], gap="medium", vertical_alignment="top")

# Text input box for entering the sentence to classify
st.session_state["user_input"] = col1.text_input(
    "🔮 Enter a sentence to predict its intent:",  # Label for the input box
    st.session_state["user_input"]  # Pre-fills the input with previously entered value
)

with col2:
    # Example sentence options
    example_options = ["Book an appointment", "Cancel my meeting", "Reschedule my consultation"]
    
    # Dropdown box to select from pre-defined example sentences
    selected_example = st.selectbox(
        "Or select an example sentence:",  # Label for the dropdown
        ["Choose an example"] + example_options  # Add default option and example sentences
    )

    # Automatically fill the text input when an example is selected
    if selected_example != st.session_state["last_selected_example"]:
        st.session_state["user_input"] = selected_example  # Update the input box with the example
        st.session_state["last_selected_example"] = selected_example  # Track the last selected example

        # Use query parameters to store the selected example for easier page navigation
        st.query_params["user_input"] = selected_example
        st.rerun()  # Rerun the app to reflect the changes

# "Predict Intent" button and action logic
with col1:
    empty_line(2)  # Add some spacing before the button

    # When the "Predict Intent" button is clicked
    if st.button("🔮 Predict Intent"):
        user_input = st.session_state["user_input"]  # Retrieve the current text input

        # Validate that the input is meaningful (at least two words)
        if user_input.strip() == "" or len(user_input.split()) < 2:
            st.warning("Please enter a meaningful sentence with at least two words.")
        else:
            # Show spinner while prediction is being made
            with st.spinner('Predicting...'):
                try:
                    # Call predict_intent from custom_funct.py to get the prediction and confidence
                    prediction, confidence = predict_intent(user_input, clf, vectorizer)

                    # If a prediction is successfully made, display it
                    if prediction is not None:
                        st.success(f"**Predicted Intent:** {prediction} (Confidence: {confidence:.2f})")
                        
                        # Save the prediction to the database for future reference
                        save_to_db(c, user_input, prediction, confidence)

                        # Collect feedback from the user on whether the prediction was correct
                        with col2:
                            feedback = st.selectbox(
                                "Is this prediction correct?",  # Label for the feedback dropdown
                                ("Choose", "Yes", "No", "I don't know")  # Feedback options
                            )

                except Exception as e:
                    # Show error if something goes wrong during prediction
                    st.error(f"An error occurred: {e}")

    empty_line(2)  # Add space before history button

    # Button to show prediction history, which redirects to another page
    st.markdown("##### View Your Prediction History")  # Header for the history section
    if st.button("📜Show History"):
        switch_page("history ")  # Navigate to the "history" page using switch_page function
