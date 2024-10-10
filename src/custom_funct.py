# Required libraries
from PIL import Image
from io import BytesIO
import streamlit as st
from streamlit_extras.app_logo import add_logo
import base64
from langdetect import detect
import sqlite3
import joblib
import validators
import requests


# Utility function to add empty lines in the Streamlit UI
def empty_line(lines=1):
    for _ in range(lines):
        st.write("")


# Function to add a logo in the sidebar with optional text below the logo
def add_logo(logo_url: str, width_percent: float = 0.8, height: int = 220, text_below_logo: str = None):
    """Adds a logo at the top of the navigation page/sidebar of the Streamlit app.
    """
    try:
        # Check if the logo_url is a valid URL or a local path
        if validators.url(logo_url):
            # Fetch the image from the URL if valid
            response = requests.get(logo_url)
            response.raise_for_status()  # Error if the request fails
            image = Image.open(BytesIO(response.content))
        else:
            # Load local image file
            image = Image.open(logo_url)

        # Convert image to RGB if in RGBA format (strip alpha channel)
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Resize image based on the width_percent parameter, maintaining aspect ratio
        original_width, original_height = image.size
        new_width = int(original_width * width_percent)
        new_height = int(original_height * (new_width / original_width))
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)  # High-quality resizing

        # Convert the resized image to base64 to embed in HTML
        buffered = BytesIO()
        resized_image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode()

        # CSS styling for the sidebar with the logo and text (if any)
        logo = f"url(data:image/png;base64,{base64_image})"
        st.markdown(
            f"""
            <style>
                [data-testid="stSidebarNav"] {{
                    background-image: {logo};
                    background-repeat: no-repeat;
                    padding-top: {height}px;
                    background-position: center 25px;
                    background-size: {width_percent * 100}%;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        # Optionally, add text below the logo in the sidebar
        if text_below_logo:
            st.markdown(
                f"""
                <style>
                    [data-testid="stSidebarNav"]::before {{
                        content: '{text_below_logo}';
                        display: block;
                        text-align: center;
                        padding-top: 0px;
                        padding-bottom: +10px;
                        font-size: 14px;
                        color: #WHITE;
                    }}
                </style>
                """,
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Error: {str(e)}")


# Set the background wallpaper and logo in the Streamlit app
def add_bg_from_url():
    """Sets a background wallpaper and adds a logo with text below it in the sidebar."""
    # CSS to set the background wallpaper
    st.markdown(
        f"""
       <style>
       .stApp {{
       background-image: url("https://static.vecteezy.com/system/resources/previews/021/565/019/non_2x/minimalist-abstract-background-design-smooth-and-clean-subtle-background-vector.jpg");
       background-attachment: fixed;
       background-size: cover
       }}
       </style>
       """,
        unsafe_allow_html=True
    )
    
    # Add logo to the sidebar
    add_logo("src/LogoBadge.png", width_percent=0.9, text_below_logo="Please select a page")


# Load the trained model and vectorizer (cached for better performance)
@st.cache_resource
def load_model():
    """Loads the pre-trained classification model and vectorizer from disk."""
    clf = joblib.load('./models/intent_classifier.pkl')
    vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')
    return clf, vectorizer


# Initialize the SQLite database and create table if it doesn't exist
def init_db():
    """Creates a database connection and ensures the 'history' table exists."""
    conn = sqlite3.connect('./db/predictions.db')
    c = conn.cursor()

    # Create the history table if it doesn't exist already
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            predicted_intent TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    ''')
    conn.commit()
    return conn


# Predict the intent of the user input and calculate the confidence score
def predict_intent(user_input, clf, vectorizer):

    # Transform the input text using the vectorizer and predict using the model
    input_vector = vectorizer.transform([user_input])
    prediction = clf.predict(input_vector)[0]
    probas = clf.predict_proba(input_vector)[0]
    confidence = probas.max()
    return prediction, confidence


# Save the prediction and confidence score to the database
def save_to_db(cursor, user_input, predicted_intent, confidence):
    """Inserts the prediction results into the SQLite database."""
    cursor.execute("INSERT INTO history (input_text, predicted_intent, confidence) VALUES (?, ?, ?)", 
                   (user_input, predicted_intent, confidence))
    cursor.connection.commit()
