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


# Utility functions for UI (background, etc.)
def empty_line(lines=1):
    for _ in range(lines):
        st.write("")

def add_logo(logo_url: str, width_percent: float = 0.8, height: int = 220, text_below_logo: str = None):
    """Add a logo (from logo_url) on the top of the navigation page of a multipage app, with width_percent for resizing.
    
    Args:
        logo_url (str): URL/local path of the logo
        width_percent (float): Proportion of the sidebar width for the logo (greater than 1 for overflow)
        height (int): Padding-top value in pixels
        text_below_logo (str): Optional text to display below the logo

    """

    try:
        # Check if the logo_url is a valid URL or local path
        if validators.url(logo_url):
            # If it's a URL, fetch the image
            response = requests.get(logo_url)
            response.raise_for_status()  # Raise an error if the request fails
            image = Image.open(BytesIO(response.content))
        else:
            # If it's a local file path, load the image
            image = Image.open(logo_url)

        # Convert the image to RGB format if it's RGBA
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Resize the image based on the width_percent, maintaining aspect ratio
        original_width, original_height = image.size
        new_width = int(original_width * width_percent)
        new_height = int(original_height * (new_width / original_width))
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality resizing

        # Convert the resized image to base64
        buffered = BytesIO()
        resized_image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode()

        logo = f"url(data:image/png;base64,{base64_image})"

        # Use the original st.markdown style block with height padding
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
        # Optionally add text below the logo within the navigation sidebar
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

# set background wallpaper and subtitle title & sidebar name
def add_bg_from_url():
    # Set background wallpaper
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
    # Add the logo at the top of the sidebar inputs are (image, width, heigth size)
    add_logo("src/LogoBadge.png",width_percent=0.9, text_below_logo="Please select a page")

# Function to load model and vectorizer
@st.cache_resource
def load_model():
    clf = joblib.load('./models/intent_classifier.pkl')
    vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')
    return clf, vectorizer

# Function to initialize the database and create the table if it doesn't exist
def init_db():
    conn = sqlite3.connect('./db/predictions.db')
    c = conn.cursor()

    # Create table if it doesn't exist
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

# Function to predict intent and calculate confidence
def predict_intent(user_input, clf, vectorizer):
    # Language detection
    try:
        lang = detect(user_input)
        if lang != 'en':
            st.warning("Please enter an English sentence.")
            return None, None
    except Exception as e:
        st.warning("Language detection failed. Please try again.")
        return None, None

    input_vector = vectorizer.transform([user_input])
    prediction = clf.predict(input_vector)[0]
    probas = clf.predict_proba(input_vector)[0]
    confidence = probas.max()
    return prediction, confidence

# Function to save prediction to the database
def save_to_db(cursor, user_input, predicted_intent, confidence):
    cursor.execute("INSERT INTO history (input_text, predicted_intent, confidence) VALUES (?, ?, ?)", 
                   (user_input, predicted_intent, confidence))
    cursor.connection.commit()