import streamlit as st
import joblib
import sqlite3
import pandas as pd
import os

# Title and Description
st.title('Intent Classification for Booking Appointments')
st.write("""
Enter a sentence below, and the system will predict its intent. All predictions are saved and can be viewed in the history.
""")

# Load Model and Vectorizer using st.cache_resource
@st.cache_resource
def load_model():
    clf = joblib.load('./models/intent_classifier.pkl')
    vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')
    return clf, vectorizer

clf, vectorizer = load_model()

# Database Connection
def init_db():
    conn = sqlite3.connect('./db/predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            predicted_intent TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn

conn = init_db()
c = conn.cursor()

# User Input
user_input = st.text_input("Enter your sentence:", "")

# Predict Intent
if st.button("Predict Intent"):
    if user_input.strip() == "":
        st.warning("Please enter a valid sentence.")
    else:
        # Preprocess the input (same as training)
        input_vector = vectorizer.transform([user_input])
        prediction = clf.predict(input_vector)[0]
        
        st.success(f"**Predicted Intent:** {prediction}")
        
        # Save to Database
        c.execute("INSERT INTO history (input_text, predicted_intent) VALUES (?, ?)", (user_input, prediction))
        conn.commit()

# Show History
if st.button("Show History"):
    history_df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC LIMIT 100", conn)
    if history_df.empty:
        st.info("No history to show.")
    else:
        st.dataframe(history_df)

# Close Database Connection
conn.close()
