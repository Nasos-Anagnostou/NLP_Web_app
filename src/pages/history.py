import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from custom_funct import init_db, add_bg_from_url

# Page configuration for the history page
st.set_page_config(page_title="Prediction History", page_icon="📜", layout="centered", initial_sidebar_state="expanded")

# Add background
add_bg_from_url()

# Connect to database
conn = init_db()
c = conn.cursor()

# Show Prediction History
st.title("Prediction History")

try:
    history_df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC LIMIT 100", conn)
    if history_df.empty:
        st.info("No history to show.")
    else:
        st.write("Here is the history of all predictions:")
        st.dataframe(history_df)

        # Plotting confidence distribution
        fig, ax = plt.subplots()
        ax.hist(history_df['confidence'], bins=10, color='blue', edgecolor='black')
        ax.set_title('Confidence Distribution of Predictions')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # Download history button
        csv = history_df.to_csv(index=False)
        st.download_button(label="Download history as CSV", data=csv, mime="text/csv")
except Exception as e:
    st.error(f"An error occurred while fetching history: {e}")

# Close database connection
if 'conn' in locals():
    conn.close()
