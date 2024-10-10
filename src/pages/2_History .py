# Import necessary libraries
import streamlit as st  # Streamlit for building the web app UI
import pandas as pd  # For handling data with DataFrames
import sqlite3  # To interact with the SQLite database
from custom_funct import empty_line,init_db, add_bg_from_url  # Custom functions for database initialization and background setup
import plotly.express as px  # For plotting the confidence distribution


# Configure the Streamlit page for the history display
st.set_page_config(
    page_title="Prediction History",  # Title of the web page
    page_icon="📜",  # Icon displayed in the tab
    layout="wide",  # Use full width of the page for a wide layout
    initial_sidebar_state="expanded"  # Expand the sidebar by default
)

# Set background image for the page
add_bg_from_url()

# Connect to the SQLite database using the init_db function (which creates the table if it doesn't exist)
conn = init_db()
c = conn.cursor()  # Create a cursor object to interact with the database

# Display the title of the history page
st.title("Prediction History")

# Try to fetch and display the prediction history from the database
try:
    # SQL query to retrieve the last 100 prediction records from the 'history' table
    history_df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC LIMIT 100", conn)
    
    # Check if the DataFrame is empty (i.e., no history records)
    if history_df.empty:
        st.info("No history to show.")  # Show a message if no records are found
    else:
        st.write("Here is the history of all predictions:")  # Text above the table

        # Display the retrieved history as a table with container width for full-page width
        st.dataframe(history_df, use_container_width=True)
        
        # Allow the user to download the history as a CSV file
        csv = history_df.to_csv(index=False)  # Convert DataFrame to CSV format without index column
        st.download_button(
            label="Download history as CSV",  # Label for the download button
            data=csv,  # CSV data to be downloaded
            mime="text/csv"  # MIME type for CSV
        )

        empty_line(2)    
        if st.button("Plot the Confidence Distribution"):
            # Plotting with Plotly for interactivity
            fig = px.histogram(
                history_df, 
                x='confidence', 
                nbins=10, 
                title="Confidence Distribution of Predictions", 
                labels={'confidence': 'Confidence'},
                template='plotly_dark'  # Dark theme for better visual
            )
            fig.update_layout(
                title_font_size=16,
                xaxis_title='Confidence',
                yaxis_title='Frequency',
                bargap=0.2
            )
            st.plotly_chart(fig)

except Exception as e:
    # If any error occurs while fetching the data, display the error message
    st.error(f"An error occurred while fetching history: {e}")


# Close the database connection at the end of the script
if 'conn' in locals():
    conn.close()
