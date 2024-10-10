# Import necessary libraries
import json  # To handle JSON files
import pandas as pd  # For data manipulation with DataFrames
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced visualizations
import string  # For handling string operations
import nltk  # Natural Language Toolkit (NLP library)
from nltk.corpus import stopwords  # For removing common words that don't contribute to meaning
from nltk.stem import WordNetLemmatizer  # For lemmatization (reducing words to base form)

# Download necessary NLTK resources
nltk.download('punkt')  # Tokenizer models
nltk.download('stopwords')  # English stopwords
nltk.download('wordnet')  # WordNet lemmatizer

# Function to load JSON data into a DataFrame
def load_data(filepath):

    with open(filepath, 'r') as f:
        data = json.load(f)  # Load JSON data
    df = pd.DataFrame(data)  # Convert JSON to DataFrame
    return df

# Function to preprocess individual text entries
def preprocess_text(text):
    """Preprocesses a given text by lowercasing, removing punctuation, tokenizing, 
    removing stopwords (except for crucial ones like 'no', 'not', 'yes'), 
    and lemmatizing the tokens.
    """
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation except for important ones (like '?')
    text = text.translate(str.maketrans('', '', string.punctuation.replace("?", "")))

    # Tokenize the text into individual words
    tokens = nltk.word_tokenize(text)
    
    # Define stopwords to remove, keeping some crucial ones
    stop_words = set(stopwords.words('english')) - {'no', 'not', 'yes'}
    
    # Remove stopwords from the tokens
    tokens = [word for word in tokens if word not in stop_words]
    
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize the tokens (reduce to base form)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join the tokens back into a single string
    return ' '.join(tokens)

# Function to apply preprocessing to an entire DataFrame
def preprocess_dataframe(df):

    df['clean_text'] = df['text'].apply(preprocess_text)  # Apply preprocessing to each row
    return df

############# Exploratory Data Analysis (EDA) Functions #############

# Function to plot the distribution of intents in the dataset
def plot_class_distribution(df):
 
    plt.figure(figsize=(8,6))  # Set the plot size
    sns.countplot(x='intent', data=df)  # Create a count plot for the 'intent' column
    plt.title('Intent Distribution')  # Title of the plot
    plt.xlabel('Intent')  # X-axis label
    plt.ylabel('Count')  # Y-axis label
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()  # Display the plot
###############################################################

# Main block of the script for data loading, preprocessing, and visualization
if __name__ == "__main__":
    # Load raw data from JSON
    df = load_data('./data/data.json')
    
    # Preprocess the text data in the DataFrame
    df = preprocess_dataframe(df)
    
    # Save the preprocessed data to a new CSV file
    df.to_csv('./data/preprocessed_data.csv', index=False)
    print("Preprocessing completed and saved to preprocessed_data.csv")

    # Optionally, load the preprocessed data and visualize class distribution
    df = pd.read_csv('./data/preprocessed_data.csv')
    plot_class_distribution(df)  # Plot the distribution of intents
