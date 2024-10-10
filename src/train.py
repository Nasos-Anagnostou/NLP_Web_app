# Import necessary libraries
import pandas as pd  # For data manipulation with DataFrames
from sklearn.model_selection import train_test_split  # For splitting the dataset into train/test sets
from sklearn.feature_extraction.text import TfidfVectorizer  # To convert text data into TF-IDF vectors
from sklearn.linear_model import LogisticRegression  # For model training using logistic regression
from sklearn.metrics import classification_report  # For model evaluation metrics
import joblib  # For saving trained models

# Function to load preprocessed data from a CSV file
def load_preprocessed_data(filepath):
    """Loads preprocessed data from a CSV file.
    
    Args:
        filepath (str): Path to the preprocessed CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    return pd.read_csv(filepath)

# Function to train the intent classification model
def train_model(df):
    """Trains a logistic regression model on the preprocessed text data.
    
    The model predicts the intent of the input text based on the cleaned and tokenized text.
    It uses TF-IDF vectorization with bigrams (unigrams and bigrams) and applies class balancing.
    
    Args:
        df (pd.DataFrame): DataFrame containing preprocessed text data and intent labels.
    
    Returns:
        None: The function saves the trained model and the TF-IDF vectorizer as .pkl files.
    """
    
    # Replace NaN values in the 'clean_text' column with values from the 'text' column if any exist
    df['clean_text'] = df['clean_text'].fillna(df['text'])
    
    # Define feature (X) and target (y) variables
    X = df['clean_text']  # Input text data (after preprocessing)
    y = df['intent']  # Target labels (intents)
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Stratified split ensures class distribution is balanced
    )
    
    # Vectorization with TF-IDF (using unigrams and bigrams)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Convert text into TF-IDF vectors considering both single words and pairs of words
    
    # Fit the vectorizer on the training data and transform both train and test data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Model Training using Logistic Regression
    # Setting class_weight='balanced' to handle any imbalance in class distribution
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')  # Using balanced class weights to handle imbalance
    
    # Fit the model to the training data
    clf.fit(X_train_tfidf, y_train)
    
    # Model Evaluation: Generate predictions on the test set
    y_pred = clf.predict(X_test_tfidf)
    
    # Print the classification report to assess model performance
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  # Display precision, recall, and F1-score for each class
    
    # Save the trained model and vectorizer for later use
    joblib.dump(clf, './models/intent_classifier.pkl')  # Save the model as a .pkl file
    joblib.dump(vectorizer, './models/tfidf_vectorizer.pkl')  # Save the vectorizer as a .pkl file
    print("Model and Vectorizer saved.")

# Main execution block
if __name__ == "__main__":
    # Load preprocessed data
    df = load_preprocessed_data('./data/preprocessed_data.csv')
    
    # Train the model on the preprocessed data
    train_model(df)
