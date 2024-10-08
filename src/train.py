import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def load_preprocessed_data(filepath):
    return pd.read_csv(filepath)

def train_model(df):
    # Replace NaN values in the 'clean_text' column with values from the 'text' column
    df['clean_text'] = df['clean_text'].fillna(df['text'])
    
    X = df['clean_text']
    y = df['intent']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # # Vectorization
    # vectorizer = TfidfVectorizer()
    # Vectorization with bigrams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # # Model Training
    # clf = LogisticRegression(max_iter=1000)
    # Model Training with class weights to handle imbalance
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')

    clf.fit(X_train_tfidf, y_train)
    
    # Evaluation
    y_pred = clf.predict(X_test_tfidf)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Model and Vectorizer
    joblib.dump(clf, './models/intent_classifier.pkl')
    joblib.dump(vectorizer, './models/tfidf_vectorizer.pkl')
    print("Model and Vectorizer saved.")

if __name__ == "__main__":
    df = load_preprocessed_data('./data/preprocessed_data.csv')
    train_model(df)