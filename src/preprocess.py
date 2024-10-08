import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.discard('yes')
    stop_words.discard('no')
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join back to string
    return ' '.join(tokens)

def preprocess_dataframe(df):
    df['clean_text'] = df['text'].apply(preprocess_text)
    return df

############# Exploratory Data Analysis #############  
def load_preprocessed_data(filepath):
    return pd.read_csv(filepath)

def plot_class_distribution(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='intent', data=df)
    plt.title('Intent Distribution')
    plt.xlabel('Intent')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
#####################################################

if __name__ == "__main__":
    df = load_data('./data/data.json')
    df = preprocess_dataframe(df)
    df.to_csv('./data/preprocessed_data.csv', index=False)
    print("Preprocessing completed and saved to preprocessed_data.csv")

    # option to see how the data is clustered
    df = pd.read_csv('./data/preprocessed_data.csv')
    plot_class_distribution(df)
