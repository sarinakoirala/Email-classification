import numpy as np
import pandas as pd
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# 1. Load and Clean Data
df = pd.read_csv('spam.csv')

# Drop unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Rename columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode target labels
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Check missing values
print("Missing values:", df.isnull().sum())

# Check and remove duplicates
print("Duplicates before removal:", df.duplicated().sum())
df = df.drop_duplicates(keep='first')
print("Duplicates after removal:", df.duplicated().sum())
print("Dataset shape:", df.shape)

# 2. Add text features for EDA
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Display statistics
print("\nTarget Distribution:")
print(df['target'].value_counts())
print("\nOverall Statistics:")
print(df[['num_characters', 'num_words', 'num_sentences']].describe())
print("\nHam Messages Statistics:")
print(df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe())
print("\nSpam Messages Statistics:")
print(df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe())

# 3. Text Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    # Lowercase
    text = text.lower()
    
    # Tokenization
    text = nltk.word_tokenize(text)
    
    # Remove special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    # Remove stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)
    
    # Stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Apply text transformation
df['transformed_text'] = df['text'].apply(transform_text)
print("\nSample transformed text:")
print(df[['text', 'transformed_text']].head())

# 4. Model Building
# Create feature matrix using TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train Naive Bayes classifier
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Make predictions
y_pred = mnb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")

# 5. Save the model and vectorizer
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))

print("\nModel saved successfully!")
print("Files created: vectorizer.pkl, model.pkl")

