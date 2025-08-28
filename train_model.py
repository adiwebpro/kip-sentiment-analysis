# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK data
nltk.download('stopwords')

def preprocess_text(text):
    """Fungsi preprocessing teks"""
    # Case folding
    text = text.lower()
    # Remove punctuation, numbers, and special chars
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Tokenization
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [t for t in tokens if t not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    # Join back to string
    return ' '.join(tokens)

def train_models():
    """Train and save ML models"""
    # Load dataset (ganti dengan path dataset Anda)
    # Dataset harus memiliki kolom 'ulasan' dan 'sentimen'
    print("Loading dataset...")
    df = pd.read_csv('dataset_kip_kuliah.csv')
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['ulasan'].apply(preprocess_text)
    
    # Split data
    X = df['processed_text']
    y = df['sentimen']  # Pastikan kolom ini ada
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Vectorization
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Naive Bayes
    print("Training Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    
    # Train SVM
    print("Training SVM...")
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_vec, y_train)
    
    # Evaluate models
    print("\n=== Evaluation Results ===")
    
    # Naive Bayes
    nb_pred = nb_model.predict(X_test_vec)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
    
    # SVM
    svm_pred = svm_model.predict(X_test_vec)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    
    # Save models and vectorizer
    print("Saving models...")
    with open('model/nb_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    
    with open('model/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Models saved successfully!")

if __name__ == '__main__':
    train_models()