# create_sample_models.py
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK data
try:
    nltk.download('stopwords')
except:
    pass

def preprocess_text(text):
    """Fungsi preprocessing teks"""
    if isinstance(text, float):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('indonesian'))
    tokens = [t for t in tokens if t not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

def create_sample_models():
    """Create sample models for demonstration"""
    print("Creating sample models...")
    
    # Create sample data
    sample_data = {
        'ulasan': [
            'kip kuliah sangat membantu mahasiswa kurang mampu',
            'program kip kuliah bagus sekali untuk pendidikan',
            'sangat terbantu dengan adanya kip kuliah',
            'proses pendaftaran kip kuliah terlalu rumit',
            'bantuan kip kuliah tidak tepat sasaran',
            'administrasi kip kuliah berbelit-belit',
            'kip kuliah memberikan harapan untuk kuliah',
            'bantuan dana kip sangat bermanfaat',
            'syarat mendapatkan kip kuliah terlalu banyak',
            'informasi kip kuliah tidak jelas',
            'kip kuliah meringankan biaya pendidikan',
            'program yang sangat bermanfaat untuk rakyat',
            'pelayanan kip kuliah perlu ditingkatkan',
            'dana kip sering terlambat cair',
            'sistem seleksi kip tidak transparan'
        ],
        'sentimen': [
            'Positif', 'Positif', 'Positif', 
            'Negatif', 'Negatif', 'Negatif',
            'Positif', 'Positif', 'Negatif',
            'Negatif', 'Positif', 'Positif',
            'Negatif', 'Negatif', 'Negatif'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Preprocess text
    df['processed_text'] = df['ulasan'].apply(preprocess_text)
    
    # Vectorization
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['sentimen']
    
    # Train models
    nb_model = MultinomialNB()
    nb_model.fit(X, y)
    
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X, y)
    
    # Save models
    with open('model/nb_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    
    with open('model/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Sample models created successfully!")
    print("Models saved in 'model/' directory")

if __name__ == '__main__':
    create_sample_models()