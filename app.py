# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, session
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg') # Important for server environment (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import time
from datetime import datetime

# Download NLTK data (run once)
try:
    nltk.download('stopwords')
except:
    pass

app = Flask(__name__)
app.secret_key = 'kip_kuliah_sentiment_analysis_secret_key_2023'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load pre-trained models and vectorizers
try:
    with open('model/nb_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    with open('model/svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Models loaded successfully!")
except FileNotFoundError:
    print("Model files not found. Please train models first.")
    nb_model = None
    svm_model = None
    vectorizer = None

# --- Helper Functions for NLP ---
def preprocess_text(text):
    """Fungsi preprocessing teks"""
    if isinstance(text, float) or text is None:
        return ""
    
    # Case folding
    text = str(text).lower()
    # Remove punctuation, numbers, and special chars
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Tokenization
    tokens = text.split()
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('indonesian'))
        tokens = [t for t in tokens if t not in stop_words]
    except:
        tokens = tokens  # Jika stopwords tidak ada, lanjut tanpa remove stopwords
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    # Join back to string
    return ' '.join(tokens)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}

# --- Define Routes ---

@app.route('/')
def index():
    """Homepage/Dashboard"""
    # Calculate summary statistics if available
    summary_stats = {
        'total_analysis': 0,
        'positive_count': 0,
        'negative_count': 0,
        'neutral_count': 0
    }
    
    return render_template('index.html', summary=summary_stats)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada file yang dipilih', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('Nama file kosong', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid filename conflicts
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Store file info in session
            session['current_file'] = filename
            session['upload_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            flash('File berhasil diunggah! Silakan lanjutkan ke preprocessing.', 'success')
            return redirect(url_for('preprocess', filename=filename))
        else:
            flash('Format file tidak didukung. Harus .csv atau .xlsx', 'danger')
    
    return render_template('upload.html')

@app.route('/preprocess/<filename>')
def preprocess(filename):
    """Show data before and after preprocessing"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        flash('File tidak ditemukan', 'danger')
        return redirect(url_for('upload_file'))
    
    try:
        # Try to read CSV first
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        flash(f'Error membaca file: {str(e)}', 'danger')
        return redirect(url_for('upload_file'))
    
    # Assume the text column is named 'ulasan'
    if 'ulasan' not in df.columns:
        flash('File harus memiliki kolom "ulasan"', 'danger')
        return redirect(url_for('upload_file'))
    
    # Get first 10 records for display
    sample_data = df.head(10).to_dict('records')
    
    # Preprocess a sample for display
    processed_sample = []
    for record in sample_data:
        processed_text = preprocess_text(record.get('ulasan', ''))
        processed_record = record.copy()
        processed_record['ulasan_processed'] = processed_text
        processed_sample.append(processed_record)
    
    # Count total rows
    total_rows = len(df)
    
    return render_template('preprocessing.html', 
                         original_data=sample_data, 
                         processed_data=processed_sample, 
                         filename=filename,
                         total_rows=total_rows)

@app.route('/analysis')
def analysis():
    """Analysis selection page"""
    filename = request.args.get('filename')
    if not filename:
        flash('Silakan unggah file terlebih dahulu', 'warning')
        return redirect(url_for('upload_file'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('File tidak ditemukan', 'danger')
        return redirect(url_for('upload_file'))
    
    # Count rows in the file
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        row_count = len(df)
    except Exception as e:
        flash(f'Error membaca file: {str(e)}', 'danger')
        return redirect(url_for('upload_file'))
    
    # Get upload time from session or use current time
    upload_time = session.get('upload_time', datetime.now().strftime('%Y-%m-%d %H:%M'))
    
    return render_template('analysis.html', 
                         filename=filename, 
                         data_count=row_count,
                         upload_time=upload_time)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Perform classification with NB and SVM"""
    algorithm = request.form.get('algorithm')
    filename = request.form.get('filename')
    
    if not filename:
        flash('Filename tidak valid', 'danger')
        return redirect(url_for('upload_file'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        flash('File tidak ditemukan', 'danger')
        return redirect(url_for('upload_file'))
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        flash(f'Error membaca file: {str(e)}', 'danger')
        return redirect(url_for('upload_file'))
    
    if 'ulasan' not in df.columns:
        flash('File harus memiliki kolom "ulasan"', 'danger')
        return redirect(url_for('upload_file'))
    
    # Preprocess all texts
    df['processed_text'] = df['ulasan'].apply(preprocess_text)
    
    # Check if models are loaded
    if vectorizer is None or nb_model is None or svm_model is None:
        flash('Model machine learning belum diload. Silakan train model terlebih dahulu.', 'danger')
        return redirect(url_for('upload_file'))
    
    # Vectorize the text
    try:
        X = vectorizer.transform(df['processed_text'])
    except Exception as e:
        flash(f'Error dalam vectorization: {str(e)}', 'danger')
        return redirect(url_for('upload_file'))
    
    # Predict based on chosen algorithm
    try:
        if algorithm == 'naive_bayes':
            predictions = nb_model.predict(X)
            model_name = "Naive Bayes"
        elif algorithm == 'svm':
            predictions = svm_model.predict(X)
            model_name = "SVM"
        else:
            flash('Algoritma tidak valid', 'danger')
            return redirect(url_for('upload_file'))
    except Exception as e:
        flash(f'Error dalam prediksi: {str(e)}', 'danger')
        return redirect(url_for('upload_file'))
    
    df['predicted_sentiment'] = predictions
    
    # Calculate value counts for chart
    sentiment_counts = df['predicted_sentiment'].value_counts().to_dict()
    
    # Generate a bar chart
    img = BytesIO()
    plt.figure(figsize=(10, 6))
    
    # Prepare data for chart
    sentiments = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())
    colors = ['green' if 'positif' in str(s).lower() else 'red' if 'negatif' in str(s).lower() else 'blue' for s in sentiments]
    
    bars = plt.bar(sentiments, counts, color=colors, alpha=0.7)
    plt.title(f'Distribusi Sentimen - {model_name}')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    # Generate word cloud data (simplified)
    positive_text = ' '.join(df[df['predicted_sentiment'].str.lower().str.contains('positif')]['ulasan'].fillna(''))
    negative_text = ' '.join(df[df['predicted_sentiment'].str.lower().str.contains('negatif')]['ulasan'].fillna(''))
    
    # Prepare results for the template
    results = df.to_dict('records')
    
    # Store results in session for download
    session['last_results'] = {
        'filename': filename,
        'algorithm': algorithm,
        'results': results,
        'sentiment_counts': sentiment_counts,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return render_template('results.html', 
                         results=results[:50],  # Show first 50 results
                         counts=sentiment_counts,
                         plot_url=plot_url,
                         algorithm=model_name,
                         filename=filename,
                         total_results=len(results),
                         positive_text=positive_text[:200] + '...' if len(positive_text) > 200 else positive_text,
                         negative_text=negative_text[:200] + '...' if len(negative_text) > 200 else negative_text)

@app.route('/compare')
def compare_models():
    """Compare both models"""
    filename = request.args.get('filename')
    
    if not filename:
        flash('Silakan unggah file terlebih dahulu', 'warning')
        return redirect(url_for('upload_file'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        flash('File tidak ditemukan', 'danger')
        return redirect(url_for('upload_file'))
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        flash(f'Error membaca file: {str(e)}', 'danger')
        return redirect(url_for('upload_file'))
    
    if 'ulasan' not in df.columns:
        flash('File harus memiliki kolom "ulasan"', 'danger')
        return redirect(url_for('upload_file'))
    
    # Preprocess text
    df['processed_text'] = df['ulasan'].apply(preprocess_text)
    
    # Check if models are loaded
    if vectorizer is None or nb_model is None or svm_model is None:
        flash('Model machine learning belum diload. Silakan train model terlebih dahulu.', 'danger')
        return redirect(url_for('upload_file'))
    
    # Vectorize
    try:
        X = vectorizer.transform(df['processed_text'])
    except Exception as e:
        flash(f'Error dalam vectorization: {str(e)}', 'danger')
        return redirect(url_for('upload_file'))
    
    # Predict with both models
    try:
        nb_predictions = nb_model.predict(X)
        svm_predictions = svm_model.predict(X)
    except Exception as e:
        flash(f'Error dalam prediksi: {str(e)}', 'danger')
        return redirect(url_for('upload_file'))
    
    # Calculate accuracy if actual labels exist
    has_actual_labels = 'sentimen' in df.columns
    nb_accuracy = accuracy_score(df['sentimen'], nb_predictions) if has_actual_labels else 0
    svm_accuracy = accuracy_score(df['sentimen'], svm_predictions) if has_actual_labels else 0
    
    # Count predictions
    nb_counts = pd.Series(nb_predictions).value_counts().to_dict()
    svm_counts = pd.Series(svm_predictions).value_counts().to_dict()
    
    # Prepare results
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append({
            'ulasan': row['ulasan'],
            'nb_prediction': nb_predictions[i],
            'svm_prediction': svm_predictions[i],
            'actual': row.get('sentimen', 'Tidak Ada')
        })
    
    # Generate comparison chart
    img = BytesIO()
    models = ['Naive Bayes', 'SVM']
    accuracies = [nb_accuracy, svm_accuracy]
    
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange']
    bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
    plt.title('Perbandingan Akurasi Model')
    plt.ylabel('Akurasi')
    plt.ylim(0, 1 if max(accuracies) > 0 else 1)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        if accuracy > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{accuracy:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    # Store comparison results in session
    session['comparison_results'] = {
        'filename': filename,
        'results': results,
        'nb_counts': nb_counts,
        'svm_counts': svm_counts,
        'nb_accuracy': nb_accuracy,
        'svm_accuracy': svm_accuracy,
        'has_actual_labels': has_actual_labels
    }
    
    return render_template('comparison.html',
                         results=results[:30],  # Show first 30 results
                         nb_counts=nb_counts,
                         svm_counts=svm_counts,
                         nb_accuracy=nb_accuracy,
                         svm_accuracy=svm_accuracy,
                         plot_url=plot_url,
                         filename=filename,
                         has_actual_labels=has_actual_labels,
                         total_results=len(results))

@app.route('/download')
def download_results():
    """Download analysis results as CSV"""
    if 'last_results' not in session:
        flash('Tidak ada hasil analisis yang tersedia untuk diunduh', 'warning')
        return redirect(url_for('index'))
    
    results_data = session['last_results']
    filename = results_data['filename']
    algorithm = results_data['algorithm']
    results = results_data['results']
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Create download filename
    download_filename = f"hasil_analisis_{algorithm}_{filename}"
    
    # Create response
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8')
    output.seek(0)
    
    return send_file(output, 
                     as_attachment=True, 
                     download_name=download_filename,
                     mimetype='text/csv')

@app.route('/visualization')
def visualization():
    """Advanced visualization page"""
    if 'last_results' not in session:
        flash('Silakan jalankan analisis terlebih dahulu', 'warning')
        return redirect(url_for('index'))
    
    results_data = session['last_results']
    sentiment_counts = results_data['sentiment_counts']
    
    # Generate multiple charts
    charts = {}
    
    # Pie chart
    img1 = BytesIO()
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('Distribusi Sentimen')
    plt.axis('equal')
    plt.savefig(img1, format='png', dpi=100)
    img1.seek(0)
    charts['pie_chart'] = base64.b64encode(img1.getvalue()).decode('utf8')
    plt.close()
    
    # Bar chart
    img2 = BytesIO()
    plt.figure(figsize=(10, 6))
    sentiments = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())
    colors = ['green' if 'positif' in str(s).lower() else 'red' if 'negatif' in str(s).lower() else 'blue' for s in sentiments]
    
    bars = plt.bar(sentiments, counts, color=colors, alpha=0.7)
    plt.title('Distribusi Sentimen')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(img2, format='png', dpi=100)
    img2.seek(0)
    charts['bar_chart'] = base64.b64encode(img2.getvalue()).decode('utf8')
    plt.close()
    
    return render_template('visualization.html',
                         charts=charts,
                         sentiment_counts=sentiment_counts,
                         algorithm=results_data['algorithm'])

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    """API endpoint for sentiment analysis"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Text is required'}), 400
    
    text = data['text']
    processed_text = preprocess_text(text)
    
    if vectorizer is None or nb_model is None or svm_model is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    # Vectorize text
    X = vectorizer.transform([processed_text])
    
    # Predict with both models
    nb_prediction = nb_model.predict(X)[0]
    svm_prediction = svm_model.predict(X)[0]
    
    return jsonify({
        'text': text,
        'processed_text': processed_text,
        'naive_bayes_prediction': nb_prediction,
        'svm_prediction': svm_prediction
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# Ganti bagian terakhir app.py
if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    # Untuk production di PythonAnywhere
    app.run(debug=False)  # Pastikan debug=False
    
    print("Server starting...")
    print("Available routes:")
    print("  - / : Homepage")
    print("  - /upload : File upload")
    print("  - /preprocess/<filename> : Preprocessing")
    print("  - /analysis : Analysis selection")
    print("  - /compare : Model comparison")
    print("  - /visualization : Advanced charts")
    print("  - /about : About page")
    print("  - /api/sentiment : API endpoint")
    
    app.run(debug=True, host='0.0.0.0', port=5000)