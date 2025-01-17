from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Inisialisasi Flask
app = Flask(__name__)

# Load dataset
try:
    data = pd.read_csv('data/dataset.csv')  # Sesuaikan path dataset
    data = data.dropna(subset=['description', 'title', 'url'])
except Exception as e:
    raise ValueError(f"Error loading dataset: {e}")

# Preprocessing teks
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()  # Ubah menjadi huruf kecil
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = re.sub(r'\d+', '', text)  # Hapus angka
    return text.strip()

# Preprocess seluruh deskripsi berita
data['processed_description'] = data['description'].apply(preprocess_text)

# TF-IDF Vectorizer
try:
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_description'])
except Exception as e:
    raise ValueError(f"Error in TF-IDF processing: {e}")

# Konfigurasi jumlah hasil per halaman
RESULTS_PER_PAGE = 10  # Menampilkan 10 hasil per halaman

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    query = request.args.get('query', '')  # Ambil query dari URL jika ada
    page = request.args.get('page', 1, type=int)  # Ambil halaman saat ini
    
    if request.method == 'POST':  # Jika pencarian dilakukan melalui form
        query = request.form.get('query', '').strip()
    
    if query:
        processed_query = preprocess_text(query)
        query_vector = tfidf_vectorizer.transform([processed_query])
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        total_results = len(similarity_scores)

        if total_results > 0:
            top_indices = similarity_scores.argsort()[::-1]
            start_index = (page - 1) * RESULTS_PER_PAGE
            end_index = start_index + RESULTS_PER_PAGE

            if start_index < total_results:
                top_indices = top_indices[start_index:end_index]
                recommendations = data.iloc[top_indices][['title', 'url', 'description', 'source']].to_dict(orient='records')

        # Hitung total halaman
        total_pages = (total_results // RESULTS_PER_PAGE) + (1 if total_results % RESULTS_PER_PAGE > 0 else 0)

        # Hitung range halaman untuk navigasi
        start_page = max(1, page - 2)
        end_page = min(total_pages, page + 2)

    else:
        # Jika tidak ada query, kosongkan hasil
        similarity_scores = []
        total_pages = 0
        start_page = 1
        end_page = 1

    return render_template('index.html', 
                           recommendations=recommendations, 
                           query=query, 
                           page=page, 
                           total_pages=total_pages, 
                           start_page=start_page, 
                           end_page=end_page)

if __name__ == '__main__':
    app.run(debug=True)
