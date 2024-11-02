# topic_modeling.py

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.preprocessing.preprocess_text import preprocess_data, remove_specific_words  # import preprocessing functions
from src.utils.file_utils import load_data, save_to_csv, save_model  # import utility functions

# Gensim based imports
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models.coherencemodel import CoherenceModel

# Define directories
DATA_DIR = "data/processed"
MODEL_DIR = "models"

# Vectorization and LDA model creation
def vectorize_text(df):
    """
    Mengubah teks menjadi representasi vektor untuk digunakan dalam LDA.
    """
    vectorizer = CountVectorizer(lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    return vectorizer, vectorizer.fit_transform(df["content_processed"])

def create_lda_model(data_vectorized, num_topics=5):
    """
    Membuat model LDA berdasarkan data vektor dan jumlah topik yang diinginkan.
    """
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=99, max_iter=2, learning_method='online')
    return lda

def get_coherence_score(model, df_):
    topics = model.components_

    n_top_words = 10
    texts = [[word for word in doc.split()] for doc in df_]

    # create the dictionary
    dictionary = Dictionary(texts)
    # Create a gensim dictionary from the word count matrix

    # Create a gensim corpus from the word count matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    feature_names = [dictionary[i] for i in range(len(dictionary))]

    # Get the top words for each topic from the components_ attribute
    top_words = []
    for topic in topics:
        top_words.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

    coherence_model = CoherenceModel(topics=top_words, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()
    return coherence

def get_top_words(model, vectorizer, n_top_words=10):
    """
    Mengambil top words dari setiap topic.
    """
    return pd.DataFrame([
        (idx, vectorizer.get_feature_names_out()[i], topic[i])
        for idx, topic in enumerate(model.components_)
        for i in topic.argsort()[:-n_top_words - 1:-1]
    ], columns=["Topic", "Word", "Weight"])

# Main function to run the topic modeling process
def run_topic_modeling(input_file):
    """
    Pipeline utama untuk melakukan topic modeling:
    - Memuat dan memproses data
    - Melakukan vektorisasi
    - Membuat model LDA
    - Menyimpan hasil topik dan model
    """
    # Load and preprocess data
    df = load_data(input_file)
    df = df.iloc[0:10]
    df = preprocess_data(df)
    df = remove_specific_words(df)
    
    # Save preprocessed data
    save_to_csv(df, os.path.join(DATA_DIR, 'ceria_preprocessed.csv'))
    
    # Vectorize and create LDA model
    vectorizer, data_vectorized = vectorize_text(df)

    # Finding optimal number of topics
    coherence_scores = {}
    for num_topics in range(2, 4):
        # Latent Dirichlet Allocation Model
        lda = create_lda_model(data_vectorized, num_topics)
        data_lda = lda.fit_transform(data_vectorized)
        score = get_coherence_score(lda, df.content_processed)
        coherence_scores[num_topics] = score

    num_optimal_topics =  max(coherence_scores, key=coherence_scores.get)
    print("Optimal Topics: {}".format(num_optimal_topics))
    lda = create_lda_model(data_vectorized, num_optimal_topics)
    lda.fit_transform(data_vectorized)

    # Extract top words and save them
    top_words_df = get_top_words(lda, vectorizer)
    top_words_path = os.path.join(DATA_DIR, 'top_words_df.csv')
    save_to_csv(top_words_df, top_words_path)
    
    # Save LDA model and vectorizer
    save_model(lda, os.path.join(MODEL_DIR, 'lda_model.pkl'))
    save_model(vectorizer, os.path.join(MODEL_DIR, 'vectorizer_model.pkl'))
    print(f"LDA model, vectorizer, and top words DataFrame saved successfully at {top_words_path}!")