import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk import download as nltk_download
from tqdm import tqdm
import pandas as pd
import os

# Download stopwords if not already available
try:
    stopwords.words('indonesian')
except LookupError:
    nltk_download('stopwords')

# Initialize settings for text cleaning
punctuations = set(string.punctuation)
stopwords_indo = set(stopwords.words('indonesian'))
additional_stopwords = ["nya", "nih", "saya", "aja", "aku", "gak", "kalau", "apa", "bri", "knpa", "ceria"]
stopwords_indo.update(additional_stopwords)

stemmer = StemmerFactory().create_stemmer()
url_pattern = re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE)
emoji_pattern = re.compile(r'[^\w\s,]')

def remove_special_characters(text):
    return emoji_pattern.sub('', url_pattern.sub('', text))

def preprocess_text(sentence):
    if isinstance(sentence, str):
        return " ".join([
            stemmer.stem(token) for token in remove_special_characters(sentence).split()
            if token not in stopwords_indo and token not in punctuations
        ])
    return ""

def preprocess_data(df):
    df = df[df['content'].str.split().str.len() >= 3]
    tqdm.pandas()
    df['content_processed'] = df['content'].progress_apply(preprocess_text)
    return df[df.content_processed.notna()]

def remove_specific_words(df):
    df['content_processed'] = df['content_processed'].apply(
        lambda text: " ".join([word for word in text.split() if word not in additional_stopwords])
    )
    return df

# Apply Replacement on Content
def replace_brand_words(text):
    return re.sub(r'\b(bri ceria|ceria|akulaku|atome|kredivo|bri)\b', 'xyz', text, flags=re.IGNORECASE)