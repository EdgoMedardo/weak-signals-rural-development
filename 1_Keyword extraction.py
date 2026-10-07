
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# Get English stopwords
stop_words = set(stopwords.words('english'))


def process_text(text):
    # Ensure the text is a string
    if isinstance(text, pd.Series):
        text = ' '.join(text.dropna().astype(str))
    elif not isinstance(text, str):
        text = str(text)

    # Tokenization and stopword removal
    tokens = [token.lower() for token in word_tokenize(text) if token.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    # Only 5-grams
    all_grams = [' '.join(gram) for gram in list(ngrams(lemmas, 5))]

    return all_grams


def extract_keywords_from_csv(file_path):
    df = pd.read_csv(file_path)
    title_text = ' '.join(df['Title'].dropna().astype(str))
    abstract_text = ' '.join(df['Abstract'].dropna().astype(str))
    combined_text = title_text + ' ' + abstract_text
    return process_text(combined_text)


def extract_author_keywords(file_path):
    df = pd.read_csv(file_path)
    keywords = df['Author Keywords'].dropna().str.split(';').explode().str.strip().tolist()
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(keyword.lower()) for keyword in keywords]



csv_files = [
    'scopus(2025).csv', 'scopus(2023-2024).csv', 'scopus(2021-2022).csv'
]

# Archivos que forman cada subperiodo para el calculo de tendencia
FILES_EARLY = ['scopus(2021-2022).csv']                       # 2021-2022
FILES_RECENT = ['scopus(2023-2024).csv', 'scopus(2025).csv']  # 2023-2025

# List 1: Keywords extracted from titles and abstracts
list1 = []
for filename in csv_files:
    list1.extend(extract_keywords_from_csv(filename))

print(f"Number of keywords in List 1: {len(set(list1))}")

# List 2: Author keywords
list2 = []
for filename in csv_files:
    list2.extend(extract_author_keywords(filename))

print(f"Number of keywords in List 2: {len(set(list2))}")

combined_list = set(list1 + list2)
print(f"Total number of unique keywords: {len(combined_list)}")


def count_keywords_by_period(csv_files, combined_list):
    """Cuenta las palabras clave en el periodo anterior (2021-2022)
    y en el periodo reciente (2023-2025)."""
    count_early = Counter()
    count_recent = Counter()

    for filename in csv_files:
        keywords = extract_keywords_from_csv(filename) + extract_author_keywords(filename)
        if filename in FILES_EARLY:
            count_early.update(keywords)
        elif filename in FILES_RECENT:
            count_recent.update(keywords)

    return count_early, count_recent


def calculate_trend(keyword, count_early, count_recent):
    return count_recent.get(keyword, 0) - count_early.get(keyword, 0)


count_early, count_recent = count_keywords_by_period(csv_files, combined_list)

positive_trend_keywords = [keyword for keyword in combined_list
                           if calculate_trend(keyword, count_early, count_recent) > 0]

print(f"Number of keywords with a positive trend: {len(positive_trend_keywords)}")


def perform_tfidf(keywords, csv_files):
    documents = []
    for file in csv_files:
        df = pd.read_csv(file)
        doc = ' '.join(df['Title'].fillna('') + ' ' + df['Abstract'].fillna(''))
        documents.append(' '.join(process_text(doc)))

    vectorizer = TfidfVectorizer(vocabulary=keywords, ngram_range=(1, 5), min_df=2)
    tfidf_matrix = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names_out()
    tfidf_means = np.array(tfidf_matrix.mean(axis=0)).flatten()

    return dict(zip(feature_names, tfidf_means))


print("\nPerforming TF-IDF analysis on keywords with a positive trend...")
tfidf_scores = perform_tfidf(positive_trend_keywords, csv_files)

sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
filtered_keywords = [keyword for keyword, score in sorted_keywords if score > 0.01]

print(f"\nNumber of keywords with a positive trend and TF-IDF > 0.01: {len(filtered_keywords)}")

print("\nTop 10 filtered keywords:")
for keyword in filtered_keywords[:10]:
    print(f"{keyword}: {tfidf_scores[keyword]:.4f}")

output_df_filtered = pd.DataFrame([(keyword, tfidf_scores[keyword]) for keyword in filtered_keywords],
                                  columns=['Keyword', 'TF-IDF Score'])
output_df_filtered.to_csv('filtered_keywords_tfidf_greater_001.csv', index=False)
print("\nSaved: 'filtered_keywords_tfidf_greater_001.csv'")
