
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def process_text(text):
    if isinstance(text, pd.Series):
        text = ' '.join(text.dropna().astype(str))
    elif not isinstance(text, str):
        text = str(text)

    tokens = [token.lower() for token in word_tokenize(text) if token.lower() not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    # N-grams (using 1 to 5-grams)
    all_grams = []
    for n in range(1, 6):
        n_grams = list(ngrams(lemmas, n))
        all_grams.extend([' '.join(gram) for gram in n_grams])

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


# ---------------------------------------------------------------
# PERIODO TOTAL (1979-2025) y PERIODO RECIENTE (2021-2025)
# ---------------------------------------------------------------
all_files = [
    'scopus(2025).csv', 'scopus(2023-2024).csv', 'scopus(2021-2022).csv',
    'scopus(2019-2020).csv', 'scopus(2014-2018).csv', 'scopus(2006-2013).csv',
    'scopus(1979-2005).csv'
]

recent_files = ['scopus(2025).csv', 'scopus(2023-2024).csv', 'scopus(2021-2022).csv']


def count_keywords(keywords, file_list):
    counter = Counter()
    for filename in file_list:
        file_keywords = extract_keywords_from_csv(filename) + extract_author_keywords(filename)
        counter.update([kw for kw in file_keywords if kw in keywords])
    return counter


# Palabras clave filtradas en la Fase 1
df_filtered = pd.read_csv("filtered_keywords_tfidf_greater_001.csv")
tfidf_map = dict(zip(df_filtered['Keyword'].str.lower(), df_filtered['TF-IDF Score']))
filtered_keywords = set(tfidf_map.keys())

recent_counts = count_keywords(filtered_keywords, recent_files)
all_counts = count_keywords(filtered_keywords, all_files)

results = []
for keyword in filtered_keywords:
    recent_freq = recent_counts[keyword]
    all_freq = all_counts[keyword]
    if all_freq > 0:
        activeness = recent_freq / all_freq
        if activeness >= 0.5:            # criterio de exclusion original
            results.append({
                'keyword': keyword,
                'activeness': activeness,
                'freq_1979_2020': all_freq - recent_freq,
                'freq_2021_2025': recent_freq,
                'total frequency': all_freq,
                'TF-IDF': tfidf_map.get(keyword, float('nan'))
            })

df_results = pd.DataFrame(results).sort_values('activeness', ascending=False)

# Columna de color (se puede editar despues para agrupar por cluster tematico)
df_results['color'] = 'gray'

print(f"Number of keywords with Activeness >= 0.5: {len(df_results)}")

df_results.to_csv('keywords_activeness.csv', index=False)
df_results.to_csv('keywords_activeness_frequency.csv', index=False)
print("Saved: 'keywords_activeness.csv' y 'keywords_activeness_frequency.csv'")
print("\nTop 20 por activeness:")
print(df_results.head(20).to_string(index=False))
