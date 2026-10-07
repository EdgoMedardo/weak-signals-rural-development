
import math
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------------
#
# ---------------------------------------------------------------
RECENT_FILES = ['scopus(2021-2022).csv', 'scopus(2023-2024).csv', 'scopus(2025).csv']
YEAR_MIN, YEAR_MAX = 2021, 2025

MIN_WORD_FREQ = 20   # el colocado debe aparecer al menos 20 veces en el subcorpus
MIN_PAIR_FREQ = 10   # el par debe aparecer al menos 10 veces
TOP_N = 8            # colocados a mostrar por señal

FUNCTION_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def tokenize(text):
    tokens = [t.lower() for t in word_tokenize(str(text)) if t.isalpha()]
    return [lemmatizer.lemmatize(t) for t in tokens]


# ---------------------------------------------------------------
#
# ---------------------------------------------------------------
targets = set(pd.read_csv('keywords_activeness_frequency.csv')['keyword'].str.lower())
single_targets = {t for t in targets if ' ' not in t}

unigrams = Counter()
bigrams = Counter()
total_tokens = 0

for filename in RECENT_FILES:
    df = pd.read_csv(filename).dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    df = df[(df['Year'] >= YEAR_MIN) & (df['Year'] <= YEAR_MAX)]

    for title, abstract, kws in zip(df['Title'].fillna(''),
                                    df['Abstract'].fillna(''),
                                    df['Author Keywords'].fillna('')):
        toks = tokenize(f"{title} {abstract} {kws}")
        unigrams.update(toks)
        bigrams.update(zip(toks, toks[1:]))
        total_tokens += len(toks)
    print('procesado', filename)

print(f"\nSubcorpus {YEAR_MIN}-{YEAR_MAX}: {total_tokens:,} tokens\n")


def ppmi(w1, w2):
    """Positive Pointwise Mutual Information del par (w1, w2)."""
    pair = bigrams[(w1, w2)]
    if pair == 0:
        return 0.0
    p_xy = pair / total_tokens
    p_x = unigrams[w1] / total_tokens
    p_y = unigrams[w2] / total_tokens
    return max(math.log2(p_xy / (p_x * p_y)), 0.0)


# ---------------------------------------------------------------
# ---------------------------------------------------------------
rows = []
for term in sorted(single_targets):
    if unigrams[term] == 0:
        continue

    preceding, following = [], []

    for (w1, w2), n in bigrams.items():
        if n < MIN_PAIR_FREQ:
            continue
        if w2 == term and w1 not in FUNCTION_WORDS and unigrams[w1] >= MIN_WORD_FREQ:
            preceding.append((w1, n, ppmi(w1, term)))
        elif w1 == term and w2 not in FUNCTION_WORDS and unigrams[w2] >= MIN_WORD_FREQ:
            following.append((w2, n, ppmi(term, w2)))

    preceding.sort(key=lambda x: x[2], reverse=True)
    following.sort(key=lambda x: x[2], reverse=True)

    print(f'### {term.upper()}   (frecuencia total: {unigrams[term]:,})')
    print('   PRECEDIDO POR: ' + ', '.join(
        f'{w} (n={n}, PPMI={p:.2f})' for w, n, p in preceding[:TOP_N]))
    print('   SEGUIDO DE:    ' + ', '.join(
        f'{w} (n={n}, PPMI={p:.2f})' for w, n, p in following[:TOP_N]))
    print()

    for w, n, p in preceding[:TOP_N]:
        rows.append({'signal': term, 'position': 'preceding', 'collocate': w,
                     'pair_frequency': n, 'collocate_frequency': unigrams[w],
                     'PPMI': round(p, 3)})
    for w, n, p in following[:TOP_N]:
        rows.append({'signal': term, 'position': 'following', 'collocate': w,
                     'pair_frequency': n, 'collocate_frequency': unigrams[w],
                     'PPMI': round(p, 3)})

pd.DataFrame(rows).to_csv('collocation_analysis_ppmi.csv', index=False)
print('=' * 70)
print(f"Parametros: MIN_WORD_FREQ={MIN_WORD_FREQ}, MIN_PAIR_FREQ={MIN_PAIR_FREQ}")
print("Guardado: 'collocation_analysis_ppmi.csv'")
