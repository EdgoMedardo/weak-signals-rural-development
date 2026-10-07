
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# ---------------------------------------------------------------
# WEEK SIGNALS
# ---------------------------------------------------------------
SIGNALS = [
    ('covid',        'External shock'),
    ('inclusive',    'Financial inclusion'),
    ('finance',      'Financial inclusion'),
    ('inclusion',    'Financial inclusion'),
    ('policymakers', 'Territorial inequality'),
    ('disparity',    'Territorial inequality'),
    ('adoption',     'Technology adoption'),
]
keywords_to_track = [k for k, _ in SIGNALS]

YEAR_MIN, YEAR_MAX = 1979, 2025
WINDOW_START, WINDOW_END = 2021, 2025
BLUE = '#2E5A88'

csv_files = [
    'scopus(2025).csv', 'scopus(2023-2024).csv', 'scopus(2021-2022).csv',
    'scopus(2019-2020).csv', 'scopus(2014-2018).csv', 'scopus(2006-2013).csv',
    'scopus(1979-2005).csv'
]

lemmatizer = WordNetLemmatizer()


def process_text(text):
    tokens = [t.lower() for t in word_tokenize(str(text)) if t.lower() not in stop_words]
    return [lemmatizer.lemmatize(t) for t in tokens]


def count_by_year(file_path, kws):
    df = pd.read_csv(file_path).dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    df = df[(df['Year'] >= YEAR_MIN) & (df['Year'] <= YEAR_MAX)]

    out = {}
    for year, sub in df.groupby('Year'):
        text = ' '.join(sub['Title'].dropna().astype(str)) + ' ' + \
               ' '.join(sub['Abstract'].dropna().astype(str)) + ' ' + \
               ' '.join(sub['Author Keywords'].dropna().astype(str))
        toks = process_text(text)
        uni = Counter(toks)
        bi = Counter(zip(toks, toks[1:]))
        res = {}
        for kw in kws:
            parts = kw.split()
            if len(parts) == 1:
                res[kw] = uni.get(parts[0], 0)
            elif len(parts) == 2:
                res[kw] = bi.get((parts[0], parts[1]), 0)
            else:
                res[kw] = sum(1 for i in range(len(toks) - len(parts) + 1)
                              if toks[i:i + len(parts)] == parts)
        out[year] = res
    return out


freqs = {kw: {} for kw in keywords_to_track}
for filename in csv_files:
    for year, counts in count_by_year(filename, keywords_to_track).items():
        for kw, n in counts.items():
            freqs[kw][year] = freqs[kw].get(year, 0) + n
    print('procesado', filename)

df_freq = pd.DataFrame(freqs).fillna(0).sort_index()
df_freq.index.name = 'Year'
df_freq.to_csv('keyword_evolution_by_year.csv')


def draw(ax, kw, cluster, letter=None):
    s = df_freq[kw]
    ax.axvspan(WINDOW_START - 0.5, WINDOW_END + 0.5, color=BLUE, alpha=0.08, zorder=0)
    ax.plot(s.index, s.values, marker='o', lw=2, markersize=3.5, color=BLUE, zorder=3)
    ax.fill_between(s.index, s.values, alpha=0.12, color=BLUE, zorder=2)
    ax.annotate(f'{int(s.iloc[-1])}', xy=(s.index[-1], s.iloc[-1]), xytext=(-3, 6),
                textcoords='offset points', ha='right', fontsize=9,
                fontweight='bold', color=BLUE)
    ax.set_xlim(YEAR_MIN, YEAR_MAX + 1)
    ax.set_ylim(0, max(s.max() * 1.20, 1))
    ax.set_xticks(np.arange(1980, YEAR_MAX + 1, 10))
    title = f'{letter}) {kw}' if letter else kw
    ax.set_title(f'{title}\n{cluster}', fontsize=10, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


# (a) paneles individuales
for kw, cluster in SIGNALS:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    draw(ax, kw, cluster)
    ax.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'trend_{kw}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# (b) figura combinada de 7 paneles (recomendada para el manuscrito)
fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=300)
letters = 'abcdefg'
for i, (kw, cluster) in enumerate(SIGNALS):
    draw(axes.flat[i], kw, cluster, letters[i])
for j in range(len(SIGNALS), 9):
    axes.flat[j].axis('off')
fig.supxlabel('Year', fontsize=12, fontweight='bold')
fig.supylabel('Frequency', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('FIG5_weak_signal_trends.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("\nGenerado: FIG5_weak_signal_trends.png (figura combinada) "
      "y 7 paneles individuales trend_*.png")
print("Datos en 'keyword_evolution_by_year.csv'")
print(df_freq.tail(8).to_string())
