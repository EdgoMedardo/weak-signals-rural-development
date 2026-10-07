
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

YEAR_MIN, YEAR_MAX = 1979, 2025
WINDOW_START = 2021

file_paths = [
    'scopus(2025).csv', 'scopus(2023-2024).csv', 'scopus(2021-2022).csv',
    'scopus(2019-2020).csv', 'scopus(2014-2018).csv', 'scopus(2006-2013).csv',
    'scopus(1979-2005).csv',
]

data_frames = []
for file_path in file_paths:
    try:
        data_frames.append(pd.read_csv(file_path, usecols=['Year', 'Title', 'Abstract', 'Author Keywords']))
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
        print(f"Error with file {file_path}: {e}")

data = pd.concat(data_frames, ignore_index=True).dropna(subset=['Year'])
data['Year'] = data['Year'].astype(int)
data = data[(data['Year'] >= YEAR_MIN) & (data['Year'] <= YEAR_MAX)]

year_counts = data['Year'].value_counts().sort_index()
year_counts.to_csv('publications_by_year.csv', header=['publications'])

years = year_counts.index.values
vals = year_counts.values.astype(float)

print(f"Total de registros ({YEAR_MIN}-{YEAR_MAX}): {int(vals.sum()):,}")
print(f"Registros en la ventana {WINDOW_START}-{YEAR_MAX}: "
      f"{int(vals[years >= WINDOW_START].sum()):,} "
      f"({100 * vals[years >= WINDOW_START].sum() / vals.sum():.1f} %)")

x = years - years.min()
b, ln_a = np.polyfit(x, np.log(vals), 1)
a = np.exp(ln_a)
y_fit = a * np.exp(b * x)
r2 = 1 - ((vals - y_fit) ** 2).sum() / ((vals - vals.mean()) ** 2).sum()

# --- Figura ---
fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
colors = ['#2E5A88' if y >= WINDOW_START else '#A9C4DE' for y in years]
ax.bar(years, vals, color=colors, width=0.78, zorder=3)
ax.plot(years, y_fit, '--', color='#C0392B', lw=2, zorder=4)

ax.set_xticks(np.arange(1980, YEAR_MAX + 1, 5))
ax.set_xlim(YEAR_MIN - 1, YEAR_MAX + 1.5)
ax.set_ylim(0, vals.max() * 1.15)
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of publications', fontsize=12, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.45, zorder=0)
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

ax.annotate(f'{int(vals[-1]):,}', xy=(years[-1], vals[-1]), xytext=(0, 6),
            textcoords='offset points', ha='center', fontsize=10,
            fontweight='bold', color='#2E5A88')

ax.legend(handles=[
    Patch(facecolor='#A9C4DE', label=f'{YEAR_MIN}–{WINDOW_START - 1}'),
    Patch(facecolor='#2E5A88', label=f'Analysis window ({WINDOW_START}–{YEAR_MAX})'),
    plt.Line2D([0], [0], color='#C0392B', ls='--', lw=2,
               label=f'Exponential trend: y = {a:.1f}·e^({b:.3f}x),  R² = {r2:.3f}')],
    loc='upper left', frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig('publications_by_year.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Ajuste: y = {a:.2f}·e^({b:.4f}x)   R² = {r2:.3f}")
print("Guardado: 'publications_by_year.png' y 'publications_by_year.csv'")
