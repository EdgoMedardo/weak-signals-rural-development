

import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FREQ_COL = 'freq_2021_2025'
OLD_COL = 'freq_1979_2020'


EXCLUDED_STATISTICAL = ['regression', 'index', 'correlation', 'heterogeneity', 'coordination']
EXCLUDED_RHETORICAL = ['influencing', 'promoting', 'comprehensive', 'mechanism',
                       'pathway', 'evolution', 'driver', 'behavior']
EXCLUDED = EXCLUDED_STATISTICAL + EXCLUDED_RHETORICAL

# ---------------------------------------------------------------

# ---------------------------------------------------------------
CLUSTERS = {
    'covid': 'External shock',
    'inclusive': 'Financial inclusion', 'inclusion': 'Financial inclusion',
    'finance': 'Financial inclusion',
    'disparity': 'Territorial inequality', 'policymakers': 'Territorial inequality',
    'adoption': 'Technology adoption',
}

df_all = pd.read_csv("keywords_activeness_frequency.csv")
df = df_all[~df_all['keyword'].isin(EXCLUDED)].copy()

print(f"Señales detectadas: {len(df_all)}")
print(f"Excluidas por el filtro tematico: {len(df_all) - len(df)}")
print(f"Señales retenidas: {len(df)}\n")

freq_median = df[FREQ_COL].median()
act_median = df['activeness'].median()


def quadrant(row):
    if row[FREQ_COL] >= freq_median and row['activeness'] >= act_median:
        return 'Strong Signals'
    if row[FREQ_COL] < freq_median and row['activeness'] >= act_median:
        return 'Weak Signals'
    if row[FREQ_COL] >= freq_median and row['activeness'] < act_median:
        return 'Well-Known Signals'
    return 'Latent Signals'


df['quadrant'] = df.apply(quadrant, axis=1)
df['growth_%'] = ((df[FREQ_COL] - df[OLD_COL]) / df[OLD_COL] * 100).round(1)

# ---------------------------------------------------------------

# ---------------------------------------------------------------
COLOR = {'Weak Signals': '#C0392B', 'Strong Signals': '#2E5A88',
         'Well-Known Signals': '#7F8C8D', 'Latent Signals': '#BDC3C7'}

fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.grid(True, linestyle=':', alpha=0.6, color='gray', zorder=0)
ax.scatter(df[FREQ_COL], df['activeness'], s=np.sqrt(df['TF-IDF'].clip(lower=0)) * 1500,
           c=df['quadrant'].map(COLOR), edgecolors='white', linewidths=0.6,
           alpha=0.9, zorder=3)

texts = [ax.text(r[FREQ_COL], r['activeness'], r['keyword'], fontsize=8.5, zorder=4)
         for _, r in df.iterrows()]
try:
    from adjustText import adjust_text
    adjust_text(texts, ax=ax)
except Exception as e:
    print(f"[aviso] adjustText no disponible ({e}); etiquetas sin ajustar.")

ax.axvline(freq_median, color='black', linestyle=':', alpha=0.6, zorder=1)
ax.axhline(act_median, color='black', linestyle=':', alpha=0.6, zorder=1)

ax.set_xlabel('Frequency 2021-2025', fontsize=11, fontweight='bold')
ax.set_ylabel('Activeness', fontsize=11, fontweight='bold')
ax.set_title('Future signals map (2021-2025)', fontsize=13, fontweight='bold', y=1.02)
ax.set_xlim(0, df[FREQ_COL].max() * 1.12)
ax.set_ylim(0.45, min(1.05, df['activeness'].max() * 1.08))

xmax, ymax = ax.get_xlim()[1], ax.get_ylim()[1]
ax.text(xmax * 0.02, ymax * 0.995, 'WEAK SIGNALS', fontsize=9, color='#C0392B',
        fontweight='bold', va='top')
ax.text(xmax * 0.98, ymax * 0.995, 'STRONG SIGNALS', fontsize=9, color='#2E5A88',
        fontweight='bold', ha='right', va='top')
ax.annotate('Circle size represents keyword relevance (TF-IDF)',
            xy=(0.02, 0.02), xycoords='axes fraction',
            bbox=dict(facecolor='yellow', alpha=0.2, edgecolor='none', pad=3), fontsize=8)
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('FIG_weak_signals_map_final.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Figura: {os.path.abspath('FIG_weak_signals_map_final.png')}\n")

# ---------------------------------------------------------------
# 4. TABLE 1
# ---------------------------------------------------------------
def status(k):
    if k in EXCLUDED_STATISTICAL:
        return 'Excluded (statistical)'
    if k in EXCLUDED_RHETORICAL:
        return 'Excluded (rhetorical)'
    return 'Retained'


t1 = df_all.copy()
t1['Status'] = t1['keyword'].apply(status)
t1 = t1.sort_values('activeness', ascending=False)
t1 = t1[['keyword', 'activeness', FREQ_COL, 'Status']]
t1.columns = ['Weak signal', 'Activeness', 'Frequency (2021-2025)', 'Status']
t1.to_csv('TABLE_1_initial_signals.csv', index=False)
print(f"Tabla 1: {len(t1)} señales iniciales "
      f"({(t1['Status'] == 'Retained').sum()} retenidas, "
      f"{(t1['Status'] != 'Retained').sum()} excluidas)")

# ---------------------------------------------------------------
# 5. TABLE 2
# ---------------------------------------------------------------
t2 = df[df['quadrant'] == 'Weak Signals'].copy()
t2['Thematic cluster'] = t2['keyword'].map(CLUSTERS).fillna('Unassigned')
order = ['External shock', 'Financial inclusion', 'Territorial inequality', 'Technology adoption']
t2['_o'] = t2['Thematic cluster'].map({c: i for i, c in enumerate(order)}).fillna(99)
t2 = t2.sort_values(['_o', 'activeness'], ascending=[True, False])
t2 = t2[['Thematic cluster', 'keyword', 'TF-IDF', 'activeness', OLD_COL, FREQ_COL, 'growth_%']]
t2.columns = ['Thematic cluster', 'Weak signal', 'TF-IDF', 'Activeness',
              'Frequency (1979-2020)', 'Frequency (2021-2025)', 'Growth (%)']
t2.to_csv('TABLE_2_weak_signals_clusters.csv', index=False)

df.to_csv('signals_filtered_final.csv', index=False)

print(f"Medianas -> frecuencia: {freq_median:.0f} | activeness: {act_median:.4f}\n")
for q in ['Weak Signals', 'Strong Signals', 'Well-Known Signals', 'Latent Signals']:
    s = df[df['quadrant'] == q].sort_values('activeness', ascending=False)
    print(f"{q} ({len(s)}): " + ', '.join(s['keyword']))

print("\n--- TABLE 2 ---")
print(t2.to_string(index=False))
print("\nGuardado: TABLE_1_initial_signals.csv, TABLE_2_weak_signals_clusters.csv, "
      "signals_filtered_final.csv, FIG_weak_signals_map_final.png")
