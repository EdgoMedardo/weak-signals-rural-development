# Weak Signals in Rural Development

Code accompanying:

Agualsaca Janeta, E. M. Detection of Weak Signals in Rural Development through
Scientific Text Mining Techniques. *Discover Sustainability* (under review).

## Pipeline

Run in this order:

1. `1_Keyword extraction.py` — preprocessing and TF-IDF keyword extraction
   (Sections 3.2–3.3 of the manuscript)
2. `2_Activeness.py` — activeness indicator and the 44 initial weak signals
   (Section 3.4)
3. `3b_Filter and final map.py` — thematic relevance filter and the final
   emergence map (Sections 3.5–3.6; generates Table 1, Table 2 and Fig. 4)
4. `3c_Collocation analysis.py` — collocation analysis used to justify the
   filter and to support the Discussion (Section 3.5)
5. `4_Evolution Keywords.py` — temporal evolution of the seven weak signals
   (Fig. 5)
6. `5_Evolution publications per year.py` — annual publication trend and
   exponential fit (Fig. 2)

## Data

The raw bibliographic records were extracted from Scopus and cannot be
redistributed due to licensing restrictions. The derived dataset (298
keywords with TF-IDF scores) is provided as Supplementary Information
accompanying the published article.

## License

MIT License (see `LICENSE`).
