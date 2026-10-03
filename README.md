# Is Taylor Swift Musically Ironic?

An exploratory data analysis project asking whether Taylor Swift systematically pairs sad lyrics with upbeat production — and if so, whether that pattern shifts across her discography.

## Approach

- **Sentiment analysis**: `TextBlob` (a lexicon/rule-based Python NLP library — no model downloads, no API calls) scores each lyric line from -1 (negative) to +1 (positive); scores are averaged to a per-song sentiment score.
- **Musical positivity**: average of Spotify's `valence` and `energy` audio features per song.
- **Irony score**: `musical_positivity - lyrical_sentiment`. A positive score means upbeat production paired with sadder lyrics — the "ironic" pattern; near-zero means the two are aligned.
- **Data sources**: a lyrics-by-line dataset and a Spotify audio-features dataset, both originally sourced from Kaggle.

## Key findings

- Taylor Swift's lyrics skew slightly positive overall (mean sentiment +0.045)
- 91.9% of songs (137 of 149) show the ironic pattern — upbeat music, sadder lyrics
- Mean irony score of 0.439 across the catalog
- Irony level varies meaningfully by album — *1989* runs highest (0.575), *Midnights* lowest (0.261)
- No single audio feature (danceability, energy, valence, tempo, etc.) individually correlates strongly with lyrical sentiment — the irony pattern is a catalog-wide tendency rather than something explained by any one production choice

## Repo structure

```
notebook/01_build_song_level_dataset.py   — documents the merge + sentiment pipeline (not run automatically — see below)
notebook/02_irony_analysis.py             — the actual analysis; runs as-is against data/song_level_sentiment.csv
data/song_level_sentiment.csv             — derived per-song scores (149 songs), included
outputs/                                  — regenerated charts from the last run
```

## Why the raw lyrics aren't in this repo

Song lyrics are copyrighted. `01_build_song_level_dataset.py` documents exactly how the raw lyrics-by-line dataset and the Spotify audio-features dataset were merged and scored, but neither raw file is bundled here, and the pipeline never writes lyric text back out — only the resulting per-song aggregate scores. `data/song_level_sentiment.csv` is a derived, transformative dataset (numeric sentiment/audio scores per song title), not a reproduction of the underlying copyrighted text.

To reproduce `song_level_sentiment.csv` yourself: source a Taylor Swift lyrics-by-line dataset and the "Spotify Musical Analysis" audio-features dataset (both were available on Kaggle at the time of the original analysis, Aug-Sep 2025), point `01_build_song_level_dataset.py` at your local copies, and run it.

## Running the analysis

```bash
pip install pandas numpy matplotlib seaborn scipy textblob
python notebook/02_irony_analysis.py
```

No corpora downloads needed — `TextBlob`'s polarity scoring uses a bundled pattern-based lexicon, not an external model.

## Origin

Originally built as a data presentation for DS 5610 (Exploratory Data Analysis), Vanderbilt MSDS program, Fall 2025.
