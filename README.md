# Is Taylor Swift Musically Ironic?

An exploratory data analysis project asking whether Taylor Swift systematically pairs sad lyrics with upbeat production — and if so, whether that pattern shifts across her discography.

## Approach

- **Sentiment analysis**: `TextBlob` (a lexicon/rule-based Python NLP library — no model downloads, no API calls) scores each lyric line from -1 (negative) to +1 (positive); scores are averaged to a per-song sentiment score.
- **Musical positivity**: average of Spotify's `valence` and `energy` audio features per song.
- **Irony score**: `musical_positivity - lyrical_sentiment`. A positive score means upbeat production paired with sadder lyrics — the "ironic" pattern; near-zero means the two are aligned.
- **Data sources**: a lyrics-by-line dataset and a Spotify audio-features dataset (see [Data sources](#data-sources) below).

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
notebook/03_DataPresentation-TSMusicalIrony.ipynb — the original class presentation notebook (needs the raw files — see Data sources)
data/song_level_sentiment.csv             — derived per-song scores (149 songs), included
outputs/                                  — regenerated charts from the last run
```

## Why the raw lyrics aren't in this repo

Song lyrics are copyrighted. `01_build_song_level_dataset.py` documents exactly how the raw lyrics-by-line dataset and the Spotify audio-features dataset were merged and scored, but neither raw file is bundled here, and the pipeline never writes lyric text back out — only the resulting per-song aggregate scores. `data/song_level_sentiment.csv` is a derived, transformative dataset (numeric sentiment/audio scores per song title), not a reproduction of the underlying copyrighted text.

## Data sources

Neither raw file is included here. To reproduce the pipeline or run the presentation notebook, you'll need to assemble them yourself:

| File | Source | Columns used |
|---|---|---|
| `taylor_swift_lyrics_full.csv` | Started from PromptCloud's [Taylor Swift Song Lyrics from all the albums](https://www.kaggle.com/datasets/PromptCloudHQ/taylor-swift-song-lyrics-from-all-the-albums) (Kaggle; covers the debut album through *reputation*), then extended by me in the same one-row-per-lyric-line format through *The Life of a Showgirl* (2025), including the Taylor's Version re-recordings | `artist, album, track_title, track_n, lyric, line, year` |
| `Taylor Swift Spotify Data 11-24-2024.csv` | Pulled from the [Spotify Web API](https://developer.spotify.com/documentation/web-api) on 2024-11-24: Taylor Swift's album tracks plus their audio features | `track_name, album_name, danceability, energy, loudness, liveness, valence, tempo` |

Place both files in `data/` with those exact names. Then either:

- point `LYRICS_PATH` / `AUDIO_FEATURES_PATH` in `01_build_song_level_dataset.py` at them and run it to regenerate `data/song_level_sentiment.csv`, or
- open `notebook/03_DataPresentation-TSMusicalIrony.ipynb` from the `notebook/` folder (it reads `../data/...`).

Note: Spotify deprecated the audio-features endpoint for new API apps in November 2024, so a fresh pull may not return `valence`, `energy`, etc.

## Running the analysis

```bash
pip install pandas numpy matplotlib seaborn scipy textblob
python notebook/02_irony_analysis.py
```

No corpora downloads needed — `TextBlob`'s polarity scoring uses a bundled pattern-based lexicon, not an external model.

## Origin

Originally built as a data presentation for DS 5610 (Exploratory Data Analysis), Vanderbilt MSDS program, Fall 2025.
