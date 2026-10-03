"""
Stage 1 — build the song-level sentiment/irony dataset from raw sources.

NOT run automatically as part of this repo (see README for why) — this
documents exactly how data/song_level_sentiment.csv was produced. To
reproduce it yourself:

  1. Get a Taylor Swift lyrics-by-line dataset (Kaggle has several; look
     for one with columns: artist, album, track_title, track_n, lyric,
     line, year).
  2. Get the "Spotify Musical Analysis" audio-features dataset (also on
     Kaggle) with columns including track_name, album_name, danceability,
     energy, loudness, liveness, valence, tempo.
  3. Point LYRICS_PATH / AUDIO_FEATURES_PATH below at your local copies
     and run this script.

Raw lyric text is never written back out by this script — only per-song
aggregate sentiment scores. That's intentional: song lyrics are
copyrighted, and a public repo shouldn't redistribute them even
incidentally via an intermediate CSV.
"""
import pandas as pd
from textblob import TextBlob

LYRICS_PATH = "your_lyrics_by_line.csv"           # not included — see README
AUDIO_FEATURES_PATH = "your_audio_features.csv"    # not included — see README
OUTPUT_PATH = "data/song_level_sentiment.csv"


def clean_track_name(name: str) -> str:
    """Strip parenthetical suffixes like '(Taylor's Version)' for matching."""
    if " (" in name:
        return name[: name.index(" (")]
    return name


def main():
    lyrics = pd.read_csv(LYRICS_PATH)
    audio = pd.read_csv(AUDIO_FEATURES_PATH)

    audio["track_name_formatted"] = audio["track_name"].apply(clean_track_name)
    keep_cols = [
        "track_name_formatted", "album_name", "danceability", "energy",
        "loudness", "liveness", "valence", "tempo",
    ]

    merged = pd.merge(
        lyrics, audio[keep_cols],
        left_on=["track_title", "album"],
        right_on=["track_name_formatted", "album_name"],
        how="inner",
    )

    # Per-line sentiment polarity via TextBlob (-1 negative to +1 positive).
    # This is the only step that touches raw lyric text — the result
    # (a float per line) is what gets aggregated and kept; the text itself
    # is discarded after this line.
    merged["sentiment_score"] = merged["lyric"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )

    song_level = merged.groupby("track_name_formatted").agg({
        "sentiment_score": "mean",
        "danceability": "first",
        "energy": "first",
        "loudness": "first",
        "liveness": "first",
        "valence": "first",
        "tempo": "first",
        "album_name": "first",
        "year": "first",
    }).reset_index()

    song_level["valence_energy_avg"] = (song_level["valence"] + song_level["energy"]) / 2
    song_level["irony_score"] = song_level["valence_energy_avg"] - song_level["sentiment_score"]

    song_level.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(song_level)} songs to {OUTPUT_PATH}")
    print(f"No lyric text included — {list(song_level.columns)}")


if __name__ == "__main__":
    main()
