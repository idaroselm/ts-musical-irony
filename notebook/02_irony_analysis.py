"""
Stage 2 — irony analysis and visualization.

Fully runnable as-is against data/song_level_sentiment.csv (included in
this repo — 149 songs, aggregate sentiment/audio scores only, no lyric
text). Reproduces the analysis: is there a systematic gap between
lyrical sentiment and musical "positivity" (valence + energy) across
Taylor Swift's discography?
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "song_level_sentiment.csv")
OUTPUT_DIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIO_FEATURES = ["danceability", "energy", "loudness", "liveness", "valence", "tempo"]


def main():
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)

    song_level = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(song_level)} songs")
    print(song_level[["sentiment_score"] + AUDIO_FEATURES].describe())

    # ── Sentiment distribution ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(song_level["sentiment_score"], bins=25, color="teal", edgecolor="black")
    axes[0].set_xlabel("Average Lyrical Sentiment Score")
    axes[0].set_ylabel("Number of Songs")
    axes[0].set_title("Distribution of Lyrical Sentiment")

    pos = (song_level["sentiment_score"] > 0.05).mean() * 100
    neu = (song_level["sentiment_score"].abs() <= 0.05).mean() * 100
    neg = (song_level["sentiment_score"] < -0.05).mean() * 100
    axes[1].bar(["Positive", "Neutral", "Negative"], [pos, neu, neg], color=["seagreen", "gray", "firebrick"])
    axes[1].set_ylabel("% of songs")
    axes[1].set_title("Sentiment Category Split")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_sentiment_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Mean sentiment: {song_level['sentiment_score'].mean():.3f} "
          f"({pos:.0f}% positive, {neu:.0f}% neutral, {neg:.0f}% negative)")

    # ── Correlation matrix ──────────────────────────────────────────────
    corr_cols = ["sentiment_score"] + AUDIO_FEATURES
    correlation_matrix = song_level[corr_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Sentiment vs. Audio Feature Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Irony score ──────────────────────────────────────────────────────
    song_level["valence_energy_avg"] = (song_level["valence"] + song_level["energy"]) / 2
    song_level["irony_score"] = song_level["valence_energy_avg"] - song_level["sentiment_score"]

    print(f"\nMean Irony Score: {song_level['irony_score'].mean():.3f}")
    print(f"Std Dev: {song_level['irony_score'].std():.3f}")

    song_level["irony_level"] = pd.cut(
        song_level["irony_score"],
        bins=[-np.inf, -0.2, 0.2, np.inf],
        labels=["Misaligned: Sad Music, Happy Lyrics", "Aligned: Music Matches Lyrics", "IRONIC: Happy Music, Sad Lyrics"],
    )
    print("\nSongs by category:")
    print(song_level["irony_level"].value_counts())

    # ── Irony visualization ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    scatter = axes[0].scatter(
        song_level["sentiment_score"], song_level["valence_energy_avg"],
        alpha=0.6, s=100, c=song_level["irony_score"], cmap="RdYlGn_r",
    )
    axes[0].set_xlabel("Lyrical Sentiment")
    axes[0].set_ylabel("Musical Positivity (valence + energy) / 2")
    axes[0].set_title("Sentiment vs. Musical Positivity")
    plt.colorbar(scatter, ax=axes[0], label="Irony Score")

    top10 = song_level.nlargest(10, "irony_score")
    axes[1].barh(top10["track_name_formatted"], top10["irony_score"], color="crimson")
    axes[1].set_xlabel("Irony Score")
    axes[1].set_title("Top 10 Most Ironic Songs")
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_irony_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Per-feature correlation with irony ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    for idx, feature in enumerate(AUDIO_FEATURES):
        axes[idx].scatter(song_level["sentiment_score"], song_level[feature], alpha=0.6, s=80)
        z = np.polyfit(song_level["sentiment_score"], song_level[feature], 1)
        p = np.poly1d(z)
        x_line = np.linspace(song_level["sentiment_score"].min(), song_level["sentiment_score"].max(), 100)
        axes[idx].plot(x_line, p(x_line), "r--", linewidth=2.5, label="Trend")
        corr, p_val = pearsonr(song_level["sentiment_score"], song_level[feature])
        tag = "MORE IRONIC" if corr < -0.1 else "ALIGNED" if corr > 0.1 else "NEUTRAL"
        axes[idx].set_title(f"{feature} (r={corr:.2f}, {tag})")
        axes[idx].set_xlabel("Sentiment")
        axes[idx].set_ylabel(feature)
        axes[idx].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_feature_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\n=== CORRELATION ANALYSIS (Irony Detection) ===")
    for feature in AUDIO_FEATURES:
        corr, p_val = pearsonr(song_level["sentiment_score"], song_level[feature])
        direction = "IRONIC" if corr < -0.1 else "ALIGNED" if corr > 0.1 else "no strong pattern"
        print(f"  {feature:<12} r={corr:.3f}  p={p_val:.4f}  ({direction})")

    # ── By album ─────────────────────────────────────────────────────────
    album_irony = song_level.groupby("album_name").agg({
        "irony_score": ["mean", "std", "count"],
        "sentiment_score": "mean",
        "valence_energy_avg": "mean",
    }).round(3)
    album_irony = album_irony.sort_values(("irony_score", "mean"), ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    means = album_irony[("irony_score", "mean")]
    ax.barh(means.index, means.values, color="mediumpurple")
    ax.set_xlabel("Mean Irony Score")
    ax.set_title("Irony Score by Album")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_irony_by_album.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\n=== IRONY BY ALBUM ===")
    print(album_irony)

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    overall_irony = song_level["irony_score"].mean()
    ironic_pct = (song_level["irony_level"] == "IRONIC: Happy Music, Sad Lyrics").mean() * 100
    print(f"Overall average irony score: {overall_irony:.3f}")
    print(f"{ironic_pct:.1f}% of songs pair upbeat production with sadder lyrics")
    print(f"Verdict: {'YES' if overall_irony > 0.1 else 'NO'} — Taylor Swift tends toward musical irony" if overall_irony > 0.1 else "Verdict: lyrics and production are broadly aligned")


if __name__ == "__main__":
    main()
