import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

SOURCE_NAMES = {"instagram": "Instagram", "news": "Portal berita (suara.com)"}
ORDER = ["positive", "neutral", "negative"]
COLORS = {"positive": "#2e9e5b", "neutral": "#9a9a9a", "negative": "#d64545"}


def load_posts():
    frames = []
    for f in glob.glob("data/mixue*_*.csv"):
        if "_comments_" in f:
            continue
        try:
            df = pd.read_csv(f, dtype={"shortcode": str})
        except Exception:
            continue
        if "shortcode" in df.columns and "caption" in df.columns:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates("shortcode")
    df["posted_at_utc"] = pd.to_datetime(df["posted_at_utc"], errors="coerce", utc=True)
    return df


def accounts_figure(posts, results):
    if posts.empty:
        print("belum ada data postingan instagram, analisis akun dilewati")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    top_posts = posts.nlargest(min(10, len(posts)), "comments")[["username", "comments"]]
    axes[0].barh([f"@{u}" for u in top_posts["username"]][::-1], top_posts["comments"][::-1],
                 color="#3b7dd8")
    axes[0].set_title("postingan dengan komentar terbanyak")
    axes[0].tick_params(labelsize=8)
    axes[0].set_xlabel("jumlah komentar")
    axes[0].spines[["top", "right"]].set_visible(False)

    ig = results[results["source"] == "instagram"]
    if ig.empty:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "belum ada komentar ig,\njalankan scrape.py dulu",
                     ha="center", va="center", fontsize=10)
    else:
        kol = ig["author"].value_counts().head(10)
        axes[1].barh([f"@{a}" for a in kol.index][::-1], kol.values[::-1], color="#2e9e5b")
        axes[1].set_title("komentator paling aktif (kandidat kol)")
        axes[1].tick_params(labelsize=8)
        axes[1].set_xlabel("jumlah komentar")
        axes[1].spines[["top", "right"]].set_visible(False)

    fig.savefig("figures/accounts.png", dpi=150)
    plt.close(fig)

    print("akun dengan komentar terbanyak di postingannya:")
    print(posts.groupby("username")["comments"].sum().sort_values(ascending=False)
          .head(5).to_string())


def media_figure(posts, results):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)

    try:
        arts = pd.read_csv("data/news_articles.csv")
        dates = pd.to_datetime(arts["date"], errors="coerce", format="mixed", utc=True)
        per_month = dates.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None).dt.to_period("M") \
            .value_counts().sort_index()
        per_month.index = per_month.index.astype(str)
        axes[0].bar(per_month.index, per_month.values, color="#3b7dd8")
        axes[0].set_title(f"volume artikel mixue di suara.com ({len(arts)} artikel)")
        axes[0].tick_params(axis="x", labelsize=7, rotation=45)
        axes[0].spines[["top", "right"]].set_visible(False)
    except FileNotFoundError:
        axes[0].axis("off")
        print("belum ada data/news_articles.csv, jalankan news.py dulu")

    share = results.groupby(["source", "label"]).size().unstack(fill_value=0) \
        .reindex(columns=ORDER, fill_value=0)
    share = share.div(share.sum(axis=1), axis=0) * 100
    sources = [s for s in share.index if s in SOURCE_NAMES]
    x = range(len(sources))
    width = 0.25
    for i, lab in enumerate(ORDER):
        axes[1].bar([xi + (i - 1) * width for xi in x],
                    [share.loc[s, lab] for s in sources], width,
                    color=COLORS[lab], label=lab)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([SOURCE_NAMES[s] for s in sources], fontsize=9)
    axes[1].set_ylabel("% komentar")
    axes[1].set_title("sentimen: instagram vs portal berita")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.savefig("figures/media.png", dpi=150)
    plt.close(fig)

    if len(sources) == 2:
        for s in sources:
            pos = share.loc[s, "positive"]
            print(f"rasio positif {SOURCE_NAMES[s]}: {pos:.0f}%")


def main():
    try:
        results = pd.read_csv("data/results.csv")
    except FileNotFoundError:
        raise SystemExit("belum ada data/results.csv, jalankan sentiment.py dulu")

    os.makedirs("figures", exist_ok=True)
    posts = load_posts()

    print("=== analisis akun (instagram) ===")
    accounts_figure(posts, results)
    print("\n=== analisis media (portal berita) ===")
    media_figure(posts, results)
    print("\ngrafik di figures/accounts.png dan figures/media.png")


if __name__ == "__main__":
    main()
