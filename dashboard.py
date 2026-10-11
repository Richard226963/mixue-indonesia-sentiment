import os

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

st.set_page_config(page_title="Analisis Sentimen Mixue Indonesia", layout="wide")

ORDER = ["positive", "neutral", "negative"]
COLORS = {"positive": "#2e9e5b", "neutral": "#9a9a9a", "negative": "#d64545"}
DATA_FILES = ["results.csv", "clean.csv", "metrics.csv", "aspect_summary.csv",
              "news_articles.csv", "news_comments.csv"]

RUN_STEPS = """python scrape.py      # instagram, login sendiri
python news.py        # suara.com
python clean.py       # preprocessing + filter bot
python sentiment.py   # labeling indoBERT + evaluasi
python absa.py
python topics.py
python analytics.py"""


def read_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def sidebar_status():
    st.sidebar.subheader("status data")
    for f in DATA_FILES:
        ada = os.path.exists(f"data/{f}")
        st.sidebar.write(f"{'ada ' if ada else 'belum'} - data/{f}")
    st.sidebar.caption("jalankan pipeline dulu kalau masih ada yang belum")


def trend_frame(df):
    dates = pd.to_datetime(df["date"], errors="coerce", format="mixed", utc=True)
    dates = dates.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
    rule = "D"
    for r in ("Y", "M", "W", "D"):
        if dates.dt.to_period(r).dropna().nunique() > 1:
            rule = r
            break
    return df.assign(p=dates.dt.to_period(rule).astype(str)) \
             .groupby(["p", "label"]).size().unstack(fill_value=0) \
             .reindex(columns=ORDER, fill_value=0)


def page_dashboard():
    st.title("Analisis Sentimen Mixue Indonesia")
    st.caption("komentar instagram (akun resmi + hashtag) dan portal berita suara.com, "
               "difilter dari bot lalu dilabeli indoBERT secara lokal")

    df = read_csv("data/results.csv")
    if df is None:
        st.warning("belum ada data/results.csv - jalankan pipeline dulu:")
        st.code(RUN_STEPS, language="bash")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("komentar organik", f"{len(df):,}")
    c2.metric("positif", f"{(df['label'] == 'positive').mean():.0%}")
    c3.metric("negatif", f"{(df['label'] == 'negative').mean():.0%}")
    clean = read_csv("data/clean.csv")
    bots = int(clean["is_bot"].sum()) if clean is not None and "is_bot" in clean else 0
    c4.metric("bot/spam difilter", f"{bots:,}")

    left, right = st.columns(2)
    with left:
        st.subheader("distribusi sentimen")
        counts = df["label"].value_counts().reindex(ORDER).fillna(0)
        st.bar_chart(counts)
    with right:
        st.subheader("instagram vs portal berita")
        share = df.groupby(["source", "label"]).size().unstack(fill_value=0) \
                  .reindex(columns=ORDER, fill_value=0)
        share = share.div(share.sum(axis=1), axis=0)
        st.bar_chart(share)

    st.subheader("tren sentimen")
    counts_t = trend_frame(df)
    if len(counts_t) > 1:
        fig, ax = plt.subplots(figsize=(9.5, 3.4), constrained_layout=True)
        counts_t.plot(kind="bar", stacked=True, ax=ax, color=[COLORS[c] for c in ORDER])
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8, rotation=45)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.caption("butuh komentar dari beberapa periode untuk tren")

    st.subheader("kata kunci per label")
    lab = st.selectbox("label", ORDER)
    text = " ".join(df.loc[df["label"] == lab, "tokens"].fillna(""))
    if text.strip():
        cmap = {"positive": "Greens", "neutral": "Greys", "negative": "Reds"}[lab]
        wc = WordCloud(width=900, height=360, background_color="white", collocations=False,
                       colormap=cmap, random_state=7, max_words=80) \
            .generate_from_frequencies(pd.Series(text.split()).value_counts().to_dict())
        fig, ax = plt.subplots(figsize=(9.5, 3.6), constrained_layout=True)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.caption("tidak ada token untuk label ini")

    st.subheader("evaluasi model (bert vs non-bert)")
    metrics = read_csv("data/metrics.csv")
    if metrics is not None:
        st.dataframe(metrics.fillna(""), width="stretch")
        st.caption("svm dilatih pada label indoBERT berconfidence tinggi lalu diuji di test split; "
                   "indoBERT diukur pada test split publik SmSA (ulasan berlabel manusia)")
    else:
        st.info("metrik muncul setelah sentiment.py berjalan dengan cukup komentar berlabel yakin")

    absa = read_csv("data/aspect_summary.csv")
    if absa is not None:
        st.subheader("sentimen per aspek (absa)")
        st.dataframe(absa, width="stretch")

    pipeline_figs = [f for f in ["sentiment_overview.png", "sentiment_trend.png", "wordclouds.png",
                                 "absa.png", "topics.png", "accounts.png", "media.png"]
                     if os.path.exists(f"figures/{f}")]
    if pipeline_figs:
        with st.expander("grafik lain dari pipeline", expanded=False):
            for f in pipeline_figs:
                st.image(f"figures/{f}")

    st.subheader("penjelajah data")
    flab = st.multiselect("filter label", ORDER, default=ORDER)
    view = df[df["label"].isin(flab)] if flab else df
    st.dataframe(view[["date", "source", "author", "label", "confidence", "text"]]
                 .reset_index(drop=True), width="stretch")


sidebar_status()
page_dashboard()
