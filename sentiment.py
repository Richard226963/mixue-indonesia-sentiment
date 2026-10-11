import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from transformers import pipeline
from wordcloud import WordCloud

MODEL = "w11wo/indonesian-roberta-base-sentiment-classifier"
BATCH = 32
PSEUDO_CONFIDENCE = 0.85  # only confident bert labels train the classical model
SMSA_TEST = ("https://raw.githubusercontent.com/IndoNLP/indonlu/master/"
             "dataset/smsa_doc-sentiment-prosa/test_preprocess.tsv")

ORDER = ["positive", "neutral", "negative"]
COLORS = {"positive": "#2e9e5b", "neutral": "#9a9a9a", "negative": "#d64545"}
SOURCE_NAMES = {"instagram": "Instagram", "news": "Portal berita (suara.com)"}


def bert_labels(texts, clf):
    out = clf(texts, batch_size=BATCH, truncation=True, max_length=512)
    return [r["label"].lower() for r in out], [round(r["score"], 4) for r in out]


def classical_comparison(df):
    # the brief wants bert vs classical on the same data. there are no hand labels,
    # so the classical model is trained on the confident part of the bert labels and
    # both get compared on a held out split. it measures how well tf-idf+svm can
    # approximate the transformer, not human accuracy.
    work = df[df["confidence"] >= PSEUDO_CONFIDENCE].copy()
    counts = work["label"].value_counts()
    if len(work) < 80 or counts.min() < 10:
        print(f"data berlabel yakin belum cukup ({len(work)} baris), perbandingan klasik dilewati")
        return []
    print(f"\nperbandingan bert vs klasik (tf-idf + svm) pada {len(work)} komentar berlabel yakin:")

    Xtr, Xte, ytr, yte = train_test_split(work["text_clean"], work["label"],
                                          test_size=0.2, stratify=work["label"], random_state=42)
    svm = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True),
        LinearSVC(class_weight="balanced", max_iter=5000),
    )
    svm.fit(Xtr, ytr)
    pred = svm.predict(Xte)
    acc = accuracy_score(yte, pred)

    rows = []
    print(f"  akurasi  svm: {acc:.3f}")
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, labels=ORDER, zero_division=0)
    for i, lab in enumerate(ORDER):
        print(f"  svm {lab:>8}: precision {p[i]:.3f} | recall {r[i]:.3f} | f1 {f1[i]:.3f}")
        rows.append({"model": "svm (tf-idf)", "kelas": lab,
                     "precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                     "f1": round(float(f1[i]), 3)})
    print(f"  svm macro: precision {p.mean():.3f} | recall {r.mean():.3f} | f1 {f1.mean():.3f}")
    rows.append({"model": "svm (tf-idf)", "kelas": "macro",
                 "precision": round(float(p.mean()), 3), "recall": round(float(r.mean()), 3),
                 "f1": round(float(f1.mean()), 3)})
    rows.append({"model": "svm (tf-idf)", "kelas": "semua (test split)",
                 "precision": round(float(acc), 3), "recall": round(float(acc), 3),
                 "f1": round(float(acc), 3)})
    kappa = cohen_kappa_score(df["label"], svm.predict(df["text_clean"]))
    print(f"  kappa kesesuaian svm vs bert di seluruh data: {kappa:.2f}")
    rows.append({"model": "svm vs bert", "kelas": "kappa kesesuaian",
                 "precision": round(float(kappa), 2), "recall": "", "f1": ""})
    return rows


def benchmark(clf):
    # the brief wants metrics on the bert side too. our scraped comments have no
    # hand labels, so the pretrained model is scored on the public smssa test
    # split instead: 500 indonesian reviews labeled by human annotators, kept out
    # of the model's own fine-tuning. benchmark numbers, not accuracy on our data.
    cache = "data/smsa_test.tsv"
    if not os.path.exists(cache):
        try:
            resp = requests.get(SMSA_TEST, timeout=30)
            resp.raise_for_status()
            os.makedirs("data", exist_ok=True)
            with open(cache, "w", encoding="utf-8") as f:
                f.write(resp.text)
        except Exception:
            print("\nbenchmark smssa dilewati (tidak bisa mengunduh data uji)")
            return []
    with open(cache, encoding="utf-8") as f:
        pairs = [line.rsplit("\t", 1) for line in f.read().splitlines() if "\t" in line]
    if len(pairs) < 100:
        print("\nbenchmark smssa dilewati (isi data uji tidak wajar)")
        return []
    texts = [t for t, _ in pairs]
    gold = [lab.strip() for _, lab in pairs]
    pred = [x["label"].lower() for x in clf(texts, batch_size=BATCH,
                                            truncation=True, max_length=512)]
    acc = accuracy_score(gold, pred)
    rows = []
    print(f"\nbenchmark indoBERT pada test split publik SmSA ({len(gold)} ulasan berlabel manusia):")
    print(f"  akurasi bert: {acc:.3f}")
    p, r, f1, _ = precision_recall_fscore_support(gold, pred, labels=ORDER, zero_division=0)
    for i, lab in enumerate(ORDER):
        print(f"  bert {lab:>8}: precision {p[i]:.3f} | recall {r[i]:.3f} | f1 {f1[i]:.3f}")
        rows.append({"model": "indoBERT (pre-trained)", "kelas": lab,
                     "precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                     "f1": round(float(f1[i]), 3)})
    print(f"  bert macro: precision {p.mean():.3f} | recall {r.mean():.3f} | f1 {f1.mean():.3f}")
    rows.append({"model": "indoBERT (pre-trained)", "kelas": "macro",
                 "precision": round(float(p.mean()), 3), "recall": round(float(r.mean()), 3),
                 "f1": round(float(f1.mean()), 3)})
    rows.append({"model": "indoBERT (pre-trained)", "kelas": "semua (smssa test)",
                 "precision": round(float(acc), 3), "recall": round(float(acc), 3),
                 "f1": round(float(acc), 3)})
    return rows


def overview_figure(df):
    counts = df["label"].value_counts().reindex(ORDER).fillna(0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    ax1.bar(ORDER, counts.values, color=[COLORS[c] for c in ORDER])
    for i, v in enumerate(counts.values):
        ax1.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    ax1.set_title("jumlah komentar")
    ax2.pie(counts.values, labels=ORDER, autopct="%1.1f%%",
            colors=[COLORS[c] for c in ORDER], wedgeprops=dict(width=0.45))
    ax2.set_title("proporsi")
    fig.savefig("figures/sentiment_overview.png", dpi=150)
    plt.close(fig)


def source_figure(df):
    sources = [s for s in df["source"].dropna().unique() if s in SOURCE_NAMES]
    if len(sources) < 2:
        return
    share = df.groupby(["source", "label"]).size().unstack(fill_value=0)
    share = share.reindex(index=[s for s in sources], columns=ORDER, fill_value=0)
    share = share.div(share.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    x = range(len(sources))
    width = 0.25
    for i, lab in enumerate(ORDER):
        vals = [share.loc[s, lab] for s in sources]
        ax.bar([xi + (i - 1) * width for xi in x], vals, width,
               color=COLORS[lab], label=lab)
    ax.set_xticks(list(x))
    ax.set_xticklabels([SOURCE_NAMES[s] for s in sources], fontsize=9)
    ax.set_ylabel("% komentar")
    ax.set_title("rasio sentimen instagram vs portal berita")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig("figures/sentiment_by_source.png", dpi=150)
    plt.close(fig)


def trend_figure(df):
    dates = pd.to_datetime(df["date"], errors="coerce", format="mixed", utc=True)
    dates = dates.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
    rule = "D"
    for r in ("Y", "M", "W", "D"):
        if dates.dt.to_period(r).dropna().nunique() > 1:
            rule = r
            break
    counts = df.assign(p=dates.dt.to_period(rule).astype(str)) \
               .groupby(["p", "label"]).size().unstack(fill_value=0) \
               .reindex(columns=ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    counts.plot(kind="bar", stacked=True, ax=ax, color=[COLORS[c] for c in ORDER])
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    fig.savefig("figures/sentiment_trend.png", dpi=150)
    plt.close(fig)


def wordcloud_figure(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, lab in zip(axes, ORDER):
        text = " ".join(df.loc[df["label"] == lab, "tokens"].fillna(""))
        if not text.strip():
            ax.axis("off")
            continue
        cmap = {"positive": "Greens", "neutral": "Greys", "negative": "Reds"}[lab]
        wc = WordCloud(width=640, height=400, background_color="white", collocations=False,
                       colormap=cmap, random_state=7, max_words=80).generate_from_frequencies(
                           pd.Series(text.split()).value_counts().to_dict())
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(lab)
    fig.savefig("figures/wordclouds.png", dpi=150)
    plt.close(fig)


def main():
    try:
        df = pd.read_csv("data/clean.csv", dtype={"comment_id": str})
    except FileNotFoundError:
        raise SystemExit("belum ada data/clean.csv, jalankan clean.py dulu")

    df = df[~df["is_bot"].astype(bool)].reset_index(drop=True)
    df = df[df["text_clean"].fillna("").str.len() > 0]
    if df.empty:
        raise SystemExit("tidak ada komentar organik setelah filter bot")

    os.makedirs("figures", exist_ok=True)
    clf = pipeline("text-classification", model=MODEL)
    print(f"{len(df)} komentar organik, pelabelan dengan {MODEL}")
    df["label"], df["confidence"] = bert_labels(df["text"].tolist(), clf)

    df.to_csv("data/results.csv", index=False)

    print("\ndistribusi sentimen:")
    print(df["label"].value_counts().reindex(ORDER).fillna(0).astype(int).to_string())

    metrics = classical_comparison(df) + benchmark(clf)
    if metrics:
        pd.DataFrame(metrics).to_csv("data/metrics.csv", index=False)
        print("\ntabel metrik evaluasi di data/metrics.csv")

    overview_figure(df)
    source_figure(df)
    trend_figure(df)
    wordcloud_figure(df)
    print("\nhasil lengkap di data/results.csv, grafik di figures/")


if __name__ == "__main__":
    main()
