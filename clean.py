import glob
import os
import re
import sys

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_.]+")
SPACES_RE = re.compile(r"\s+")
NONWORD_RE = re.compile(r"[^a-z\s]")
REPEAT_RE = re.compile(r"(.)\1{2,}")

# colloquial indonesian, sastrawi does not know half of these
SLANG = {
    "gak": "tidak", "gk": "tidak", "nggak": "tidak", "ngga": "tidak", "ga": "tidak",
    "udh": "sudah", "udah": "sudah", "blm": "belum", "belom": "belum",
    "yg": "yang", "dgn": "dengan", "bgt": "banget", "bangett": "banget",
    "klo": "kalau", "kalo": "kalau", "karna": "karena", "tp": "tapi",
    "jg": "juga", "msh": "masih", "aja": "saja", "trs": "terus", "trus": "terus",
    "emg": "memang", "dpt": "dapat", "bnyk": "banyak", "dr": "dari", "smpe": "sampai",
    "sy": "saya", "aq": "saya", "gw": "saya", "lu": "kamu", "lo": "kamu", "kmu": "kamu",
    "enakeun": "enak", "lemot": "lambat", "lelet": "lambat", "gercep": "cepat",
    "apk": "aplikasi", "org": "orang", "gmn": "gimana", "knp": "kenapa", "bner": "benar",
    "makasih": "terima kasih", "makasi": "terima kasih", "mksh": "terima kasih", "thx": "terima kasih",
    "min": "", "kak": "", "bang": "", "gan": "", "wkwk": "", "haha": "", "hehe": "",
}

# words sastrawi keeps but that are pure filler here
EXTRA_STOP = {"sih", "deh", "dong", "kok", "ya", "nih", "tuh", "the", "a", "an", "and", "or"}

# shop / reseller handles, the usual spam suspects on instagram
SHOP_WORDS = ("shop", "store", "toko", "jasa", "supplier", "agen", "reseller", "official", "kios", "olshop")
PROMO_WORDS = ("jual", "ready stock", "open po", "wa.me", "whatsapp", "hubungi", "cek ig",
               "klik link", "link bio", "kode promo", "gratis ongkir", "cashback", "diskon",
               "daftar sekarang", "garansi", "order sekarang", "buruan order")

_stemmer = StemmerFactory().create_stemmer()
_stopwords = set(StopWordRemoverFactory().get_stop_words()) | EXTRA_STOP
_stem_cache = {}


def stem(word):
    if word not in _stem_cache:
        _stem_cache[word] = _stemmer.stem(word)
    return _stem_cache[word]


def tidy(text):
    # noise removal: urls, mentions, emoji/symbols, repeated letters
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = "".join(ch if ch.isascii() else " " for ch in text)
    text = NONWORD_RE.sub(" ", text.lower())
    text = REPEAT_RE.sub(r"\1", text)
    words = []
    for w in SPACES_RE.sub(" ", text).strip().split():
        w = SLANG.get(w, w)
        for part in w.split():
            if part:
                words.append(part)
    return words


def preprocess(text):
    words = tidy(text)
    cleaned = " ".join(w for w in words if len(w) > 1)
    stems = [stem(w) for w in words if w not in _stopwords]
    tokens = " ".join(s for s in stems if len(s) > 1)
    return cleaned, tokens


def flag_bot(author, text, cleaned, seen_texts):
    # stage 02 of the brief: drop promo/spam accounts, keep organic opinions
    a = str(author).lower()
    if any(w in a for w in SHOP_WORDS) and any(ch.isdigit() for ch in a):
        return "akun toko/jasa"
    if any(ch.isdigit() for ch in a) and len(re.sub(r"\d", "", a)) <= 3:
        return "akun bot (hanya angka)"
    low = text.lower()
    promo_hits = sum(1 for w in PROMO_WORDS if w in low)
    if promo_hits >= 2 or (promo_hits >= 1 and URL_RE.search(text)):
        return "promosi/spam"
    if cleaned == "":
        return "tanpa kata"
    if seen_texts.get((a, cleaned), 0) >= 1:
        return "komentar duplikat"
    seen_texts[(a, cleaned)] = 1
    return ""


def load_ig_comments():
    frames = []
    for f in glob.glob("data/*_comments_*.csv"):
        try:
            df = pd.read_csv(f, dtype=str)
        except Exception:
            continue
        if "shortcode" in df.columns and "comment" in df.columns:
            frames.append(df[["username", "comment", "likes", "date", "comment_id"]])
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates("comment_id")
    df["source"] = "instagram"
    return df


def load_news_comments():
    f = "data/news_comments.csv"
    if not os.path.exists(f):
        return pd.DataFrame()
    df = pd.read_csv(f, dtype=str).drop_duplicates("comment_id")
    df["source"] = "news"
    return df


def main():
    ig = load_ig_comments()
    news = load_news_comments()
    if ig.empty and news.empty:
        sys.exit("belum ada data komentar, jalankan scrape.py dan/atau news.py dulu")

    rows = []
    for df in (ig, news):
        if df.empty:
            continue
        d = df.rename(columns={"username": "author", "comment": "text"}) \
              .dropna(subset=["text"]).reset_index(drop=True)
        d["text"] = d["text"].astype(str)
        rows.append(d[["source", "author", "text", "likes", "date"]])
    df = pd.concat(rows, ignore_index=True)

    before = len(df)
    texts = df["text"].map(preprocess)
    df["text_clean"] = texts.map(lambda t: t[0])
    df["tokens"] = texts.map(lambda t: t[1])

    seen = {}
    flags = [flag_bot(a, t, c, seen) for a, t, c in zip(df["author"], df["text"], df["text_clean"])]
    df["is_bot"] = [f != "" for f in flags]
    df["bot_reason"] = flags

    # timestamps are optional per the brief; undated comments still get labeled,
    # the trend chart simply skips them
    df.to_csv("data/clean.csv", index=False)

    print(f"{before} -> {len(df)} komentar di data/clean.csv")
    for source, n in df.groupby("source").size().items():
        bots = int((df[df["source"] == source]["is_bot"]).sum())
        print(f"  {source}: {n} baris, {bots} terdeteksi bot/spam ({bots / n:.0%})")


if __name__ == "__main__":
    main()
