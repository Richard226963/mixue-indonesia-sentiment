import os

import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["positive", "neutral", "negative"]
COLORS = {"positive": "#2e9e5b", "neutral": "#9a9a9a", "negative": "#d64545"}

# aspect keyword lists, matched on the raw lowercase text so negation stays visible
ASPECTS = {
    "harga": ["harga", "murah", "mahal", "murmer", "goceng", "pricelist", "5rb", "8rb", "10rb", "seribu"],
    "rasa & menu": ["enak", "hambar", "manis", "pahit", "gurih", "rasa", "menu", "matcha",
                    "boba", "vanila", "latte", "coklat", "es krim", "ice cream", "seger", "creamy"],
    "pelayanan": ["pelayanan", "staff", "kasir", "ramah", "servis", "karyawan", "antri", "antre",
                  "antrean", "lama", "lelet", "cepat", "waiting"],
    "tempat": ["tempat", "suasana", "gerai", "outlet", "bersih", "berantakan", "sempit", "luas",
               "parkir", "lokasi", "papan", "kursi"],
}


def main():
    try:
        df = pd.read_csv("data/results.csv")
    except FileNotFoundError:
        raise SystemExit("belum ada data/results.csv, jalankan sentiment.py dulu")

    rows = []
    text = df["text"].fillna("").str.lower()
    for aspect, keywords in ASPECTS.items():
        mask = text.map(lambda t: any(k in t for k in keywords))
        sub = df[mask]
        if sub.empty:
            continue
        counts = sub["label"].value_counts().reindex(ORDER).fillna(0).astype(int)
        rows.append({"aspek": aspect, "total": len(sub), **{c: int(counts[c]) for c in ORDER}})
        for c in ORDER:
            df.loc[mask, f"aspek_{c}"] = df.loc[mask, "label"].eq(c)

    if not rows:
        raise SystemExit("tidak ada komentar yang menyebut aspek manapun")

    summary = pd.DataFrame(rows)
    summary.to_csv("data/aspect_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    y = range(len(summary))
    left = pd.Series(0, index=summary.index, dtype=float)
    for c in ORDER:
        share = summary[c] / summary["total"] * 100
        ax.barh(list(y), share, left=left, color=COLORS[c], label=c)
        left += share
    ax.set_yticks(list(y))
    ax.set_yticklabels(summary["aspek"] + "  (n=" + summary["total"].astype(str) + ")", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% komentar")
    ax.set_title("sentimen per aspek (absa)")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig("figures/absa.png", dpi=150)
    plt.close(fig)

    print(summary.to_string(index=False))
    print("\nringkasan aspek di data/aspect_summary.csv, grafik di figures/absa.png")


if __name__ == "__main__":
    main()
