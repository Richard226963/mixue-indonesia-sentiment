import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

N_TOPICS = 6
N_WORDS = 10


def main():
    try:
        df = pd.read_csv("data/results.csv")
    except FileNotFoundError:
        raise SystemExit("belum ada data/results.csv, jalankan sentiment.py dulu")

    docs = df["tokens"].fillna("")
    docs = docs[docs.str.len() > 0]
    if len(docs) < 50:
        raise SystemExit("data terlalu sedikit untuk topic modeling (<50 komentar)")

    vec = CountVectorizer(max_df=0.9, min_df=3, max_features=3000)
    X = vec.fit_transform(docs)

    lda = LatentDirichletAllocation(n_components=N_TOPICS, max_iter=20,
                                    learning_method="online", random_state=7)
    lda.fit(X)
    names = vec.get_feature_names_out()

    dominant = lda.transform(X).argmax(axis=1)
    sizes = pd.Series(dominant).value_counts().sort_index()

    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4), constrained_layout=True)
    for k, (ax, comp) in enumerate(zip(axes.ravel(), lda.components_)):
        top = comp.argsort()[-N_WORDS:]
        words = [names[i] for i in top]
        weights = comp[top]
        ax.barh(words, weights, color="#3b7dd8")
        ax.set_title(f"topik {k + 1} ({sizes.get(k, 0)} komentar)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[N_TOPICS:]:
        ax.axis("off")
    fig.suptitle("lda topic modeling", fontsize=12)
    fig.savefig("figures/topics.png", dpi=150)
    plt.close(fig)

    print(f"{len(docs)} komentar, {N_TOPICS} topik:")
    for k, comp in enumerate(lda.components_):
        top = comp.argsort()[-N_WORDS:][::-1]
        print(f"  topik {k + 1} ({sizes.get(k, 0)} komentar): " + ", ".join(names[i] for i in top))
    print("\ngrafik di figures/topics.png")


if __name__ == "__main__":
    main()
