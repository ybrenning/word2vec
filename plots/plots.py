import numpy as np
import matplotlib.pyplot as plt


def plot_p1():
    files = [
        "../scores/scores-baseline-d.npy",
        "../scores/scores-unigram-d.npy",
        "../scores/scores-subsampling-d.npy",
        "../scores/scores-both-d.npy",
    ]

    accs = []
    for f in files:
        acc = np.load(f)
        print(f)
        print(acc)
        print(round(acc.mean(), 3), round(acc.std(), 3))
        accs.append(acc)

    plt.figure(figsize=(7, 4))

    plt.boxplot(
        accs,
        labels=[
            "Baseline",
            "Negative sampling",
            "Subsampling",
            "Negative sampling\n + subsampling",
        ],
        # patch_artist=True,
        # boxprops=dict(facecolor='tab:lightblue', color='tab:blue'),
        medianprops=dict(color='tab:red', linewidth=2)
    )

    plt.xlabel("Version de modèle", fontsize=12)
    plt.ylabel("Taux de réussite", fontsize=12)

    plt.tight_layout()
    plt.savefig("boxplot.png", dpi=300)
    plt.show()


def plot_p2():
    files = [
        "../times/times-exhaustive.npy",
        "../times/times-local.npy",
        # "../times/times-sklearn.npy",
    ]

    times = []
    for f in files:
        t = np.load(f)
        print(f)
        print(t)
        print(round(t.mean(), 3), round(t.std(), 3))
        times.append(t)

    plt.figure(figsize=(7, 4))

    plt.boxplot(
        times,
        labels=[
            "Exhaustive",
            "KDTree",
            # "KDTree (Sklearn)"
        ],
        # patch_artist=True,
        # boxprops=dict(facecolor='tab:lightblue', color='tab:blue'),
        medianprops=dict(color='tab:red', linewidth=2)
    )

    plt.xlabel("Type de recherche", fontsize=12)
    plt.ylabel("Durée d'exécution (en secondes)", fontsize=12)

    plt.tight_layout()
    plt.savefig("times.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_p2()
