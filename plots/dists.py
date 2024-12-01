import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import vocab


config = {
    "file_path": "data/Le_comte_de_Monte_Cristo.txt",
    "L": 2,
    "k": 10,
    "minc": 5,
    "unigram": False,
    "subsampling": False,
}

with open(config["file_path"], 'r') as file:
    content = file.read()


content = content.lower() \
    .replace('.', ' </s>') \
    .replace('-', ' ') \
    .replace('!', '') \
    .replace('?', '') \
    .replace(':', '') \
    .replace(';', '') \
    .replace(',', '') \
    .replace('\n', ' ') \
    .replace('«', '') \
    .replace('»', '') \
    .replace('—', ' ') \
    .replace('...', '') \
    .replace('…', '') \
    .replace('(', '') \
    .replace(')', '')

corpus = [w for w in content.split(' ') if w != '']

lexic = {}
counter = Counter(corpus)
i = 0
for word, count in counter.items():
    if count >= config["minc"] and word != "</s>":
        lexic[word] = i, count
        i += 1

tuples = [(k, v) for k, v in lexic.items()]
tuples.sort(key=lambda t: t[1][1], reverse=True)

counts = np.array([t[1][1] for t in tuples])
smoothed_counts = counts**0.75
denom = smoothed_counts.sum()
unigram = smoothed_counts / denom

default = counts / counts.sum()
print(unigram)

assert len(unigram) == len(lexic)

n = np.arange(len(unigram))

plt.figure(figsize=(5, 4))
plt.fill_between(n, default, color="#E24A33", alpha=0.3, label="$U(w)$")
plt.plot(n, default, color="#E24A33", lw=2)

plt.fill_between(n, unigram, color="#348ABD", alpha=0.3, label="$U(w)^{3/4}$")
plt.plot(n, unigram, color="#348ABD", lw=2)
plt.xlim(xmin=0, xmax=len(n))

words = [t[0] for t in tuples]
plt.yscale("log")
plt.legend()
# plt.bar(words, unigram)
# plt.xticks(words)
plt.xlabel("Word index")
plt.ylabel("Probability")
plt.tight_layout()
plt.savefig("unigram.png", dpi=300, bbox_inches="tight")
plt.show()
