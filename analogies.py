import time
import gc
import pickle
import numpy as np

from tqdm import tqdm


RAM_SIZE = 100
CHUNK_SIZE = 1000
DIM = 100


def analogy_query(kdtree, w2v, word_a, word_b, word_c):

    exclude_w = [word_a, word_b, word_c]
    exclude_e = [w2v[w] for w in exclude_w]

    print(word_b, "-", word_a, "+", word_c)
    analogy_vector = np.array(w2v[word_b]) - np.array(w2v[word_a]) + np.array(w2v[word_c])

    if not kdtree:
        result = None
        min = np.inf
        for w, e in tqdm(w2v.items()):
            if w in exclude_w:
                continue

            dist = np.linalg.norm(e - analogy_vector)
            if dist < min:
                min = dist
                result = e

        return result
    elif kdtree.__class__.__module__ == "kdtree":
        return kdtree.nearest_neighbor(analogy_vector, exclude_e)[0]
    elif kdtree.__class__.__module__ == "sklearn.neighbors._kd_tree":
        embeddings = list(w2v.values())
        dist, ind = kdtree.query(analogy_vector.reshape(1, -1), k=4)
        for i in ind[0]:
            if embeddings[i] not in exclude_e:
                result = embeddings[i]
                break

        del embeddings
        gc.collect()

        return result


def read_file(filename, encoding="latin-1"):

    with open(filename, "r", encoding=encoding) as fin:
        next(fin)
        n = 1
        while True:
            word_embeddings = {}
            for _ in range(RAM_SIZE):
                for _ in range(CHUNK_SIZE):
                    line = fin.readline()
                    if line == "":
                        with open(f"data/embeddings-{n}.pkl", "wb") as fout:
                            pickle.dump(word_embeddings, fout)
                        print("Clearing memory")
                        del word_embeddings
                        gc.collect()

                        return

                    content = line.strip().split(" ")
                    try:
                        w = content[0]
                        w = w.encode("latin-1").decode("utf-8")
                    except UnicodeError as ue:
                        print(f"{ue} encountered on word: {w}")
                        continue

                    e = [float(c) for c in content[1:]]
                    d = len(e)

                    if d == DIM:
                        word_embeddings[w] = e
                    else:
                        print(
                            f"Embedding dimension missing on word {w}: "
                            f"expected {DIM}, actual: {d}"
                        )

            print(f"Dumping to: embeddings-{n}.pkl")
            with open(f"embeddings-{n}.pkl", "wb") as fout:
                pickle.dump(word_embeddings, fout)

            print("Clearing memory")
            del word_embeddings
            gc.collect()
            n += 1


def eval(eval_file, kdtree, w2v, v2w):

    with open(eval_file, "r") as f:
        correct = 0
        total = 0
        times = []
        for line in f:
            line = line.strip().split(" ")
            word_a = line[0]
            word_b = line[1]
            word_c = line[2]
            y_true = line[3]

            words = w2v.keys()
            if any(w not in words for w in line):
                print(f"Missing words: {[w for w in line if w not in words]}")
                continue

            start_time = time.time()
            result = tuple(
                analogy_query(kdtree, w2v, word_a, word_b, word_c)
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)

            print(f"Time taken: {elapsed_time:.6f} seconds")
            print(f"Predicted: {v2w[result]}, expected: {y_true}")

            if v2w[result] == y_true:
                correct += 1
            total += 1

    acc = correct / total
    return acc, np.array(times)


def build_kdtree(w2v, implementation=None):

    if not implementation:
        return None

    embeddings = np.array(list(w2v.values()))
    if implementation == "local":
        from kdtree import KDTree as MyKDTree
        kdtree = MyKDTree(embeddings)
    elif implementation == "sklearn":
        from sklearn.neighbors import KDTree as SklKDTree
        kdtree = SklKDTree(embeddings)

    del embeddings
    gc.collect()

    return kdtree


def run_experiment(embeddings_file, eval_file):

    with open(embeddings_file, "rb") as f:
        w2v = pickle.load(f)

    v2w = {tuple(vec): word for word, vec in w2v.items()}

    accs = []
    for implementation in [None, "local", "sklearn"]:
        if implementation:
            times_file = f"times/times-{implementation}.npy"
        else:
            times_file = "times/times-exhaustive.npy"

        kdtree = build_kdtree(w2v, implementation=implementation)

        acc, times = eval(eval_file, kdtree, w2v, v2w)

        print("Accuracy:", acc)
        print(f"Saving times to {times_file}")

        accs.append(acc)
        np.save(times_file, times)

    assert len(set(accs)) == 1


def main():
    # read_file("data/model.txt")
    run_experiment("data/embeddings-1.pkl", "data/analogies.txt")


if __name__ == "__main__":
    main()
