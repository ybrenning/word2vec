import json

import numpy as np

from vocab import Vocab
from w2v import Word2Vec
from eval_w2v import Evaluate


class Experiment:

    def __init__(self, params):
        print("Running experiment with config:", end="\n\n")
        print(json.dumps(params, indent=4), end="\n\n")

        self.file_path = params["file_path"]
        self.eval_path = params["eval_path"]
        self.embedding_path = params["embedding_path"]
        self.losses_path = params["losses_path"]
        self.scores_path = params["scores_path"]
        self.results_path = params["results_path"]

        self.n_runs = params["n_runs"]
        self.d = params["d"]
        self.L = params["L"]
        self.k = params["k"]
        self.eta = params["eta"]
        self.n_epochs = params["n_epochs"]
        self.minc = params["minc"]

        self.unigram = params["unigram"]
        self.subsampling = params["subsampling"]

    def run(self):
        accs = []
        for _ in range(self.n_runs):
            self.vocab = Vocab(
                self.file_path,
                self.L,
                self.k,
                self.minc,
                self.unigram,
                self.subsampling
            )

            w2v = Word2Vec(
                len(self.vocab.lexic),
                self.d,
                self.embedding_path,
                self.losses_path
            )
            w2v.fit(self.vocab, self.k, self.eta, self.n_epochs)

            eval = Evaluate(
                self.embedding_path, self.eval_path, self.results_path
            )
            accs.append(eval.accuracy)

        np.save(f"{self.scores_path}", accs)
        print("Mean accuracy:", np.mean(np.array(accs)))


baseline_params = {
    # Fichiers
    "file_path": "data/Le_comte_de_Monte_Cristo.txt",
    "eval_path": "data/Le_comte_de_Monte_Cristo.100.sim",

    "embedding_path": "embeddings/embeddings-baseline-d.txt",
    "losses_path": "results/losses-baseline-d.npy",
    "scores_path": "scores/scores-baseline-d.npy",
    "results_path": "results/results-baseline-d.npy",

    # Hyperparamètres
    "n_runs": 10,
    "d": 100,
    "L": 2,
    "k": 10,
    "eta": 0.1,
    "n_epochs": 5,
    "minc": 5,

    "unigram": False,
    "subsampling": False,
}

if __name__ == '__main__':
    experiment = Experiment(baseline_params)
    experiment.run()
