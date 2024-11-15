import numpy as np
from tqdm import tqdm

import utils


class Word2Vec:

    def __init__(self, lexic_length, dim, embedding_path, losses_path):
        self.embedding_path = embedding_path
        self.losses_path = losses_path

        self.M = np.random.uniform(
            low=-0.5/dim, high=0.5/dim, size=(lexic_length, dim)
        )
        self.C = np.random.uniform(
            low=-0.5/dim, high=0.5/dim, size=(lexic_length, dim)
        )

    def fit(self, vocab, k, learning_rate=0.1, iter=100):
        positives = vocab.positives
        negatives = vocab.negatives

        losses = []
        for epoch in tqdm(range(iter)):
            loss = 0
            i = 0
            for context_words, ooc_words in zip(positives, negatives):
                c_index = 0
                for c_pos in context_words:
                    # k exemples négatifs pour chaque exemple positif
                    c_negs = ooc_words[c_index:c_index+k]
                    c_index += k

                    # Gradient de c_pos
                    grad_pos = utils.gradient_pos(
                        self.M[i, :], self.C[c_pos, :]
                    )
                    new_c = self.C[c_pos, :] - learning_rate * grad_pos

                    # Gradient de c_neg
                    grads_neg = np.array([
                        utils.gradient_neg(self.M[i, :], self.C[cn])
                        for cn in c_negs
                    ])

                    new_ooc = self.C[c_negs, :] - learning_rate * grads_neg

                    assert len(new_ooc) == k

                    # On a besoin des vecteurs de c pour le calcul de m
                    grad_m = utils.gradient_m(
                        self.M[i, :], self.C[c_pos, :], self.C[c_negs]
                    )
                    self.M[i, :] = self.M[i, :] - learning_rate * grad_m

                    self.C[c_pos, :] = new_c
                    self.C[c_negs, :] = new_ooc

                    loss += utils.loss_function(
                        self.M[i, :], self.C[c_pos, :], self.C[c_negs, :]
                    )
                i += 1

            avg_loss = loss / len(positives)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{iter}, Loss: {avg_loss:.4f}")

        np.save(self.losses_path, losses)
        self.save(vocab, self.M)

    def save(self, vocab, M):
        with open(self.embedding_path, "w") as file:
            n, d = M.shape
            file.write(f"{n} {d}\n")
            for i in range(n):
                word = vocab.get_word_from_index(i)
                vector = ' '.join(map(str, M[i, :]))
                file.write(f"{word}\t{vector}\n")
