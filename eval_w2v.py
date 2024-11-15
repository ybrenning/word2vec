import numpy as np


class Evaluate:

    def __init__(self, embedding_file_path, test_file_path, results_path):
        self.embedding_file = open(embedding_file_path, 'r')
        self.test_file = open(test_file_path, 'r')

        self.results_path = results_path

        self.load_embedding()
        self.evaluate()

    def load_embedding(self):
        self.E = {}
        for line in self.embedding_file:
            word, *vec = line.split()
            self.E[word] = np.array(vec, dtype=float)

    def evaluate(self):
        with open(self.results_path, 'w') as file:
            counter = 0
            i = 0
            for line in self.test_file:
                i += 1
                word1, word2, word3 = line.split()

                # Ignorer les mots qui n'ont pas d'embedding
                if set([word1, word2, word3]).issubset(self.E.keys()):
                    if self.similarity(word1, word2) > self.similarity(word1, word3):
                        file.write(f"{word1} {word2} {word3} 1\n")
                        counter += 1
                    else:
                        file.write(f"{word1} {word2} {word3} 0\n")

        self.accuracy = counter / i
        print(f"Accuracy: {counter / i}")

    def similarity(self, word1, word2):
        """
        Calculer la similarité cosinus:
        cos(a, b) = (a @ b) / (||a|| * ||b||)
        """
        num = np.dot(self.E[word1], self.E[word2])
        denom = np.linalg.norm(self.E[word1]) * np.linalg.norm(self.E[word2])

        return num / denom
