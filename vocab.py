from collections import Counter
import random
import numpy as np


class Vocab:

    def __init__(self, filepath, L, k, minc, unigram, subsampling):
        with open(filepath, 'r') as file:
            content = file.read()

        self.minc = minc
        self.corpus = self.compute_corpus(content)

        self.lexic = self.compute_indexed_lexic(self.corpus, self.minc)

        t = 1e-3 if subsampling else 0
        self.filter_corpus(self.corpus, self.lexic, t)

        self.lexic = self.compute_indexed_lexic(self.corpus, self.minc)

        self.positives = self.compute_positives(
            self.corpus, self.lexic, L
        )
        self.negatives = self.compute_negatives(
            self.positives, k, unigram
        )

    def compute_corpus(self, content):
        """
        Parameters:
            content (str): contenu d'un fichier de texte

        Returns:
            mots dans le texte sans ponctuation, majuscules, etc.
        """
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

        # TODO: Comment gérer les apostrophes?
        # par ex. 'obscurité' et 'l'obscurité'

        # Il faut ignorer les résultats vides
        words = [w for w in content.split(' ') if w != '']

        return words

    def compute_indexed_lexic(self, corpus, minc):
        """
        Construit la "lexique"/"vocabulaire" d'un texte

        Returns:
            dictionnaire avec le format {'<mot>': (<id>, <n_apparances>)}

        Ex.:
            {
                'the': (1, 2),
                'quick': (2, 1),
                'brown': (3, 1),
                'fox': (4, 1),
                ...
            }
        """
        lexic = {}
        counter = Counter(corpus)
        i = 0
        for word, count in counter.items():
            if count >= minc and word != "</s>":
                lexic[word] = i, count
                i += 1

        return lexic

    def filter_corpus(self, corpus, lexic, t):
        new = []
        for i, w in enumerate(corpus):
            # Ignorer les mots rares
            if w in lexic:
                id, count = lexic[w]

                # Subsampling actif
                if t > 0:
                    z = count / len(corpus)
                    p = (np.sqrt(z/t) + 1) * (t/z)
                    # p = 1 - np.sqrt(t / z)

                    p = min(p, 1)
                    p = max(0, p)

                    # Right val is prob of entering the block
                    if random.random() < p:
                        new.append(w)
                else:
                    new.append(w)
            elif w == "</s>":
                new.append(w)

        assert len(new) < len(corpus)

        self.corpus = new

    def compute_positives(self, corpus, lexic, L):
        """
        Trouver les instances positives pour chaque mot cible.
        (c.a.d. les mots qui apparaissent dans le contexte d'un mot cible)

        Si `subsampling` est activé, on lance un mot dans le contexte
        avec la probabilité P(w) = 1 - sqrt(t/f(w)) [1].

        Returns:
            liste de np.array tel que:
            pos[<id>] = [<liste de ids positifs correspondants>]

        À noter:
            L'implementation de Google calcule la probabilité pour
            subsampling un peu différement:

            z = c_count / len(self.corpus)
            p = (np.sqrt(z/t) + 1) * (t/z)

            Alors p est la probabilité de conserver un mot.

        Références:
            [1] https://arxiv.org/pdf/1310.4546
        """
        positives = [[] for _ in range(len(lexic))]

        for i, target_word in enumerate(corpus):
            if target_word == "</s>":
                continue

            assert target_word in lexic

            L = np.random.randint(1, L+1)

            # Trouver le contexte
            lower_index = max(0, i - L)
            higher_index = min(len(corpus) - 1, i + L) + 1
            prev = corpus[lower_index:i]
            next = corpus[i + 1:higher_index]

            # Arrêter le contexte au début/à la fin d'une phrase
            for j, word in enumerate(prev):
                if word == "</s>":
                    prev = prev[j+1:]

            for j, word in enumerate(next):
                if word == "</s>":
                    next = next[:j]
                    break

            context = prev + next

            t_id = lexic[target_word][0]
            for c in context:
                assert c in lexic
                c_id, _ = lexic[c]

                positives[t_id].append(c_id)

        assert len(lexic) == len(positives)

        return positives

    def compute_negatives(self, positives, k, unigram):
        """
        Créer un dictionnaire comme `positives`
        avec `k` mots qui n'apparaissent pas dans le contexte.

        Dans ce cas, on fait l'hypothèse que tirer les `k` IDs
        aléatoirement du lexique suffit.
        """
        if unigram:
            dist = self.create_unigram_dist()
        else:
            dist = None

        negatives = []
        ids = np.arange(len(positives))
        for pos in positives:
            n = np.random.choice(ids, k * len(pos), p=dist, replace=True)

            assert len(n) == (len(pos) * k)

            negatives.append(n)

        assert len(negatives) == len(positives)

        return negatives

    def create_unigram_dist(self):
        r"""
        Probabilité de selectionner un mot w dépend de la fréquence f(w):
        P(w) = f(w) / \sum_{j=0}^n f(w)

        Word2Vec utilise la puissance 3/4:
        P(w) = f(w)^0.75 / \sum_{j=0}^n f(w)^0.75
        """
        counts = np.array([value[1] for value in self.lexic.values()])
        smoothed_counts = counts**0.75
        denom = smoothed_counts.sum()
        probs = smoothed_counts / denom

        assert len(probs) == len(self.lexic)

        return probs

    def get_word_from_index(self, index):
        for k, v in self.lexic.items():
            if v[0] == index:
                return k

        raise IndexError

    def get_index_from_word(self, word):
        return self.lexic[word][0]
