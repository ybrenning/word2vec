import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

with open("data/embeddings-1.pkl", "rb") as f:
    w2v = pickle.load(f)

plot_words = [
    # Royalty
    "roi", "reine", "prince", "princesse", "empereur",

    # Pronouns
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",

    # Countries
    "france", "allemagne", "espagne", "italie", "angleterre",

    # Professions
    "professeur", "élève", "docteur", "avocat", "ingénieur",

    # Places/Location
    "ville", "campagne", "montagne", "plage", "forêt", "rue", "parc", "île",

    # Food and Drink
    "pain", "fromage", "vin", "eau", "café", "viande", "poisson", "légume",

    # Nature/Animals
    "chat", "chien", "oiseau", "poisson", "cheval", "lion", "éléphant",

    # Emotions/States
    "bonheur", "tristesse", "colère", "joie", "peur", "amour", "haine",

    # Time/Date
    "jour", "nuit", "semaine", "mois", "année", "heure", "minute", "seconde",

    # Colors
    "rouge", "bleu", "vert", "jaune", "noir", "blanc",

    # Adjectives
    "grand", "petit", "beau", "fort", "vieux", "jeune", "intelligent",

    # Transportation
    "voiture", "train", "avion", "vélo", "moto",

    # Technology
    "ordinateur", "smartphone", "internet", "jeu",

    # Objects
    "livre", "stylo", "table", "chaise", "porte", "fenêtre", "montre",

    # Family
    "mère", "père", "frère", "sœur", "fils", "fille",

    # Social Relations
    "ami", "collègue", "partenaire", "voisin", "ennemi"
]

# Convert the embeddings to a matrix (each row is an embedding)
words = list(w2v.keys())
plot_embeddings = []
for pw in plot_words:
    if pw in words:
        plot_embeddings.append(w2v[pw])
    else:
        raise IndexError(f"{pw} missing")

# embeddings = np.array(list(w2v.values()))

# Reduce the dimensionality to 2D using PCA
plot_embeddings = np.array(plot_embeddings)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# reduced_embeddings = pca.fit_transform(plot_embeddings)
tsne = TSNE(n_components=2)
reduced_embeddings = tsne.fit_transform(plot_embeddings)


# Plot the reduced embeddings
plt.figure(figsize=(10, 8))
plt.scatter(
    reduced_embeddings[:, 0],
    reduced_embeddings[:, 1],
    marker=".", color="#E24A33"
)

# Annotate points with the corresponding words
for i, word in enumerate(plot_words):
    plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

plt.savefig("embedding_plot.png", dpi=300, bbox_inches="tight")
plt.show()
