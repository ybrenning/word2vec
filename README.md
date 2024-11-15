# Modèles de Langage (Language Models)

This repo contains material from a project on Word2Vec for the course "Modèles de Langage" at Aix-Marseille Université.

The project comprises two parts: the implementation and improvement of a base Word2Vec model using negative sampling and subsampling, and the evaluation of word analogies using existing embeddings from [NLPL](http://vectors.nlpl.eu/repository/).

The findings are documented in [this report](https://github.com/ybrenning/word2vec/blob/main/report/report.pdf).

## Setup

### Install dependencies 

```bash
$ python -m venv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```

### Word2Vec Model

The Word2Vec experiment for part one of the report can be configured and executed via `main.py`. The corpus and evaluation set can be found in the `data` directory.

### Analogies

Download the pre-generated embeddings:

```bash
$ wget http://vectors.nlpl.eu/repository/20/43.zip
$ unzip 43.zip -d 43
```

Feel free to read the metadata and README contained in the downloaded folder, then move the embeddings to the correct directory:

```bash
$ mv 43/model.txt data
$ rm -rf 43 43.zip
```

The analogies experiment for the second part of the report is contained in `analogies.py`.

As of right now, the `read_file` function must be called prior to executing the experiment in order to split and save the embeddings in chunks.


## References

Tomás Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. In Yoshua Bengio and Yann LeCun, editors, 1st International Conference on Learning Representations, ICLR 2013, Scottsdale, Arizona, USA, May 2-4, 2013, Workshop Track Proceedings, 2013a. URL http://arxiv.org/abs/1301.3781.

Tomás Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. Distributed representations of words and phrases and their compositionality. In Christopher J. C. Burges, Léon Bottou, Zoubin Ghahramani, and Kilian Q. Weinberger, editors, Advances in Neural Information Processing Systems 26 : 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings of a meeting held December 5–8, 2013, Lake Tahoe, Nevada, United States, pages 3111–3119, 2013b. URL https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html

Murhaf Fares, Andrey Kutuzov, Stephan Oepen, and Erik Velldal. Word vectors, reuse, and replicability : Towards a community repository of large-text resources. In Jörg Tiedemann and Nina Tahmasebi, editors, Proceedings of the 21st Nordic Conference on Computational Linguistics, pages 271–276, Gothenburg, Sweden, May 2017. Association for Computational Linguistics. URL https://aclanthology.org/W17-0237. Jerome H. Friedman, Jon Louis Bentley, and Raphael Ari Finkel. An algorithm for finding best matches in logarithmic expected time. ACM Transactions on Mathematical Software (TOMS), 3(3) :209–226, 1977.

Daniel Jurafsky. Speech and language processing, 2000.
