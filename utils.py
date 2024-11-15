import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss_function(m, c_pos, c_neg_list):
    r"""
    Fonction de perte donnée par:

    L = -[
            \log(\sigma(m \cdot c_{pos}))
            + \sum_{i = 1}^k \log(\sigma(-m \cdot c_{neg_i}))
        ]
    """
    result = np.log(sigmoid(m @ c_pos))
    for c_neg in c_neg_list:
        result += np.log(sigmoid(-m @ c_neg))
    return -result


def gradient_pos(m, c_pos):
    r"""
    Dérivé de L par rapport à c_pos:

    [\sigma(m \cdot c_{pos}) - 1] m
    """
    return (sigmoid(m @ c_pos) - 1) * m


def gradient_neg(m, c_neg):
    r"""
    Dérivé de L par rapport à c_neg:

    [\sigma(m \cdot c_{neg})] m
    """
    return sigmoid(m @ c_neg) * m


def gradient_m(m, c_pos, c_neg_list):
    r"""
    Dérivé de L par rapport à m:

    [\sigma(m \cdot c_{pos}) - 1] c_{pos}
    + \sum_{i=1}^k [\sigma(m \cdot c_{neg_i})] c_{neg_i}
    """
    result = (sigmoid(m @ c_pos) - 1) * c_pos
    for c_neg in c_neg_list:
        result += sigmoid(m @ c_neg) * c_neg
    return result
