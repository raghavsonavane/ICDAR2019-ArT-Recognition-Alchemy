from __future__ import absolute_import

from .metrics import Accuracy, RecPostProcess, Accuracy_with_lexicon


__factory = {
    "accuracy": Accuracy,
    "accuracy_with_lexicon": Accuracy_with_lexicon,
}


def names():
    return sorted(__factory.keys())


def factory():
    return __factory
