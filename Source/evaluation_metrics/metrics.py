# from __future__ import absolute_import

import numpy as np
import string
import math

import torch
import torch.nn.functional as F

import sys

sys.path.append("./")


def _normalize_text(text):
    text = "".join(filter(lambda x: x in string.digits + string.ascii_letters, text))
    return text.lower()


def get_str(output):

    char2id = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "a": 10,
        "b": 11,
        "c": 12,
        "d": 13,
        "e": 14,
        "f": 15,
        "g": 16,
        "h": 17,
        "i": 18,
        "j": 19,
        "k": 20,
        "l": 21,
        "m": 22,
        "n": 23,
        "o": 24,
        "p": 25,
        "q": 26,
        "r": 27,
        "s": 28,
        "t": 29,
        "u": 30,
        "v": 31,
        "w": 32,
        "x": 33,
        "y": 34,
        "z": 35,
        "!": 36,
        '"': 37,
        "#": 38,
        "$": 39,
        "%": 40,
        "&": 41,
        "'": 42,
        "(": 43,
        ")": 44,
        "*": 45,
        "+": 46,
        ",": 47,
        "-": 48,
        ".": 49,
        "/": 50,
        ":": 51,
        ";": 52,
        "<": 53,
        "=": 54,
        ">": 55,
        "?": 56,
        "@": 57,
        "[": 58,
        "\\": 59,
        "]": 60,
        "^": 61,
        "_": 62,
        "`": 63,
        "{": 64,
        "|": 65,
        "}": 66,
        "~": 67,
        "EOS": 68,
        "PADDING": 69,
        "UNKNOWN": 70,
        "∑": 71,
        "，": 72,
        "º": 73,
        "ó": 74,
        "ʃ": 75,
        "ü": 76,
        "ε": 77,
        "ä": 78,
        "è": 79,
        "и": 80,
        "í": 81,
        "ö": 82,
        "λ": 83,
        "á": 84,
        "：": 85,
        "®": 86,
        "é": 87,
        "·": 88,
        "＃": 89,
    }
    id2char = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "a",
        11: "b",
        12: "c",
        13: "d",
        14: "e",
        15: "f",
        16: "g",
        17: "h",
        18: "i",
        19: "j",
        20: "k",
        21: "l",
        22: "m",
        23: "n",
        24: "o",
        25: "p",
        26: "q",
        27: "r",
        28: "s",
        29: "t",
        30: "u",
        31: "v",
        32: "w",
        33: "x",
        34: "y",
        35: "z",
        36: "!",
        37: '"',
        38: "#",
        39: "$",
        40: "%",
        41: "&",
        42: "'",
        43: "(",
        44: ")",
        45: "*",
        46: "+",
        47: ",",
        48: "-",
        49: ".",
        50: "/",
        51: ":",
        52: ";",
        53: "<",
        54: "=",
        55: ">",
        56: "?",
        57: "@",
        58: "[",
        59: "\\",
        60: "]",
        61: "^",
        62: "_",
        63: "`",
        64: "{",
        65: "|",
        66: "}",
        67: "~",
        68: "EOS",
        69: "PADDING",
        70: "UNKNOWN",
        71: "∑",
        72: "，",
        73: "º",
        74: "ó",
        75: "ʃ",
        76: "ü",
        77: "ε",
        78: "ä",
        79: "è",
        80: "и",
        81: "í",
        82: "ö",
        83: "λ",
        84: "á",
        85: "：",
        86: "®",
        87: "é",
        88: "·",
        89: "＃",
    }

    end_label = char2id["EOS"]
    num_samples, max_len_labels = output.size()
    num_classes = len(char2id.keys())
    output = to_numpy(output)

    # list of char list
    pred_list = []
    for i in range(num_samples):
        pred_list_i = []
        for j in range(max_len_labels):
            if output[i, j] != end_label:
                pred_list_i.append(id2char[output[i, j]])
            else:
                break
        pred_list.append(pred_list_i)

    pred_list = ["".join(pred) for pred in pred_list]

    return pred_list


def Accuracy(output, target, dataset=None, isReturnPred=True):
    pred_list, targ_list = get_str_list(output, target, dataset)
    acc_list = [(pred == targ) for pred, targ in zip(pred_list, targ_list)]
    accuracy = 1.0 * sum(acc_list) / len(acc_list)
    return accuracy, pred_list


def Accuracy_with_lexicon(output, target, dataset=None, file_names=None):
    pred_list, targ_list = get_str_list(output, target, dataset)
    accuracys = []

    # with no lexicon
    acc_list = [(pred == targ) for pred, targ in zip(pred_list, targ_list)]
    accuracy = 1.0 * sum(acc_list) / len(acc_list)
    accuracys.append(accuracy)

    # lexicon50
    if len(file_names) == 0 or len(dataset.lexicons50[file_names[0]]) == 0:
        accuracys.append(0)
    else:
        refined_pred_list = [
            _lexicon_search(dataset.lexicons50[file_name], pred) for file_name, pred in zip(file_names, pred_list)
        ]
        acc_list = [(pred == targ) for pred, targ in zip(refined_pred_list, targ_list)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)
        accuracys.append(accuracy)

    # lexicon1k
    if len(file_names) == 0 or len(dataset.lexicons1k[file_names[0]]) == 0:
        accuracys.append(0)
    else:
        refined_pred_list = [
            _lexicon_search(dataset.lexicons1k[file_name], pred) for file_name, pred in zip(file_names, pred_list)
        ]
        acc_list = [(pred == targ) for pred, targ in zip(refined_pred_list, targ_list)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)
        accuracys.append(accuracy)

    # lexiconfull
    if len(file_names) == 0 or len(dataset.lexiconsfull[file_names[0]]) == 0:
        accuracys.append(0)
    else:
        refined_pred_list = [
            _lexicon_search(dataset.lexiconsfull[file_name], pred) for file_name, pred in zip(file_names, pred_list)
        ]
        acc_list = [(pred == targ) for pred, targ in zip(refined_pred_list, targ_list)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)
        accuracys.append(accuracy)

    return accuracys


def RecPostProcess(output, target, score, dataset=None):
    pred_list, targ_list = get_str_list(output, target, dataset)
    max_len_labels = output.size(1)
    score_list = []

    score = to_numpy(score)
    for i, pred in enumerate(pred_list):
        len_pred = len(pred) + 1  # eos should be included
        # maybe the predicted string don't include a eos.
        len_pred = min(max_len_labels, len_pred)
        score_i = score[i, :len_pred]
        score_i = math.exp(sum(map(math.log, score_i)))
        score_list.append(score_i)

    return pred_list, targ_list, score_list


def to_numpy(tensor):
    """
    also send the data to cpu
    :param tensor:
    :return:
    """
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor
