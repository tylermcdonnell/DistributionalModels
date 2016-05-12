from math import sqrt
from sklearn.metrics import precision_recall_fscore_support
import filters

# feature tags for final sentiment feature dictionary
OWN = 'own'                 # sentiment intensity of the word itself
SAME_SENT = 'same_sent'     # avg sentiment intensity of containing sentences
ADJ_SENT = 'adjacent_sent'  # avg. sentiment intensity of adjacent sentences

# frequency metric for distributional models
RAW = 'raw'     # raw frequency counts
LMI = 'lmi'     # LMI value
PPMI = 'ppmi'   # PPMI value

# tags for word pairs
SYNONYM = "SYNONYMS"
SYNONYM_LABEL = 0
ANTONYM = "ANTONYMS"
ANTONYM_LABEL = 1
# UNRELATED = "UNRELATED"
# UNRELATED_LABEL = 2

# dictionary mapping labels to integers
LABEL = {SYNONYM: SYNONYM_LABEL,
         ANTONYM: ANTONYM_LABEL,
         # UNRELATED: UNRELATED_LABEL
         }

# modes for reading feature matrix
BUILD_FEATURES = "build_features"   # builds features from word pairs list
LOAD_FROM_CSV = "load_from_csv"     # loads feature matrix from a csv


def sorted_tuple(first, second):
    """Creates a tuple of the two words with elements sorted alphabetically.

    Args:
        first (string): First word.
        second (string): Second word.

    Returns:
        tuple: Tuple of the two words.
    """
    return tuple(sorted([first, second]))


def cosine_similarity(vec_1, vec_2):
    """Computes the cosine similarity between two vectors, each represented as
    a dictionary.

    Args:
        vec_1 (dict): Dictionary for the first vector.
        vec_2 (dict): Dictionary for the second vector.

    Returns:
        float: cosine similarity of the two vectors. If one of the dictionaries
            is empty, -1 is returned.
    """
    dot = 0.0
    for key in vec_1:
        if key in vec_2:
            dot = dot + vec_1[key] * vec_2[key]

    norm_1 = sqrt(sum([val**2 for _, val in vec_1.items()]))
    norm_2 = sqrt(sum([val**2 for _, val in vec_2.items()]))

    if norm_1 == 0 or norm_2 == 0:
        return 0.0
    else:
        return dot / (norm_1 * norm_2)


def lemmatize_and_stem(word):
    # original order: "lemmatize then stem"
    lemmatize = filters.Lemmatize()
    stem = filters.Stem()

    word = lemmatize.apply([word])[0]
    word = stem.apply([word])[0]
    return word


def stem_and_lemmatize(word):
    # reverse order: "stem then lemmatize"
    lemmatize = filters.Lemmatize()
    stem = filters.Stem()

    word = stem.apply([word])[0]
    word = lemmatize.apply([word])[0]
    return word


def calc_accuracy(y_pred, y_true):
    count = 0
    for pred, truth in zip(y_pred, y_true):
        if pred == truth:
            count += 1
    return float(count) / len(y_pred)


def calc_class_wise_accuracy(y_pred, y_true):
    # assumes that the classes are labelled 0,1,2,...
    n_classes = len(set(y_true))
    correct_predictions = [0 for _ in range(0, n_classes)]
    class_prevelance = [0 for i in range(0, n_classes)]

    for pred, truth in zip(y_pred, y_true):
        class_prevelance[truth] += 1
        if pred == truth:
            correct_predictions[pred] += 1
    class_accuracies = []
    for num, total in zip(correct_predictions, class_prevelance):
        class_accuracies.append(float(num) / total)
    return class_accuracies


def calc_prec_recall_f1(y_pred, y_true):
    return precision_recall_fscore_support(y_true, y_pred, average=None)
