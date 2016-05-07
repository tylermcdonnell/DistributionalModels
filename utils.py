from math import sqrt

# feature tags for final sentiment feature dictionary
OWN = 'own'                 # sentiment intensity of the word itself
SAME_SENT = 'same_sent'     # avg sentiment intensity of containing sentences
ADJ_SENT = 'adjacent_sent'  # avg. sentiment intensity of adjacent sentences

# frequency metric for distributional models
RAW = 'raw'     # raw frequency counts
LMI = 'lmi'     # LMI value
PPMI = 'ppmi'   # PPMI value


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
        return -1.0
    else:
        return dot / (norm_1 * norm_2)
