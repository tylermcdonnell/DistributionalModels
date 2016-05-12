from extractvectors import SentimentVector, PatternVector, PartOfSpeechVector
from utils import *


class SentimentFeatures(object):
    """Constructs sentiment related features for tuple of words. The features
    include the difference between sentiment scores (intensities) of the two
    members of a tuple. Three categories of sentiment scores are used:
    word-level score, average score of containing sentences, and average score
    of adjoining sentences.

    Attributes:
        sentiment (SentimentVector): Instance of SentimentVector class.
    """

    def __init__(self, filename, target_words_filename):
        """
        Args:
            filename (string): Path of the database dump.
            target_words_filename (string): Path of target words file.
        """
        self.sentiment = SentimentVector(filename, target_words_filename)
        self.sentiment.build_features()

    def sentiment_features(self, tuples):
        """Builds sentiment related features for every tuple in the supplied
        tuple list.

        Args:
            tuples (list(tuple)): List of tuples of words.

        Returns:
            dict: Dictionary of feature dictionaries for each tuple, which
                looks like:
                {tuple(word1, word2):{OWN:score, SAME_SENT:score,
                                      ADJ_SENT:score}, ...}

        Raises:
            KeyError: Throws KeyError if any of the words is absent from the
                pre-trained sentiment vectors.
        """
        features = {}
        for tup in tuples:
            features[tup] = {}
            if tup[0] not in self.sentiment.features or \
                    tup[1] not in self.sentiment.features:
                raise KeyError('Member of tuple not in Sentiment vector',
                               tup[0], tup[1])

            for key in self.sentiment.features[tup[0]]:
                features[tup][key] = abs(self.sentiment.features[tup[0]][key] -
                                         self.sentiment.features[tup[1]][key])
        return features


class PatternFeatures(object):
    """Constructs Pattern related features for tuple of words. This deals with
    the frequency of occurence of a pattern involving a word pair in the corpus
    during training. At the time of writting, the patterns used were:
    "Either X or Y", "Neither X nor Y", "From X to Y".

    Attributes:
        pattern (PatternVector): Instance of PatternVector class.
    """

    def __init__(self, filename):
        """
        Args:
            filename (string): Path of the database dump.
        """
        self.pattern = PatternVector(filename)
        self.pattern.build_features()

    def pattern_features(self, tuples):
        """Builds Pattern related features for every tuple in the supplied
        tuple list.

        Args:
            tuples (list(tuple)): List of tuples of words.

        Returns:
            dict: Dictionary of tuple: frequency
        """
        features = {}
        for tup in tuples:
            if tup in self.pattern.features:
                features[tup] = self.pattern.features[tup]
            else:
                features[tup] = 0
        return features


class PartOfSpeechFeatures(object):
    """Constructs features from Part-of-Speech specific distributional vectors.
    This includes cosine similarity of the word-vectors of the two members of a
    tuple. The metric used to measure co-occurrence of peers of a word needs to
    be specified - RAW, LMI, PPMI (see utils.py)

    Attributes:
        pos (PartOfSpeechVector): Instance of the PartOfSpeechVector class.
    """

    def __init__(self, filename, freq_metric):
        """
        Args:
            filename (string): Path of the database dump
            freq_metric (string): Type of frequency metric to be used. Possible
                values are RAW, LMI, PPMI (see utils.py)
        """
        self.pos = PartOfSpeechVector(filename, freq_metric)
        self.pos.build_features()

    def pos_features(self, tuples):
        """Builds features from the Part-of-Speech specific distributional
        vectors for the members of every tuple in the supplied tuple list.

        Args:
            tuples (list(tuple)): List of tuples of words.

        Returns:
            dict: Dictionary of tuple: cosine_similarity

        Raises:
            KeyError: Throws KeyError if any of the words is absent from the
            pre-trained distributional vectors.
        """
        features = {}
        for tup in tuples:
            # for tup, orig_tup in zip(tuples, orig_tuples):
            #     if tup[0] not in self.pos.features:
            #         print(tup[0], orig_tup[0])
            #     if tup[1] not in self.pos.features:
            #         print(tup[1], orig_tup[1])

            # if tup[0] not in self.pos.features or \
            #         tup[1] not in self.pos.features:
            #     raise KeyError('Member of tuple not in POS vector',
            #                    tup[0], tup[1])

            if tup[0] in self.pos.features:
                vec_1 = self.pos.features[tup[0]]
            else:
                vec_1 = {}

            if tup[1] in self.pos.features:
                vec_2 = self.pos.features[tup[1]]
            else:
                vec_2 = {}

            features[tup] = cosine_similarity(vec_1, vec_2)
        return features


def main():
    '''
    sf = SentimentFeatures(
        filename="./feature-dump/sentiment",
        target_words_filename="./WordLists/target_words.txt")
    tuples = [('good', 'bad'), ('slow', 'even')]
    tuples = [sorted_tuple(tup[0], tup[1]) for tup in tuples]
    print(sf.sentiment_features(tuples))
    '''

    '''
    pf = PatternFeatures("./feature-dump/pattern_either")
    tuples = [('good', 'bad'), ('slow', 'even')]
    tuples = [sorted_tuple(tup[0], tup[1]) for tup in tuples]
    print(pf.pattern_features(tuples))
    '''

    '''
    posf = PartOfSpeechFeatures("./feature-dump/adverb", LMI)
    tuples = [('good', 'bad'), ('slow', 'even')]
    tuples = [sorted_tuple(tup[0], tup[1]) for tup in tuples]
    print(posf.pos_features(tuples))
    '''

if __name__ == '__main__':
    main()
