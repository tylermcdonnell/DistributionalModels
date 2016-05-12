from distributional import SentimentModel, PatternModel, PartOfSpeechModel
from modelstore import BerkeleyStore
from utils import *

from abc import abstractmethod
from math import log

from nltk.sentiment.vader import SentimentIntensityAnalyzer


class ExtractVectors(object):
    """Abstract class providing skeleton for extracting and manipulating
    feature dumps of different types, such as Sentiment Vectors, Pattern
    Vectors, and Distributional Vectors.

    Attributes:
        features (dict): Dictionary of feature vectors for each word
        model: Corresponding model instance
    """

    def __init__(self):
        self.features = {}
        self.model = None

    @abstractmethod
    def __load_model(self, filename):
        """Reads the file from persistent storage and loads the corresponding
        model. Examples of model classes are SentimentModel, PatternModel,
        PartOfSpeechModel

        Args:
            filename (string): Path of the database dump
        """
        pass

    @abstractmethod
    def build_features(self):
        """Manipulates the feature vectors loaded from persistent storage, and
        populates the features attribute appropriately.
        """
        pass


class SentimentVector(ExtractVectors):
    """Loads and manipulates the Sentiment-based feature dump from persistent
    storage.

    Attributes:
        model (SentimentModel): Instance of the SentimentModel class
        target_words_filename (string): Path of target words file
    """

    def __init__(self, filename, target_words_filename):
        """
        Args:
            filename (string): Path of the database dump
            target_words_filename (string): Path of target words file
        """
        super(SentimentVector, self).__init__()
        self.target_words_filename = target_words_filename

        self.__load_model(filename)

    def __load_model(self, filename):
        self.model = SentimentModel()
        sentiment_store = BerkeleyStore(filename)
        self.model.load(sentiment_store)

    def __word_level_sentiment(self):
        """Calculates the sentiment of each word in the target word list.

        Returns:
            dict: Dictionary of word, sentiment score
        """
        analyzer = SentimentIntensityAnalyzer()

        word_sentiments = {}
        with open(self.target_words_filename, 'r') as f:
            for line in f:
                word = line[:-1].lower()
                score = analyzer.polarity_scores(word)['compound']
                word = lemmatize_and_stem(word)
                word_sentiments[word] = score
        return word_sentiments

    def build_features(self):
        """Builds the feature dictionary from Sentiment dump, which looks like:
        {word: {OWN:score, SAME_SENT:score, ADJ_SENT:score}, ...}
        """
        word_sentiments = self.__word_level_sentiment()

        # feature tags in dump
        POS = '#F_SSS_Positive'
        NEG = '#F_SSS_Negative'
        APOS = '#F_ASS_Positive'
        ANEG = '#F_ASS_Negative'
        SSIS = '#F_SSIS'
        ASIS = '#F_ASIS'

        # feature tags for final feature dictionary in utils.py

        for word, feats in self.model.model.items():
            self.features[word] = {}

            self.features[word][OWN] = word_sentiments[word]

            same_count = feats[POS] + feats[NEG]
            self.features[word][SAME_SENT] = float(feats[SSIS]) / same_count

            adjacent_count = feats[APOS] + feats[ANEG]
            self.features[word][ADJ_SENT] = float(feats[ASIS]) / adjacent_count


class PatternVector(ExtractVectors):
    """Loads and manipulates the Pattern-based feature dump from persistent
    storage.

    Attributes:
        model (PatternModel): Instance of the PatternModel class
    """

    def __init__(self, filename):
        """
        Args:
            filename (string): Path of the database dump
        """
        super(PatternVector, self).__init__()
        self.__load_model(filename)

    def __load_model(self, filename):
        self.model = PatternModel([])
        pattern_store = BerkeleyStore(filename)
        self.model.load(pattern_store)

    def build_features(self):
        """Builds the feature dictionary from Pattern dump, which looks like:
        {tuple(word_1, word_2): frequency, ...}
        """
        for word, vec in self.model.model.items():
            for peer, count in vec.items():
                tup = sorted_tuple(word, peer)
                if tup in self.features:
                    self.features[tup] = self.features[tup] + count
                else:
                    self.features[tup] = count


class PartOfSpeechVector(ExtractVectors):
    """Loads and manipulates the Part-of-Speech-based distributional vectors
    dump from persistent storage.

    Attributes:
        freq_metric (string): The type of metric to use for frequencies in
            word vectors.
        model (PartOfSpeechModel): Instance of the PartOfSpeechModel class
    """

    def __init__(self, filename, freq_metric):
        """
        Args:
            filename (string): Path of the database dump
            freq_metric (string): Type of frequency metric to be used. Possible
                values are RAW, LMI, PPMI (see utils.py)
        """
        super(PartOfSpeechVector, self).__init__()
        self.freq_metric = freq_metric
        self.__word_freq = {}

        self.__load_model(filename)

    def __load_model(self, filename):
        self.model = PartOfSpeechModel(None, None)
        pos_store = BerkeleyStore(filename)
        self.model.load(pos_store)

    def __word_frequency(self):
        """Sums up the frequency of occurrence of each word across every
        distributional vector and populates the __word_freq attribute
        """
        for _, vec in self.model.model.items():
            for word, freq in vec.items():
                if word in self.__word_freq:
                    self.__word_freq[word] += freq
                else:
                    self.__word_freq[word] = freq

    def build_features(self):
        """Builds the feature dictionary from Part-of-Speech dump, which looks
        like {word: {peer1: count, peer2: count, ...}, ...}
        """
        if self.freq_metric == RAW:
            self.features = self.model.model
        else:
            self.__word_frequency()

            global_freq = 0
            for _, freq in self.__word_freq.items():
                global_freq += freq

            for word, vec in self.model.model.items():
                self.features[word] = {}

                total_freq = 0
                for _, freq in vec.items():
                    total_freq += freq

                for peer, freq in vec.items():
                    numerator = float(freq) / total_freq
                    denominator = float(self.__word_freq[peer]) / global_freq
                    val = log(numerator / denominator)

                    if self.freq_metric == LMI:
                        self.features[word][peer] = val * freq
                    elif self.freq_metric == PPMI:
                        self.features[word][peer] = max(0, val)


def main():
    '''
    sentiment_feats = SentimentVector(
        filename="./feature-dump/sentiment",
        target_words_filename="./WordLists/target_words.txt")
    sentiment_feats.build_features()
    '''

    '''
    pattern_feats = PatternVector("./feature-dump/pattern_either")
    pattern_feats.build_features()
    '''

    '''
    pos_feats = PartOfSpeechVector("./feature-dump/verb", LMI)
    pos_feats.build_features()
    '''

if __name__ == '__main__':
    main()
