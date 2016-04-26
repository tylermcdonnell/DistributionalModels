import sys

from abc import abstractmethod
from collections import Counter, defaultdict, namedtuple

class FeatureExtractor(object):

    '''
    Extracts distributional context features from data.
    '''
    @abstractmethod
    def run(self, data, wordmap=None):
        pass

    '''
    Returns all extracted features in form: { label : features }
    '''
    @abstractmethod
    def feature_vectors(self):
        pass

    '''
    Clears memory- useful when processing very large data.
    '''
    @abstractmethod
    def clear(self):
        pass

    '''
    Returns an estimate of the memory footprint of this model.
    '''
    @abstractmethod
    def footprint(self):
        pass



class DistributionalExtractor(FeatureExtractor):

    def __init__(self, window_size):
        self.window_size = window_size
        self.model       = defaultdict(lambda: Counter)

    def run(self, text, wordmap=None):
        self.model = defaultdict(lambda: Counter())
        for w in window(text, self.window_size):
            if wordmap:
                word    = wordmap.apply(w.word)
                context = [wordmap.apply(c) for c in w.context]
            self.model[word].update(context)
        return self.model

    def feature_vectors(self):
        return self.model

    def clear(self):
        self.model = defaultdict(lambda: Counter)

    def footprint(self):
        r = sum([sys.getsizeof(self.model[k] for k in self.model)])
        return 



class SentimentExtractor(FeatureExtractor):

    def __init__(self):
        pass

    def run(self, text, wordmap=None):
        pass



def window(seq, size):
    Context = namedtuple('Context', ['word', 'context'])
    for i in range(len(seq)):
        context = seq[max(i - size, 0):i] + \
                  seq[min(i + 1, len(seq)):min(i + size, len(seq))]
        yield Context(word = seq[i], context = context)
