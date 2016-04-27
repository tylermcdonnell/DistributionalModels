##################################################################
'''
@author: Tyler McDonnell / http://tylermcdonnell.com
'''

# Standard
import sys
from modelstore import BerkeleyStore
from abc import abstractmethod
from collections import Counter, defaultdict, namedtuple
# NLTK
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
##################################################################

class DistributionalModel(object):
    '''
    "Words are defined by the company they keep."
   
    A skeleton for building distributional models of semantic meaning.

    The Vocabulary is represented as a dictionary of { word : features }. Distributional
    models the feature vector by the contexts the word appears in. The resultant vectors
    are generally assumed to be sparse, which is why we choose to explicitly map features
    in a dictionary, rather than an array.
    '''

    @abstractmethod
    def train(self, text, preprocessing_filters=None, token_filters=None, wordmap=None):
        '''
        Trains the model on the input data.
        
        preprocessing_filters -- Filters to apply to input text before processing.
        token_filters         -- Filters to exclude tokens from feature vector creation.
        wordmap               -- WordMap for storage efficiency.
        '''
        pass

    def train_on_multiple(self, files, preprocessing_filters=None, token_filters=None):
        '''
        Trains the model on multiple input .txt files.
        
        files                 -- Absolute paths to files.
        preprocessing_filters -- Filters to apply to input text before processing.
        token_filters         -- Filters to exclude tokens from feature vector creation.
        '''
        count = 0
        for file in files:
            count += 1
            print ("Training model on %d of %d: %s" % (count, len(files), file))
            try:
                raw = open(file, 'rb').read()
                text = nltk.word_tokenize(raw)
                self.train(text, preprocessing_filters, token_filters)
            except:
                print ("Aborted training on %s - Unicode Error." % file)

    @abstractmethod
    def feature_vectors(self):
        '''
        Returns a dictionary of extracted vectors in form : { word : features }
        '''
        pass            



class StandardModel(DistributionalModel):
    '''
    A standard Distributional Model (or Vector Space Model of Semantic Meaning). Given
    a vocabulary V over a corpus C, a Distributional Model represents a word w in V as
    a vector in R^{V}. Each index i in the vector is a count of the occurrences of word 
    i in Vocabulary V within a window size of N of the word.

    e.g. Consider the sentence:

    "Words are defined by the company they keep."
    
    Using a window size of 2, building a distributional model on this sentence would
    increment the indices of the words "by", "the", "they", and "keep" in the vector
    representing "company".

    The feature vectors are thus vectors in R^{V} for each word: co-occurrence counts.
    '''

    def __init__(self, windowsize):
        # Data structures and model parameters.
        self.model      = defaultdict(lambda: Counter())
        self.windowsize = windowsize

    def train(self, text, preprocessing_filters=None, token_filters=None, wordmap=None):
        text = self._preprocessing(text, preprocessing_filters)
        for w in window(text, self.windowsize):
            word    = w.word
            context = w.context
            if self._token_desired(word, token_filters):
                self.model[word].update(context)

    def feature_vectors(self):
        return self.model

    def save(self, filename):
        db = BerkeleyStore(filename)
        db.update(self.model)

    def load(self, filename):
        db = BerkeleyStore(filename)
        for word in sorted(db.keys()):
            self.model.update( { word : db.context(word) })

    def _preprocessing(self, text, preprocessing_filters):
        for ppf in preprocessing_filters:
            text = ppf.apply(text)
        return text

    def _token_desired(self, token, token_filters):
        for tf in token_filters:
            if token not in tf.apply([token]):
                return False
        return True

    def _clear(self):
        '''
        Clears the in-memory mode--useful when the model is too big for memory.
        '''
        self.model = defaultdict(lambda: Counter())

    def _footprint(self):
        '''
        Returns an estimate of the current memory footprint of the model.
        '''
        return sum([sys.getsizeof(self.model[k] for k in self.model)])

def window(seq, size):
    Context = namedtuple('Context', ['word', 'context'])
    for i in range(len(seq)):
        context = seq[max(i - size, 0):i] + \
                  seq[min(i + 1, len(seq)):min(i + size, len(seq))]
        yield Context(word = seq[i], context = context)



class SentimentModel(DistributionalModel):

    def __init__(self):
        pass

    def run(self, text, wordmap=None):
        pass




