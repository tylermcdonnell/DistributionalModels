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
import nltk.data
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
##################################################################

class DistributionalModel(object):
    '''
    "Words are defined by the company they keep."
   
    A skeleton for building distributional models of semantic meaning.

    In a distributional model, each word in a vocabulary V is represented by an array of
    features, where each feature is derived from the contexts in which that word appears.
    The resultant vectors are generally sparse, so we choose to model them here as dicts,
    where each feature has a string label. 
    '''

    def train_on_multiple(self, files, preprocessing_filters=None, token_filters=None):
        '''
        Trains the model on multiple input .txt files.
        
        files                 -- Absolute paths to files.
        preprocessing_filters -- Filters to apply to input text before processing.
        token_filters         -- Filters to exclude tokens from feature vector creation.
        '''
        count = 0
        for file in files:
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 
            count += 1
            print ("Training model on %d of %d: %s" % (count, len(files), file))
            #try:
            raw = open(file, 'rb').read()
            sentences = tokenizer.tokenize(raw)
            text = [nltk.word_tokenize(s) for s in sentences]
            self.train(text, preprocessing_filters, token_filters)
            #except:
            #    print ("Aborted training on %s - Unicode Error." % file)

    def _preprocessing(self, text, preprocessing_filters=None):
        '''
        Applies the preprocessing filters to the text in the order provided.
        
        text                  -- List of tokenized sentences.
        preprocessing_filters -- Filters to apply.
        '''
        processed = []
        for ppf in preprocessing_filters:
            for sentence in text:
                processed_sentence = ppf.apply(sentence)
                processed.append(processed_sentence)
        return processed

    def _token_desired(self, token, token_filters):
        for tf in token_filters:
            if token not in tf.apply([token]):
                return False
        return True

    @abstractmethod
    def train(self, text, preprocessing_filters=None, token_filters=None, wordmap=None):
        '''
        Trains the model on the input data.
        
        text                  -- List of tokenized sentences.
        preprocessing_filters -- Filters to apply to input text before processing.
        token_filters         -- Filters to exclude tokens from feature vector creation.
        wordmap               -- WordMap for storage efficiency.
        '''
        pass

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
        for sentence in text:
            for w in window(sentence, self.windowsize):
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



class SimpleDistribution(DistributionalModel):
    '''
    Simple 1-gram probability distribution of tokens in model.
    '''
    
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.total  = 0
    
    def train(self, text, preprocessing_filters=None, token_filters=None, wordmap=None):
        text = self._preprocessing(text, preprocessing_filters)
        for sentence in text:
            for token in sentence:
                self.total += 1
                if self._token_desired(token, token_filters):
                    self.counts[token]['#F_WordCount'] += 1

    def feature_vectors(self):
        return self.counts

    def save(self, filename):
        db = BerkeleyStore(filename)
        db.update(self.counts)

    def load(self, filename):
        db = BerkeleyStore(filename)
        for word in sorted(db.keys()):
            self.counts.update( { word : db.context(word) })



class PartOfSpeechModel(DistributionalModel):
    '''
    The Part-of-Speech model is a variant of the traditional distributional model which,
    for a given window size N, selects the nearest N words of the specified POS within
    the sentence boundary. 
    '''



class SentimentModel(DistributionalModel):

    def __init__(self):
        pass

    def run(self, text, wordmap=None):
        pass




