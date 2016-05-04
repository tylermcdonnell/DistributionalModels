######################################################################################
'''
@author: Tyler McDonnell / http://tylermcdonnell.com
'''

# Standard
import sys
import traceback
import itertools
from modelstore import BerkeleyStore, PickleStore
from abc import abstractmethod
from collections import Counter, defaultdict, namedtuple
# NLTK
import nltk
import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
######################################################################################

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
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 
        for file in sorted(files):
            count += 1
            print ("Training model on %d of %d: %s" % (count, len(files), file))
            try:
                raw = open(file, 'rb').read().decode('utf8')
                sentences = tokenizer.tokenize(raw)
                text = [nltk.word_tokenize(s) for s in sentences]
                self.train(text, preprocessing_filters, token_filters)
            except:
                print ("Aborted training on %s." % file)
                print (sys.exc_info()[0])
                traceback.print_exc()

    def _preprocessing(self, text, preprocessing_filters=None):
        '''
        Applies the preprocessing filters to the text in the order provided.
        
        text                  -- List of tokenized sentences.
        preprocessing_filters -- Filters to apply.
        '''
        for sentence in text:
            for ppf in preprocessing_filters:
                sentence = ppf.apply(sentence)
            yield sentence

    def _token_desired(self, token, token_filters):
        if token_filters:
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
        self.model      = defaultdict(counter)
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
        db.update_all(self.model)

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



class SimpleDistribution(DistributionalModel):
    '''
    Simple 1-gram probability distribution of tokens in model.
    '''
    
    def __init__(self):
        self.counts = defaultdict(dd)
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
        db.update_all(self.counts)

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

    def __init__(self, windowsize, pos):
        '''
        POS -- The part of speech we are interested in. This string must correspond
               to one of the 'universal' parts of speech: 
                 
               ADJ  - adjective
               ADP  - adposition
               ADV  - adverb
               CONJ - conjunction
               DET  - determiner, article
               NOUN - noun
               NUM  - numeral
               PRT  - particle
               PRON - pronoun
               VERB - verb
               .    - punctuation
               X    - other
        '''
        self.pos        = pos
        self.model      = defaultdict(counter)
        self.windowsize = windowsize

    def train(self, text, preprocessing_filters=None, token_filters=None, wordmap=None):
        for sentence in text:
            for w in pos_window(sentence, self.windowsize, self.pos):
                word    = w.word
                context = self._preprocessing([w.context], preprocessing_filters)
                if self._token_desired(word, token_filters):
                    self.model[word].update(next(context))

    def feature_vectors(self):
        return self.model

    def save(self, filename):
        db = BerkeleyStore(filename)
        db.update_all(self.model)

    def load(self, filename):
        db = BerkeleyStore(filename)
        for word in sorted(db.keys()):
            self.model.update( { word : db.context(word) })



class SentimentModel(DistributionalModel):
    '''
    Models sentiment-based distributional features using the Vader rule-based Sentiment Analyzer in NLTK.

    For occurrences of the word, these features are provided:

    1. Same Sentence Similarity (SSS) - Counts of sentiment of containg sentences of token.
       '#F_SSS_Positive' - '#F_SSS_Negative'

    2. Same Sentence Intensity Sum (SSIS) - Sum of same sentence intensities.
       '#F_SSIS'

    3. Adjacent Sentence Similarity (ASS) - Counts of sentiment of adjacent sentences.
       '#F_ASS_Positive' - '#F_ASS_Negative'

    4. Adjacent Sentence Intensity Sum (ASIS) - Sum of adjacent sentence intensities.
       '#F_ASIS'
    '''

    def __init__(self):
        self.model    = defaultdict(dd)
        self.analyzer = SentimentIntensityAnalyzer()

    def train(self, text, preprocessing_filters=None, token_filters=None, wordmap=None):
        '''
        Note: Preprocessing filters are applied after sentiment analysis scores are computed.
        '''
        # Calculate sentiment scores once upfront for processing.
        POS    = '#F_SSS_Positive'
        NEG    = '#F_SSS_Negative'
        APOS   = '#F_ASS_Positive'
        ANEG   = '#F_ASS_Negative'
        SSIS   = '#F_SSIS'
        ASIS   = '#F_ASIS'
        scores = [self.analyzer.polarity_scores((' ').join(sentence)) for sentence in text]
        text   = self._preprocessing(text, preprocessing_filters)
        for idx, sentence in enumerate(text):
            for token in sentence:
                if self._token_desired(token, token_filters):
                    # Categorize sentence of occurrence.
                    category = self.categorize(scores[idx])
                    if (category == 'pos'):
                        self.model[token][POS] += 1
                    if (category == 'neg'):
                        self.model[token][NEG] += 1
                    # Intensity. -1 = Most Negative | 1 = Most Positive
                    self.model[token][SSIS] += scores[idx]['compound']

                    # Categorize adjacent sentences. Do they tell us something different?
                    # e.g., do they tell us something extra in neutral cases?
                    adjacent = []
                    if (idx - 1 >= 0):
                        adjacent.append(idx - 1)
                    if (idx + 1 < len(scores)):
                        adjacent.append(idx + 1)
                    adjacent = [self.categorize(scores[i]) for i in adjacent]
                    for category in adjacent:
                        if (category == 'pos'):
                            self.model[token][APOS] += 1
                        if (category == 'neg'):
                            self.model[token][ANEG] += 1
                        # Intensity. -1 = Most Negative | 1 = Most Positive
                        self.model[token][ASIS] += scores[idx]['compound']

    def categorize(self, vader_score):
        '''
        Given a Vader polarity score, categorizes as either Positive or Negative.
        :param vader_score: NLTK Vader polarity score.
        :return: 'pos' if Positive; 'neg' is Negative.
        '''
        # Vader overwhelmingly favors Neutral; We will simply take the maximum of Positive and Negative.
        vader_score = [(s, vader_score[s]) for s in vader_score \
                       if (s != 'compound') and (s != 'neu')]
        vader_score.sort(key=lambda s: s[1], reverse=True)
        return vader_score[0][0]

    def feature_vectors(self):
        return self.model

    def save(self, filename):
        db = PickleStore(filename)
        db.update_all(self.model)

    def load(self, filename):
        db = PickleStore(filename)
        for word in sorted(db.keys()):
            self.model.update({word: db.context(word)})


class PatternModel(DistributionalModel):
    '''
    Models distributional string patterns in text. Popular patterns from literature include:

    "Either X or Y"
    "Neither X or Y"
    "From X to Y"
    '''

    def __init__(self, patterns):
        '''
        Initializes a PatternModel for the specified patterns.
        :param patterns: Iterable of patterns. Each pattern should be a string, where variables are represented by
        the substring "$V$". For example, the popular pattern "Either X or Y" could be represented by the pattern:
        "Either $v$ or $v$"
        '''
        self.model      = defaultdict(dd)
        self.stemmer    = SnowballStemmer("english")
        self.lemmatizer = WordNetLemmatizer()
        self.VARIABLE   = '$v$'
        self.patterns   = [pattern.split() for pattern in patterns]

    def train(self, text, preprocessing_filters=None, token_filters=None, wordmap=None):
        text = self._preprocessing(text, preprocessing_filters)
        for pattern in self.patterns:
            pattern = next(self._preprocessing([pattern], preprocessing_filters))
            for sentence in text:
                found = self.search_for_pattern(sentence, pattern)
                for p in found:
                    for permutation in itertools.permutations(p, 2):
                        self.model[permutation[0]][permutation[1]] += 1
                        print ('Found matching pattern: {} - {}'.format(permutation[0], permutation[1]))

    def search_for_pattern(self, text, pattern):
        for t_idx in range(len(text)):
            if self.pattern_match(text, t_idx, pattern):
                yield [text[t_idx + i] for i in range(len(pattern)) if pattern[i] == self.VARIABLE]

    def pattern_match(self, text, t_idx, pattern):
        p_idx = 0
        while ((t_idx < len(text) and text[t_idx] == pattern[p_idx]) or pattern[p_idx] == self.VARIABLE):
            p_idx += 1
            t_idx += 1
            if p_idx == len(pattern):
                return True
        return False

    def feature_vectors(self):
        return self.model

    def save(self, filename):
        db = PickleStore(filename)
        db.update_all(self.model)

    def load(self, filename):
        db = PickleStore(filename)
        for word in sorted(db.keys()):
            self.model.update({word: db.context(word)})






def window(seq, size):
    Context = namedtuple('Context', ['word', 'context'])
    for i in range(len(seq)):
        context = seq[max(i - size, 0):i] + \
                  seq[min(i + 1, len(seq)):min(i + size, len(seq))]
        yield Context(word = seq[i], context = context)

def pos_window(seq, size, pos):
    Context = namedtuple('Context', ['word', 'context'])
    seq = nltk.pos_tag(seq, 'universal')
    for i in range(len(seq)):
        word          = seq[i][0]
        before_target = [t[0] for t in seq[0:i] if t[1] == pos]
        after_target  = [t[0] for t in seq[min(i+1, len(seq)):len(seq)] if t[1] == pos]
        context = before_target[max(len(before_target) - size, 0):len(before_target)] + \
                  after_target[0:min(size, len(after_target))]
        yield Context(word, context = context)

'''
# To accomodate Pickle, which doesn't accept lambda definitions.
'''
def counter():
    return Counter()

'''
# To accomodate Pickle, which doesn't accept lambda definitions.
'''
def dd():
    return defaultdict(int)