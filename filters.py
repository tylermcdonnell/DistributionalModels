##################################################################
'''
@author: Tyler McDonnell / http://tylermcdonnell.com
'''

# Standard Imports.
from collections import namedtuple, Set
from abc import abstractmethod

# NLTK Imports.
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
##################################################################

class Filter(object):
    '''
    Implements a pre-processing filter on input text.
    '''

    @abstractmethod
    def apply(self, text):
        pass



class SpecialWords(Filter):
    '''
    Filters all but the specified words.
    '''
    
    def __init__(self, words=None):
        self.selected = set(words)

    def update(self, words):
        '''
        Updates the list of selected words.
        '''
        self.selected = words

    def apply(self, text):
        return [t for t in text if t in self.selected]
       
 

class Lemmatize(Filter):
    '''
    Lemmatizes words in text.
    '''
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
    def apply(self, text):
        return [self.lemmatizer.lemmatize(t) for t in text]



class Stem(Filter):
    '''
    Stems words in text.
    '''

    def __init__(self):
        self.stemmer = SnowballStemmer("english")
        
    def apply(self, text):
        return [self.stemmer.stem(t) for t in text]



class LowerCase(Filter):
    '''
    Transforms to all lower-case text.
    '''
    
    def apply(self, text):
        return [t.lower() for t in text]



class StopWord(Filter):
    '''
    Filters out stopwords (common words).
    '''

    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

    def apply(self, text):
        return [w for w in text if w not in self.stopwords]


class ContentWord(Filter):    
    '''
    Filters out non-content words: nouns, adjectives, and verbs.
    
    We could use more sophisticated parsers, but since we are 
    using course-grained tags, it probably doesn't matter.
    
    Note: This uses POS tagging and requires source sentences.
    '''

    def apply(self, text):
        content = set(['NOUN', 'ADJ', 'VERB'])
        TaggedWord = namedtuple("TaggedWord", ['word', 'pos'])
        tagged = [TaggedWord(word=w[0], pos=w[1]) for w in nltk.pos_tag(text, 'universal')]
        return [t.word for t in tagged if t.pos in content]
