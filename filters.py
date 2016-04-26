
# Standard Imports.
from collections import namedtuple
from abc import abstractmethod

# NLTK Imports.
import nltk
from nltk.corpus import stopwords

'''
Implements a pre-processing filter on input text.
'''
class Filter(object):

    @abstractmethod
    def apply(self, text):
        pass



'''
Filters out stopwords (common words).
'''
class StopWordFilter(Filter):

    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

    def apply(self, text):
        return [w for w in text if w not in self.stopwords]




'''
Filters out non-content words: nouns, adjectives, and verbs.

We could use more sophisticated parsers, but since we are 
using course-grained tags, it probably doesn't matter.

Note: This uses POS tagging and requires source sentences.
'''
class ContentWordFilter(Filter):    

    def apply(self, text):
        content = set(['NOUN', 'ADJ', 'VERB'])
        TaggedWord = namedtuple("TaggedWord", ['word', 'pos'])
        tagged = [TaggedWord(word=w[0], pos=w[1]) for w in nltk.pos_tag(text, 'universal')]
        return [t.word for t in tagged if t.pos in content]

'''
path = "/media/1tb/tyler/cs/data/Corpora/Gutenberg/files/10022/10022.txt"
tokens = nltk.word_tokenize(open(path, 'rb').read())
print (tokens[0:50])
print (StopWordFilter().filter(tokens)[0:50])
'''
