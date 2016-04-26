'''
Created on Mar 11, 2016

@author: Tyler McDonnell / http://tylermcdonnell.com
'''
# Standard Imports.
import os

from collections import namedtuple, defaultdict, Counter
from featureextractor import FeatureExtractor, DistributionalExtractor
from filters import Filter, StopWordFilter, ContentWordFilter
from modelstore import MemoryStore, BerkeleyStore
from wordmap import WordMap

# NLTK imports.
import nltk

from filter import StopWordFilter, ContentWordFilter
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer


class ModelComponent(object):

    def __init__(self, extractor, *filters):
        self.extractor = extractor
        self.filters   = filters


        
class DistributionalModel(object):

    def __init__(self, components, memlimit=10000000000, wordmap=None, store=None):
        self.lemmatizer = WordNetLemmatizer()
        if wordmap and store:
            self.wordmap = WordMap.load(wordmap)
            self.store   = BekerleyStore(store)
        else:
            self.store   = BerkeleyStore('store')
            self.wordmap = WordMap()
        self.memlimit   = memlimit

    def preprocess(self, text, filters):
        for f in filters:
            text = f.apply(text)
        text = [s.lower() for s in text]
        text = [self.lemmatizer.lemmatize(w) for w in text]
        return text

    '''
    Persists all feature vectors from extractor to database.
    '''
    def persist(self, components):
        for component in components:
            extractor = component.extractor
            self.store.update(extractor.feature_vectors())
            extractor.clear()
        self.wordmap.save('wordmap')

    def context(self, word):
        wordmap        = self.wordmap
        mapped_context = self.store.context(str(wordmap.apply(word)))
        word_context = {}
        for key in mapped_context:
            word_context.update({ wordmap.demap(key) : mapped_context[key] })
        return word_context

    def train(self, text, components, persist=True):
        wordmap = self.wordmap
        for component in components:
            extractor = component.extractor
            filtered = self.preprocess(text, component.filters)  
            extractor.run(filtered, wordmap)
        if persist:
            self.persist(components)

    def train_multiple(self, files, components, persist_every=10000000):
        size_tracker = 0
        count        = 0
        for file in files:
            count += 1
            size   = os.path.getsize(file)
            size_tracker += size
            print ("Processing %d of %d (%d bytes/%d bytes): %s" %
                   (count, len(files), size_tracker, persist_every, file))
            # If we can't decode as UTF-8 nltk won't work.
            try:
                raw = open(file, 'rb').read().decode('utf8')
            except:
                print ("Ignored %s because of decode error." % file)
                continue
            text = nltk.word_tokenize(raw)
            model.train(text, components, persist=False)
            if size_tracker >= persist_every:
                size_tracker = 0
                self.persist(components)
        self.persist(components)
        


            
'''
Returns a list of all .txt files in the given directory.
'''
def extract_corpus(directory):
    print ("Extracting .txt files from corpus...")
    extracted = []
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file = os.path.join(root, file)
                extracted += [file]       
    print ("Extracted %d .txt files." % len(extracted))            
    return extracted

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]



corpus = extract_corpus('/media/1tb/tyler/cs/data/Corpora/Gutenberg/files/')
cf = ContentWordFilter()
sf = StopWordFilter()
d_component = ModelComponent(DistributionalExtractor(3), sf)
components = [d_component]
model = DistributionalModel(components)
model.train_multiple(corpus[0:1000], components)
print (model.context('gutenberg'))
model.store.db.close()
