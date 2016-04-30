'''
Created on Mar 11, 2016

@author: Tyler McDonnell / http://tylermcdonnell.com
'''
# Standard Imports.
import os
import filters
from collections import namedtuple, defaultdict, Counter
from distributional import DistributionalModel, StandardModel, SimpleDistribution, PartOfSpeechModel
from modelstore import MemoryStore, BerkeleyStore
from wordmap import WordMap

# NLTK imports.
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

        
class ModelTrainer(object):

    def __init__(self, components):
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
        for k in mapped_context:
            word_context.update({ wordmap.demap(k) : mapped_context[k] })
        return word_context

    def train(self, text, components, persist=True):
        wordmap = self.wordmap
        for component in components:
            extractor = component.extractor
            filtered = self.preprocess(text, component.filters)  
            extractor.run(filtered, wordmap)
        if persist:
            self.persist(components)

    def train_multiple(self, files, components, persist_every=1000000000):
        size_tracker = 0
        count        = 0
        for file in files:
            count += 1
            size   = os.path.getsize(file)
            size_tracker += size
            print ("Processing %d of %d (%f until persist): %s" %
                   (count, len(files), float(size_tracker) / persist_every, file))
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

special = ['end', 'finish',
           'cry', 'sob',
           'cold', 'icy',
           'begin', 'start',
           'save', 'keep',
           'hope', 'wish',
           'choose', 'pick',
           'paste', 'glue', 
           'hurry', 'rush', 
           'brave', 'courageous',
           'sad', 'unhappy',
           'friend', 'pal',
           'enjoy', 'like',
           'error', 'mistake',
           'smile', 'grin',
           'harm', 'hurt',
           'large', 'big',
           'small', 'little', 
           'mother', 'mom',
           'father', 'dad']
special = filters.SpecialWords(special)
lemmatize = filters.Lemmatize()
stem = filters.Stem()
lowercase = filters.LowerCase()
content_word = filters.ContentWord()
stop_word = filters.StopWord()

# Standard Distribution
'''
model = StandardDistribution(3)
model.train_on_multiple(corpus[0:10], 
                        preprocessing_filters = [lemmatize, stem, lowercase, stop_word],
                        token_filters = [special])
model.save('standard.db')
'''

# Count
model = SimpleDistribution()
model.train_on_multiple(corpus[0:10000],
                        preprocessing_filters = [lemmatize, stem, lowercase],
                        token_filters = None)
model.save('count.db')

# Adjective Model
'''
model = PartOfSpeechModel(2, 'ADJ')
model.train_on_multiple(corpus[0:1],
                        preprocessing_filters = [lemmatize, stem, lowercase, stop_word],
                        token_filters = [special])
model.save('adjective.db')
'''

# Noun Model
'''
model = PartOfSpeechModel(2, 'NOUN')
model.train_on_multiple(corpus[0:1],
                        preprocessing_filters = [lemmatize, stem, lowercase, stop_word],
                        token_filters = [special])
model.save('noun.db')
'''

# Verb Model
'''
model = PartOfSpeechModel(2, 'VERB')
model.train_on_multiple(corpus[0:1],
                        preprocessing_filters = [lemmatize, stem, lowercase, stop_word],
                        token_filters = [special])
model.save('verb.db')
'''
