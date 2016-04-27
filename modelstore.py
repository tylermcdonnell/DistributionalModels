import pickle
import anydbm

from abc import abstractmethod
from collections import Counter, defaultdict

class ModelStore(object):

    '''
    Updates the model of the given word with occurrences of the context features.
    '''
    @abstractmethod
    def update(self, word, context):
        pass

    '''
    Returns context of word as dictionary of features.
    '''
    @abstractmethod
    def context(self, word):
        pass

    '''
    Returns N-dimensional context vector of word, where N is the number of features.
    '''
    @abstractmethod
    def vector(self, word):
        pass



class MemoryStore(ModelStore):

    def __init__(self):
        self.store = defaultdict(Counter)

    def update(self, vectors):
        for word, vector in vectors.items():
            self.store[word] += vector

    def context(self, word):
        return self.store[word]

    def vector(self, word):
        return None



class BerkeleyStore(ModelStore):
    
    def __init__(self, filename):
        # Persistent model storage.
        self.db = anydbm.open(filename, 'c')
        # Most processing should be in-memory for speed.
        self.reset_memory()

    def reset_memory(self):
        self.memory = defaultdict(Counter)

    def keys(self):
        return self.db.keys()

    def update(self, vectors):
        print ("Updating Berkeley store...")
        for word, features in vectors.items():
            word = str(word)
            if word in self.db:
                persistent = self.db[str(word)]
                persistent = pickle.loads(persistent)
                persistent.update(features)
                self.db[word] = pickle.dumps(persistent)
            else:
                self.db[word] = pickle.dumps(features)
        print ("Finished updating Berkeley Store!")

    def context(self, word):
        return pickle.loads(self.db[word])

    def vector(self, word):
        pass

