import pickle
import os.path
# Python 2
#import anydbm
# Python 3
import dbm 

from abc import abstractmethod
from collections import Counter, defaultdict

class ModelStore(object):

    @abstractmethod
    def update(self, word, context):
        '''
        Updates the model of the given word with occurrences of the context features.
        '''
        pass

    @abstractmethod
    def context(self, word):
        '''
        Returns context of word as dictionary of features.
        '''
        pass


    @abstractmethod
    def vector(self, word):
        '''
        Returns N-dimensional context vector of word, where N is the number of features.
        '''
        pass

    @abstractmethod
    def keys(self):
        '''
        :return: All keys in the database.
        '''
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



class PickleStore(ModelStore):

    def __init__(self, filename):
        self.filename = filename
        if os.path.isfile(filename):
            self.model = pickle.load(open(filename, 'rb'))
        else:
            self.model = {}

    def reset_memory(self):
        self.model = defaultdict(lambda : Counter())

    def keys(self):
        return self.model.keys()

    def update(self, vectors):
        self.model = vectors
        pickle.dump(vectors, open(self.filename, 'wb'))

    def context(self, word):
        return self.model[word]

    def vector(self, word):
        pass



class BerkeleyStore(ModelStore):
    
    def __init__(self, filename):
        # Persistent model storage.
        self.db = dbm.open(filename, 'c')
        # Most processing should be in-memory for speed.
        self.reset_memory()

    def reset_memory(self):
        self.memory = defaultdict(lambda : Counter())

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

