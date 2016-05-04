######################################################################################
'''
@author: Tyler McDonnell / http://tylermcdonnell.com
'''

#  Python 2
#import anydbm
# Python 3
import dbm

import pickle
import os.path

from abc import abstractmethod
from collections import Set
######################################################################################

class ModelStore(object):
    '''
    Model Store implements persistent storage for distributional models. In general, though distributional
    vector models are very sparse, the resource requirements are often very large due to the extremely high
    number of words being modeled. Thus, persistent storage is important for both preserving the results of
    distributional models built over large corpuses and for storing models that are too large to fit in memory.

    We assume that a distributional model builds a sparse vector of labeled features for a string S.

    i.e., for a particular string in the vocabulary S, the distributional model contains a vector:

    S : { 'feature1label' : 'feature1value', 'feature2label' : 'feature2value', ... }
    '''

    @abstractmethod
    def update(self, string, vector):
        '''
        Updates the persistent image of the vector with the associated labeled context features.
        i.e., Persistent(string) = context

        Note: This may be slow. For updating many vectors, update_all is recommended.
        :param string: String being modeled by vector.
        :param vector: Dictionary of labeled features constituting feature vector. e.g., { featureLabel : feature }
        '''
        pass

    def update_all(self, vectors):
        '''
        Updates the persistent image of all supplied vectors.
        :param vectors: Dictionary of { string : feature vector }.
        '''

    @abstractmethod
    def keys(self):
        '''
        :return: All strings modeled by this Model Store with a feature vector.
        '''
        pass

    @abstractmethod
    def context(self, string):
        '''
        Returns context of the string as a dictionary of labeled features.
        '''
        pass


    @abstractmethod
    def vector(self, word):
        '''
        Returns N-dimensional context vector of word, where N is the number of features.
        '''
        pass



class PickleStore(ModelStore):
    '''
    A Pickle-based Model Store implementation. In general, this model is not recommended.

    Advantages of Pickle:
    - All base Python data-types can be saved and loaded easily using a single Pickle command.

    Disadvantages of Pickle:
    - Pickle uses stupid amounts of memory. This makes it infeasible for many distributional models.
    - Pickle is not efficient for saving extremely large data structures.
    - Pickling objects is sensitive to class changes.
    '''

    def __init__(self, filename):
        self.filename = filename
        if os.path.isfile(filename + '.pkl'):
            self.model = pickle.load(open(filename, 'rb'))
        else:
            self.model = {}

    def keys(self):
        return self.model.keys()

    def update(self, string, vector):
        self.model.update({ string : vector })
        pickle.dump(self.model, open(self.filename, 'wb'))

    def update_all(self, vectors):
        self.model = vectors
        pickle.dump(self.model, open(self.filename, 'wb'))

    def context(self, word):
        return self.model[word]

    def vector(self, word):
        pass



class BerkeleyStore(ModelStore):
    '''
    A database Model Store implementation. When creating a new Model, this class will prefer the Berkeley DB, but
    when it is unavailable on a given machine, it will default to the default of either anydbm (Python 2) or dbm
    (Python 3). Accordingly, it will also support loads form any variety of db supported by these two modules.
    '''
    
    def __init__(self, filename):
        # Persistent model storage.
        self.db = dbm.open(filename, 'c')

    def keys(self):
        # Depending on the type of database, reads from database may be in bytes.
        return [s.decode('utf-8') for s in self.db.keys()]

    def update(self, string, vector):
        string = str(string)
        keys = set([s.decode('utf-8') for s in self.db.keys()])
        if string in keys:
            # Berkeley DB only accepts strings. Load the Pickled dictionary.
            persistent = pickle.loads(self.db[string])
            persistent.update(vector)
            self.db[string] = pickle.dumps(persistent)
        else:
            self.db[string] = pickle.dumps(vector)

    def update_all(self, vectors):
        for string, vector in vectors.items():
            self.update(string, vector)

    def context(self, string):
        # Berkeley DB only accepts strings. Pickle the vector.
        return pickle.loads(self.db[string])

    def vector(self, word):
        pass

