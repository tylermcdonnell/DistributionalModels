#####################################################################################
'''
@author: Tyler McDonnell / http://tylermcdonnell.com
'''

# Standard
import pickle
# Python 2
#import anydbm
# Python 3
import dbm
######################################################################################

class WordMap(object):
    '''
    Maps words to integers for space efficiency and/or array transformation: e.g., 
    sparse feature vectors may more efficiently be described as dicts of features
    which can be converted to mapped arrays on demand.
    '''
    
    def __init__(self):
        self.count = 0
        self.wmap  = {}
        self.rwmap = {}

    def wmap(self, word):
        '''
        Returns the integer mapping for this word. If this word is not in the
        WordMap, an integer mapping will be created for it.
        '''
        if word not in self.wmap:
            self.wmap.update({ word : str(self.count) })
            self.rwmap.update({ str(self.count) : word })
            self.count += 1
        return int(self.wmap[word])

    def dewmap(self, map_id):
        '''
        Returns the word for this integer mapping. If this word is not in the
        WordMap, returns None.
        '''
        map_id = str(map_id)
        return self.rwmap[map_id] if map_id in self.rwmap else None

    def save(self, filename):
        '''
        Saves this wordmap to disk.
        '''
        print ("Saving WordMap to disk: %s" % filename)
        db = anydbm.open(filename, 'c')
        for key in self.wmap:
            db[key] = self.wmap[key]
        print ("Finished saving WordMap to disk.")
        db.close()
    
    @classmethod
    def load(filename):
        print ("Loading WordMap: %s" % filename)
        loaded = WordMap()
        db = anydbm.open(filename, 'c')
        for key in db:
            self.wmap.update( { key : db[key] })
            self.rwmap.update( { db[key] : key })
        db.close()
        print ("Loaded %d words into WordMap." % len(self.wmap))
        
