import pickle

class WordMap(object):
    
    def __init__(self):
        self.count = 0
        self.map   = {}
        self.rmap  = {}

    def apply(self, word):
        if word not in self.map:
            self.map.update({ word : self.count })
            self.rmap.update({ str(self.count) : word })
            self.count += 1
        return self.map[word]

    def demap(self, mapped_word):
        return self.rmap[mapped_word]

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))
    
    @classmethod
    def load(filename):
        return pickle.load(filename)
        
