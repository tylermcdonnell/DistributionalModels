######################################################################################
'''
@author: Tyler McDonnell / http://tylermcdonnell.com

Hooks for creating pairs of synonyms, antonyms, and unrelated words.

Initially conceived to evaluate synonym and antonym detection in distributional models.
'''
# Standard
import sys
import thesaurus
import random
from collections import namedtuple, Set
######################################################################################  

def generate_pairs(seed, pairs):
    Pair = namedtuple('Pair', ['relationship', 'w1', 'w2'])
    for word in seed:
        result = thesaurus.search(word)
        # Only use words of higher relevance rating.
        synonyms = [s for s in result.synonyms if s.relevance == 3]
        antonyms = [a for a in result.antonyms if a.relevance == 3]
        # Randomly sample; else in alphabetical.
        synonyms = random.sample(synonyms, min(pairs, len(synonyms)))
        antonyms = random.sample(antonyms, min(pairs, len(antonyms)))
        for s in synonyms:
            yield Pair(relationship="SYNONYMS", w1=word, w2=s)
        for a in antonyms:
            yield Pair(relationship="ANTONYMS", w1=word, w2=a)

if __name__ == "__main__":
    seed = [line.rstrip('\n') for line in open('seed.txt')]
    all_words = set()
    count = 0
    for pair in generate_pairs(seed, 10):
        print ('%s - { %s : %s }' % (pair.relationship, pair.w1, pair.w2))
        count += 2
        all_words.add(pair.w1)
        all_words.add(pair.w2)
    print len(all_words)
    print (count)
